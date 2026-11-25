"""Unit tests for Milestone 4 local RAG technical manual search (scikit-learn + FAISS)."""

import unittest

from src.analysis import analyze_incident
from src.config import SAMPLE_DATA_DIR, load_config
from src.ingestion import DocumentPage, ingest_files
from src.rag import ManualRAGIndex, build_manual_index, chunk_pages


class TestRAGModule(unittest.TestCase):
    """Verify chunking, FAISS indexing, page citations, and incident-driven retrieval."""

    @classmethod
    def setUpClass(cls) -> None:
        files = [
            SAMPLE_DATA_DIR / "server_metrics.csv",
            SAMPLE_DATA_DIR / "app_events.log",
            SAMPLE_DATA_DIR / "incident_summary_ticket.txt",
            SAMPLE_DATA_DIR / "technical_manual.pdf",
        ]
        cls.config = load_config()
        cls.ingestion = ingest_files(files)
        cls.analysis = analyze_incident(cls.ingestion, config=cls.config)
        cls.rag_index = build_manual_index(cls.ingestion.manual_pages, config=cls.config)

    def test_manual_chunking_and_citations(self) -> None:
        self.assertTrue(self.rag_index.is_ready)
        self.assertGreaterEqual(self.rag_index.chunk_count, 3)
        self.assertEqual(self.rag_index.vector_mode, "tfidf")

        hits = self.rag_index.search("ConnectionPoolExhausted idle-in-transaction db-proxy")
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.page_number, 1)
        self.assertEqual(hits[0].citation, "technical_manual.pdf (p. 1)")
        self.assertGreater(hits[0].score, 0.0)

    def test_page_specific_queries(self) -> None:
        p2_hits = self.rag_index.search("504 Gateway Timeout circuit breaker half-open")
        self.assertGreaterEqual(len(p2_hits), 1)
        self.assertEqual(p2_hits[0].chunk.page_number, 2)

        p3_hits = self.rag_index.search("nightly_analytics_sync read-replica verification checklist")
        self.assertGreaterEqual(len(p3_hits), 1)
        self.assertEqual(p3_hits[0].chunk.page_number, 3)

    def test_incident_driven_rag_retrieval(self) -> None:
        hits = self.rag_index.search_for_incident(self.analysis, top_k=3)
        self.assertGreaterEqual(len(hits), 2)
        cited_pages = {hit.chunk.page_number for hit in hits}
        # Expect both DB pool runbook (p. 1) and API gateway 504 runbook (p. 2) to surface
        self.assertTrue({1, 2}.issubset(cited_pages))

    def test_svd_lsa_mode_on_larger_corpus(self) -> None:
        synthetic_pages = [
            DocumentPage(
                source_file="ops_guide.pdf",
                page_number=i,
                text=(
                    f"Section {i}: Kubernetes pod eviction OOMKilled memory pressure limits."
                    if i % 3 == 0
                    else f"Section {i}: PostgreSQL connection pool latency timeout vacuuming tuning."
                ),
            )
            for i in range(1, 26)
        ]
        index = ManualRAGIndex(synthetic_pages, config=self.config)
        self.assertEqual(index.vector_mode, "tfidf+svd")

        hits = index.search("Kubernetes pod OOMKilled memory pressure", top_k=2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("OOMKilled", hits[0].chunk.text)

    def test_empty_manual_and_unknown_query_safety(self) -> None:
        empty_index = build_manual_index([], config=self.config)
        self.assertFalse(empty_index.is_ready)
        self.assertEqual(empty_index.search("anything"), [])
        self.assertEqual(self.rag_index.search("zzzzunlikelytokenxyz"), [])
        self.assertEqual(chunk_pages([DocumentPage("blank.pdf", 1, "   ")]), [])


if __name__ == "__main__":
    unittest.main()
