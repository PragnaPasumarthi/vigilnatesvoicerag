"""Profile pipeline.query directly."""
import sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

from voice_rag.dataset import load_msmarco_xi
from voice_rag.vector_store import WordHashEmbedding, VectorStore, RetrievalConfig
from voice_rag.guardrails import Guardrails, GuardrailConfig, GuardrailVerdict
from voice_rag.pipeline import RAGPipeline, RAGConfig
from voice_rag.chunking import chunk_document, ChunkingConfig

dataset = load_msmarco_xi("data/msmarco_xi_sample.json")
passage_texts = dataset.get_all_passage_texts()

config = RAGConfig(embedding_fn=WordHashEmbedding(dim=128))
pipeline = RAGPipeline(config=config)
pipeline.ingest_documents(passage_texts)

query = dataset.examples[0].query
print(f"Query: {query}")

# Profile pipeline.query
for trial in range(5):
    t0 = time.perf_counter()
    response = pipeline.query(query)
    total = (time.perf_counter() - t0) * 1000
    lat = response.latency
    print(f"\nTrial {trial+1}: total={total:.2f}ms")
    for k, v in lat.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.2f}ms")

# Also profile raw components with harness
print("\n--- Raw component timing (no harness) ---")
guardrails = Guardrails(GuardrailConfig())
guardrails.update_topic_index_from_documents(passage_texts)

for trial in range(5):
    t0 = time.perf_counter()
    g_in = guardrails.check_input(query)
    t1 = time.perf_counter()
    sr = pipeline._vector_store.query(query, RetrievalConfig(top_k=10))
    t2 = time.perf_counter()
    g_out = guardrails.check_output(
        sr[0].chunk.content if sr else "no answer",
        [s.chunk.content[:200] for s in sr],
        [s.score for s in sr],
    )
    t3 = time.perf_counter()
    print(f"Trial {trial+1}: guard_in={(t1-t0)*1000:.2f}ms  retrieval={(t2-t1)*1000:.2f}ms  guard_out={(t3-t2)*1000:.2f}ms  total={(t3-t0)*1000:.2f}ms")
