"""Profile the RAG query path to find latency bottlenecks."""
import sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

from voice_rag.dataset import load_msmarco_xi
from voice_rag.vector_store import WordHashEmbedding, VectorStore, RetrievalConfig
from voice_rag.guardrails import Guardrails, GuardrailConfig

dataset = load_msmarco_xi("data/msmarco_xi_sample.json")
passage_texts = dataset.get_all_passage_texts()

# Build store
emb_fn = WordHashEmbedding(dim=128)
store = VectorStore(emb_fn)
from voice_rag.chunking import chunk_document, ChunkingConfig
all_chunks = []
for text in passage_texts:
    all_chunks.extend(chunk_document(text))
store.add_chunks(all_chunks)
print(f"Store: {len(store._chunks)} chunks")

# Setup guardrails
guardrails = Guardrails(GuardrailConfig())
guardrails.update_topic_index_from_documents(passage_texts)

query = dataset.examples[0].query
print(f"Query: {query}")

# Profile each step
for trial in range(3):
    print(f"\n--- Trial {trial+1} ---")

    t0 = time.perf_counter()
    # Step 1: Input guardrails
    g_in = guardrails.check_input(query)
    t1 = time.perf_counter()
    print(f"  Guardrails input:  {(t1-t0)*1000:.2f}ms")

    # Step 2: Embed query
    t2 = time.perf_counter()
    q_emb = emb_fn([query])
    t3 = time.perf_counter()
    print(f"  Query embedding:   {(t3-t2)*1000:.2f}ms")

    # Step 3: Cosine similarity
    t4 = time.perf_counter()
    scores = store._vectors @ q_emb[0]
    t5 = time.perf_counter()
    print(f"  Cosine similarity: {(t5-t4)*1000:.2f}ms (matrix: {store._vectors.shape})")

    # Step 4: Sort + top-k
    t6 = time.perf_counter()
    top_idx = np.argsort(-scores)[:10]
    t7 = time.perf_counter()
    print(f"  Top-k selection:   {(t7-t6)*1000:.2f}ms")

    # Step 5: Build results
    t8 = time.perf_counter()
    results = [store._chunks[int(idx)] for idx in top_idx]
    t9 = time.perf_counter()
    print(f"  Build results:     {(t9-t8)*1000:.2f}ms")

    # Step 6: Output guardrails
    t10 = time.perf_counter()
    g_out = guardrails.check_output("test answer", [r.content[:200] for r in results], [float(scores[idx]) for idx in top_idx])
    t11 = time.perf_counter()
    print(f"  Guardrails output: {(t11-t10)*1000:.2f}ms")

    total = (t11-t0)*1000
    print(f"  TOTAL:             {total:.2f}ms")
