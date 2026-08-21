"""
CLI interface for the Voice RAG system.

Commands:
  ingest   - Ingest documents into the vector store
  query    - Query the pipeline with text
  voice    - Query with audio file (requires ElevenLabs API key)
  bench    - Run latency benchmarks
  demo     - Run end-to-end demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from voice_rag.chunking import ChunkingConfig, chunk_document
from voice_rag.pipeline import RAGConfig, RAGPipeline
from voice_rag.stt import create_stt
from voice_rag.vector_store import HashEmbedding


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest documents from file or directory."""
    documents: list[str] = []

    path = Path(args.path)
    if path.is_file():
        documents.append(path.read_text(encoding="utf-8"))
    elif path.is_dir():
        for f in path.rglob("*"):
            if f.is_file() and f.suffix in {".txt", ".md", ".json"}:
                try:
                    documents.append(f.read_text(encoding="utf-8"))
                except Exception:
                    print(f"[warn] Skipping {f}: read error")
    else:
        print(f"[error] Path not found: {path}")
        sys.exit(1)

    print(f"[ingest] Found {len(documents)} documents")

    # Show chunking strategies
    print("[ingest] Running chunking strategies...")
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        strategies = set(c.strategy for c in chunks)
        print(f"  Document: {len(chunks)} chunks from strategies: {strategies}")

    print(f"[ingest] Total chunks: {len(all_chunks)}")

    # Save chunks if requested
    if args.output:
        output = [
            {
                "content": c.content,
                "index": c.index,
                "strategy": c.strategy,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "metadata": c.metadata,
            }
            for c in all_chunks
        ]
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[ingest] Chunks saved to {args.output}")


def cmd_query(args: argparse.Namespace) -> None:
    """Query the pipeline with text."""
    # For quick demo, use hash embeddings
    config = RAGConfig(embedding_fn=HashEmbedding(dim=384))
    pipeline = RAGPipeline(config=config)

    # Ingest documents if provided
    if args.documents:
        docs = [Path(p).read_text(encoding="utf-8") for p in args.documents]
        pipeline.ingest_documents(docs)
        print(f"[query] Ingested {len(docs)} documents")

    # Query
    t0 = time.perf_counter()
    response = pipeline.query(args.text)
    total_ms = (time.perf_counter() - t0) * 1000

    print(f"\n{'='*60}")
    print(f"Query: {args.text}")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{response.answer}")
    print(f"\nConfidence: {response.confidence:.3f}")
    print(f"Is Refusal: {response.is_refusal}")

    if response.sources:
        print(f"\nSources ({len(response.sources)}):")
        for i, src in enumerate(response.sources[:3]):
            print(f"  [{i+1}] (score: {src['score']:.3f}, strategy: {src['strategy']})")
            print(f"      {src['content_preview'][:100]}...")

    print(f"\nLatency: {json.dumps({k: f'{v:.1f}ms' for k, v in response.latency.items() if isinstance(v, (int, float))}, indent=2)}")


def cmd_voice(args: argparse.Namespace) -> None:
    """Query with audio file."""
    api_key = args.api_key
    if not api_key:
        print("[error] ElevenLabs API key required (--api-key)")
        sys.exit(1)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[error] Audio file not found: {audio_path}")
        sys.exit(1)

    audio_bytes = audio_path.read_bytes()

    # Setup
    stt = create_stt(api_key=api_key)
    config = RAGConfig(embedding_fn=HashEmbedding(dim=384))
    pipeline = RAGPipeline(config=config, stt=stt)

    # Ingest if documents provided
    if args.documents:
        docs = [Path(p).read_text(encoding="utf-8") for p in args.documents]
        pipeline.ingest_documents(docs)

    # Voice query
    t0 = time.perf_counter()
    response = pipeline.voice_query(audio_bytes)
    total_ms = (time.perf_counter() - t0) * 1000

    print(f"\n{'='*60}")
    print(f"Voice Query Result")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{response.answer}")
    print(f"\nLatency: {total_ms:.1f}ms")

    stt.close()


def cmd_bench(args: argparse.Namespace) -> None:
    """Run benchmarks."""
    from voice_rag.benchmark import (
        BenchmarkConfig,
        BenchmarkHarness,
        print_benchmark_report,
    )

    config = RAGConfig(embedding_fn=HashEmbedding(dim=384))
    pipeline = RAGPipeline(config=config)

    # Ingest sample docs if provided
    if args.documents:
        docs = [Path(p).read_text(encoding="utf-8") for p in args.documents]
    else:
        # Use built-in sample docs
        docs = [
            "Machine learning is a subset of artificial intelligence. "
            "It provides systems the ability to automatically learn and improve. "
            "Neural networks are computing systems inspired by biological neural networks. "
            "Deep learning uses neural networks with multiple layers. "
            "Data preprocessing includes cleaning, transformation, and feature engineering. "
            "Cloud computing provides on-demand access to computing resources. "
            "Natural language processing enables computers to understand human language. "
            "Supervised learning uses labeled training data. "
            "Unsupervised learning finds patterns in unlabeled data. "
            "Reinforcement learning learns by interacting with an environment.",
        ]

    pipeline.ingest_documents(docs)

    queries = args.queries.split("|") if args.queries else [
        "What is machine learning?",
        "How do neural networks work?",
        "What is deep learning?",
        "Explain data preprocessing.",
        "What are the benefits of cloud computing?",
    ] * 3

    bench_config = BenchmarkConfig(
        num_queries=args.num_queries,
        warmup_queries=args.warmup,
    )
    harness = BenchmarkHarness(pipeline, bench_config)
    result = harness.run(queries)
    print_benchmark_report(result)


def cmd_demo(args: argparse.Namespace) -> None:
    """Run end-to-end demo."""
    print("=" * 60)
    print("  VOICE RAG PIPELINE DEMO")
    print("=" * 60)

    # Sample knowledge base
    docs = [
        """
        Company Handbook - Engineering

        Our engineering team follows agile methodologies with two-week sprints.
        We use Python, TypeScript, and Go as our primary languages.
        Code reviews are mandatory before merging any pull request.
        We deploy to production using CI/CD pipelines with automated testing.
        Our infrastructure runs on AWS with Kubernetes orchestration.

        On-call rotations are managed through PagerDuty.
        Each engineer takes one week of on-call per quarter.
        Incident response follows the runbook in our internal wiki.
        """,
        """
        Product Documentation - Features

        The platform supports real-time data processing with sub-second latency.
        Key features include:
        - Natural language query interface
        - Automated report generation
        - Custom dashboard creation
        - API access for programmatic integration
        - Role-based access control

        The system processes over 1 million events per second.
        Data retention policy: 90 days for raw data, 2 years for aggregates.
        """,
        """
        Machine Learning Guide

        Our ML pipeline uses the following components:
        1. Data ingestion from multiple sources (Kafka, S3, APIs)
        2. Feature engineering with Apache Spark
        3. Model training with PyTorch and scikit-learn
        4. Model serving with TorchServe and custom REST APIs
        5. Monitoring with Prometheus and Grafana

        Model retraining happens weekly on fresh data.
        A/B testing is used to compare model versions in production.
        We track model drift using statistical tests on prediction distributions.
        """,
    ]

    print("\n[1/4] Ingesting documents...")
    config = RAGConfig(embedding_fn=HashEmbedding(dim=384))
    pipeline = RAGPipeline(config=config)
    result = pipeline.ingest_documents(docs)
    print(f"  Chunks indexed: {pipeline.vector_store_stats['total_chunks']}")
    print(f"  Ingestion time: {result.total_duration_ms:.1f}ms")

    print("\n[2/4] Running guardrails test...")
    # Test unsafe input
    unsafe_response = pipeline.query("How do I hack into a system?")
    print(f"  Unsafe query: '{unsafe_response.answer[:80]}...' (refusal={unsafe_response.is_refusal})")

    # Test off-topic
    off_topic_response = pipeline.query("What's the weather like on Mars?")
    print(f"  Off-topic query: '{off_topic_response.answer[:80]}...' (refusal={off_topic_response.is_refusal})")

    print("\n[3/4] Running sample queries...")
    queries = [
        "What programming languages does the team use?",
        "How does the ML pipeline work?",
        "What are the data retention policies?",
        "How do you handle code reviews?",
    ]

    for q in queries:
        response = pipeline.query(q)
        print(f"\n  Q: {q}")
        print(f"  A: {response.answer[:150]}...")
        print(f"  Confidence: {response.confidence:.3f}, Refusal: {response.is_refusal}")
        if response.sources:
            top = response.sources[0]
            print(f"  Top source: (score={top['score']:.3f}, strategy={top['strategy']})")

    print("\n[4/4] Latency summary...")
    total_times = []
    for q in queries:
        t0 = time.perf_counter()
        pipeline.query(q)
        total_times.append((time.perf_counter() - t0) * 1000)

    total_times.sort()
    n = len(total_times)
    print(f"  Queries: {n}")
    print(f"  P50: {total_times[n//2]:.1f}ms")
    print(f"  P70: {total_times[int(n*0.7)]:.1f}ms")
    print(f"  P100: {total_times[-1]:.1f}ms")
    print(f"  Mean: {sum(total_times)/n:.1f}ms")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Voice-enabled RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="File or directory to ingest")
    ingest_parser.add_argument("-o", "--output", help="Save chunks to JSON file")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the pipeline")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument("-d", "--documents", nargs="+", help="Document files to ingest first")

    # Voice command
    voice_parser = subparsers.add_parser("voice", help="Voice query")
    voice_parser.add_argument("audio", help="Audio file path")
    voice_parser.add_argument("--api-key", help="ElevenLabs API key")
    voice_parser.add_argument("-d", "--documents", nargs="+", help="Document files to ingest")

    # Benchmark command
    bench_parser = subparsers.add_parser("bench", help="Run benchmarks")
    bench_parser.add_argument("-d", "--documents", nargs="+", help="Document files to ingest")
    bench_parser.add_argument("-n", "--num-queries", type=int, default=50)
    bench_parser.add_argument("-w", "--warmup", type=int, default=5)
    bench_parser.add_argument("-q", "--queries", help="Pipe-separated queries")

    # Demo command
    subparsers.add_parser("demo", help="Run end-to-end demo")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "voice":
        cmd_voice(args)
    elif args.command == "bench":
        cmd_bench(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
