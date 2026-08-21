"""
Benchmark harness for measuring pipeline latency.

Measures P50/P70/P100 latency across multiple test queries,
with per-step breakdown and warm-up runs.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

from voice_rag.pipeline import RAGConfig, RAGPipeline, RAGResponse


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    num_queries: int = 50
    warmup_queries: int = 5
    report_percentiles: list[int] = field(default_factory=lambda: [50, 70, 90, 95, 100])


@dataclass
class LatencyStats:
    """Latency statistics for a single metric."""
    p50: float = 0.0
    p70: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p100: float = 0.0
    mean: float = 0.0
    stdev: float = 0.0
    min: float = 0.0
    max: float = 0.0
    count: int = 0

    def to_dict(self) -> dict:
        return {
            "p50_ms": round(self.p50, 2),
            "p70_ms": round(self.p70, 2),
            "p90_ms": round(self.p90, 2),
            "p95_ms": round(self.p95, 2),
            "p100_ms": round(self.p100, 2),
            "mean_ms": round(self.mean, 2),
            "stdev_ms": round(self.stdev, 2),
            "min_ms": round(self.min, 2),
            "max_ms": round(self.max, 2),
            "count": self.count,
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    total_latency: LatencyStats = field(default_factory=LatencyStats)
    chunking_latency: LatencyStats = field(default_factory=LatencyStats)
    retrieval_latency: LatencyStats = field(default_factory=LatencyStats)
    generation_latency: LatencyStats = field(default_factory=LatencyStats)
    guardrails_input_latency: LatencyStats = field(default_factory=LatencyStats)
    guardrails_output_latency: LatencyStats = field(default_factory=LatencyStats)
    num_queries: int = 0
    num_successful: int = 0
    target_met: bool = False
    query_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "num_queries": self.num_queries,
                "num_successful": self.num_successful,
                "target_200ms_met": self.target_met,
            },
            "total_latency": self.total_latency.to_dict(),
            "chunking_latency": self.chunking_latency.to_dict(),
            "retrieval_latency": self.retrieval_latency.to_dict(),
            "generation_latency": self.generation_latency.to_dict(),
            "guardrails_input_latency": self.guardrails_input_latency.to_dict(),
            "guardrails_output_latency": self.guardrails_output_latency.to_dict(),
        }


def _compute_stats(values: list[float]) -> LatencyStats:
    """Compute latency statistics from a list of values."""
    if not values:
        return LatencyStats()

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    return LatencyStats(
        p50=sorted_vals[int(n * 0.5)] if n > 0 else 0,
        p70=sorted_vals[int(n * 0.7)] if n > 0 else 0,
        p90=sorted_vals[int(n * 0.9)] if n > 0 else 0,
        p95=sorted_vals[int(n * 0.95)] if n > 0 else 0,
        p100=sorted_vals[-1] if n > 0 else 0,
        mean=statistics.mean(values),
        stdev=statistics.stdev(values) if n > 1 else 0,
        min=sorted_vals[0],
        max=sorted_vals[-1],
        count=n,
    )


class BenchmarkHarness:
    """
    Runs benchmarks on the RAG pipeline with multiple queries.
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        config: Optional[BenchmarkConfig] = None,
    ):
        self._pipeline = pipeline
        self._config = config or BenchmarkConfig()

    def run(self, queries: list[str]) -> BenchmarkResult:
        """
        Run benchmarks across all queries.

        Args:
            queries: List of test queries

        Returns:
            BenchmarkResult with latency statistics
        """
        result = BenchmarkResult()

        # Warm up
        print(f"[benchmark] Warming up with {self._config.warmup_queries} queries...")
        for i in range(min(self._config.warmup_queries, len(queries))):
            self._pipeline.query(queries[i % len(queries)])

        # Run benchmark queries
        total_latencies: list[float] = []
        chunking_latencies: list[float] = []
        retrieval_latencies: list[float] = []
        generation_latencies: list[float] = []
        guardrails_input_latencies: list[float] = []
        guardrails_output_latencies: list[float] = []
        query_results: list[dict] = []

        num_queries = min(self._config.num_queries, len(queries))

        print(f"[benchmark] Running {num_queries} benchmark queries...")

        for i in range(num_queries):
            query = queries[i % len(queries)]

            t0 = time.perf_counter()
            response = self._pipeline.query(query)
            total_ms = (time.perf_counter() - t0) * 1000

            # Extract per-step latencies
            latency = response.latency
            total_latencies.append(total_ms)
            chunking_latencies.append(latency.get("chunking_ms", 0))
            retrieval_latencies.append(latency.get("retrieval_ms", 0))
            generation_latencies.append(latency.get("generation_ms", 0))
            guardrails_input_latencies.append(latency.get("guardrails_input_ms", 0))
            guardrails_output_latencies.append(latency.get("guardrails_output_ms", 0))

            query_results.append({
                "query": query[:100],
                "total_ms": round(total_ms, 2),
                "is_refusal": response.is_refusal,
                "confidence": response.confidence,
            })

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_queries} queries")

        # Compute statistics
        result.total_latency = _compute_stats(total_latencies)
        result.chunking_latency = _compute_stats(chunking_latencies)
        result.retrieval_latency = _compute_stats(retrieval_latencies)
        result.generation_latency = _compute_stats(generation_latencies)
        result.guardrails_input_latency = _compute_stats(guardrails_input_latencies)
        result.guardrails_output_latency = _compute_stats(guardrails_output_latencies)
        result.num_queries = num_queries
        result.num_successful = sum(1 for r in query_results if not r.get("error"))
        result.target_met = result.total_latency.p50 <= 200
        result.query_results = query_results

        return result


def print_benchmark_report(result: BenchmarkResult) -> None:
    """Print a formatted benchmark report."""
    print("\n" + "=" * 70)
    print("  RAG PIPELINE BENCHMARK REPORT")
    print("=" * 70)

    print(f"\n  Queries: {result.num_queries} "
          f"(successful: {result.num_successful})")
    print(f"  200ms Target: {'[MET]' if result.target_met else '[NOT MET]'}")

    print("\n  Latency Breakdown (ms):")
    print("  " + "-" * 66)
    print(f"  {'Component':<30} {'P50':>8} {'P70':>8} {'P100':>8} {'Mean':>8}")
    print("  " + "-" * 66)

    stats = [
        ("Total (end-to-end)", result.total_latency),
        ("Chunking", result.chunking_latency),
        ("Retrieval", result.retrieval_latency),
        ("Answer Generation", result.generation_latency),
        ("Guardrails (input)", result.guardrails_input_latency),
        ("Guardrails (output)", result.guardrails_output_latency),
    ]

    for name, s in stats:
        print(f"  {name:<30} {s.p50:>8.1f} {s.p70:>8.1f} {s.p100:>8.1f} {s.mean:>8.1f}")

    print("  " + "-" * 66)

    # Detailed stats
    print(f"\n  Detailed Total Latency:")
    d = result.total_latency.to_dict()
    print(f"    Min: {d['min_ms']:.1f}ms")
    print(f"    P50: {d['p50_ms']:.1f}ms")
    print(f"    P70: {d['p70_ms']:.1f}ms")
    print(f"    P90: {d['p90_ms']:.1f}ms")
    print(f"    P95: {d['p95_ms']:.1f}ms")
    print(f"    P100: {d['p100_ms']:.1f}ms")
    print(f"    Mean: {d['mean_ms']:.1f}ms")
    print(f"    Stdev: {d['stdev_ms']:.1f}ms")

    print("\n" + "=" * 70)


def main():
    """CLI entry point for benchmarking."""
    import json

    from voice_rag.chunking import ChunkingConfig
    from voice_rag.vector_store import HashEmbedding

    # Sample test queries
    test_queries = [
        "What is the capital of France?",
        "Explain machine learning in simple terms.",
        "How do neural networks work?",
        "What are the benefits of cloud computing?",
        "Describe the process of data preprocessing.",
        "What is natural language processing?",
        "How does Python handle memory management?",
        "Explain the concept of overfitting.",
        "What are the types of machine learning?",
        "How do you evaluate a classification model?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain gradient descent algorithm.",
        "What are transformers in deep learning?",
        "How does attention mechanism work?",
        "What is transfer learning?",
        "Describe the vanishing gradient problem.",
        "What is batch normalization?",
        "Explain convolutional neural networks.",
        "What is reinforcement learning?",
        "How do you handle missing data?",
    ] * 5  # Repeat for more queries

    # Sample documents
    sample_docs = [
        """
        Machine Learning Fundamentals

        Machine learning is a subset of artificial intelligence that provides systems
        the ability to automatically learn and improve from experience without being
        explicitly programmed. ML focuses on the development of computer programs that
        can access data and use it to learn for themselves.

        The process of learning begins with observations or data, such as examples,
        direct experience, or instruction, in order to look for patterns in data and
        make better decisions in the future based on the examples that we provide.

        Types of Machine Learning:
        1. Supervised Learning: The algorithm learns from labeled training data.
        2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
        3. Reinforcement Learning: The algorithm learns by interacting with an environment.

        Key Concepts:
        - Training Data: The dataset used to train the model.
        - Features: Individual measurable properties of the data.
        - Labels: The output variables we're trying to predict.
        - Model: The mathematical representation of patterns in the data.
        """,
        """
        Neural Networks and Deep Learning

        Neural networks are computing systems inspired by biological neural networks
        in the brain. They consist of layers of interconnected nodes (neurons) that
        process information using connectionist approaches.

        Architecture:
        - Input Layer: Receives the raw data
        - Hidden Layers: Process the data through weighted connections
        - Output Layer: Produces the final result

        Deep Learning:
        Deep learning is a subset of machine learning that uses neural networks with
        multiple layers (hence "deep"). These networks can learn hierarchical
        representations of data.

        Key architectures:
        1. Convolutional Neural Networks (CNNs): For image processing
        2. Recurrent Neural Networks (RNNs): For sequential data
        3. Transformers: For attention-based processing
        4. Generative Adversarial Networks (GANs): For data generation

        Training Techniques:
        - Backpropagation: Computing gradients for weight updates
        - Gradient Descent: Optimizing the loss function
        - Batch Normalization: Stabilizing training
        - Dropout: Preventing overfitting
        """,
        """
        Data Science and Analytics

        Data science is an interdisciplinary field that uses scientific methods,
        processes, algorithms, and systems to extract knowledge and insights from
        structured and unstructured data.

        Data Preprocessing:
        1. Data Cleaning: Handling missing values, removing duplicates
        2. Feature Engineering: Creating new features from existing ones
        3. Feature Selection: Choosing the most relevant features
        4. Data Transformation: Normalization, scaling, encoding

        Evaluation Metrics:
        - Classification: Accuracy, Precision, Recall, F1-score
        - Regression: MSE, RMSE, MAE, R-squared
        - Clustering: Silhouette score, Davies-Bouldin index

        Tools and Technologies:
        - Python: pandas, scikit-learn, TensorFlow, PyTorch
        - R: dplyr, ggplot2, caret
        - SQL: Data querying and manipulation
        - Big Data: Hadoop, Spark for large-scale processing
        """,
        """
        Cloud Computing and Deployment

        Cloud computing delivers computing services over the internet, providing
        on-demand access to resources like servers, storage, databases, and
        applications.

        Service Models:
        1. Infrastructure as a Service (IaaS): Virtual machines, storage
        2. Platform as a Service (PaaS): Development platforms
        3. Software as a Service (SaaS): Ready-to-use applications

        Benefits:
        - Cost Efficiency: Pay only for what you use
        - Scalability: Scale resources up or down as needed
        - Reliability: Built-in redundancy and backup
        - Global Reach: Deploy applications worldwide

        ML Deployment:
        - Model Serving: REST APIs for real-time inference
        - Batch Processing: Scheduled predictions on large datasets
        - Edge Computing: Running models on edge devices
        - A/B Testing: Comparing model versions in production

        Best Practices:
        - Containerization with Docker and Kubernetes
        - CI/CD pipelines for automated deployment
        - Model monitoring and drift detection
        - Version control for models and data
        """,
    ]

    # Create pipeline
    print("[main] Setting up pipeline...")
    config = RAGConfig(
        embedding_fn=HashEmbedding(dim=384),
    )
    pipeline = RAGPipeline(config=config)

    # Ingest documents
    print("[main] Ingesting documents...")
    ingest_result = pipeline.ingest_documents(sample_docs)
    print(f"[main] Ingested {len(sample_docs)} documents")
    print(f"[main] Vector store: {pipeline.vector_store_stats}")

    # Run benchmarks
    bench_config = BenchmarkConfig(
        num_queries=50,
        warmup_queries=5,
    )
    harness = BenchmarkHarness(pipeline, bench_config)

    print("\n[main] Running benchmarks...")
    bench_result = harness.run(test_queries)

    # Print report
    print_benchmark_report(bench_result)

    # Save results
    output_path = "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(bench_result.to_dict(), f, indent=2)
    print(f"\n[main] Results saved to {output_path}")


if __name__ == "__main__":
    main()
