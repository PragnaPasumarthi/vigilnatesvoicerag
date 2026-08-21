"""
Main RAG pipeline orchestrating all components.

Pipeline flow:
1. Ingest documents → chunk → embed → store in vector DB
2. (Optional) Transcribe audio → query text
3. Guardrails check on input
4. Retrieve relevant chunks
5. Generate answer (or return retrieved context)
6. Guardrails check on output
7. Return structured result with latency metrics
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from voice_rag.chunking import (
    Chunk,
    ChunkingConfig,
    chunk_document,
)
from voice_rag.guardrails import (
    GuardrailConfig,
    GuardrailResult,
    GuardrailVerdict,
    Guardrails,
)
from voice_rag.harness import (
    PipelineHarness,
    PipelineResult,
    RetryConfig,
    StepResult,
    StructuredIO,
)
from voice_rag.stt import (
    ElevenLabsSTT,
    MockSTT,
    TranscriptionResult,
    create_stt,
)
from voice_rag.vector_store import (
    EmbeddingFunction,
    RetrievalConfig,
    SearchResult,
    VectorStore,
    get_default_embedding_function,
)


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""
    # Chunking
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    chunking_strategies: Optional[list[str]] = None

    # Retrieval
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # Guardrails
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)

    # Retry
    retry: RetryConfig = field(default_factory=RetryConfig)

    # Embedding
    embedding_fn: Optional[EmbeddingFunction] = None


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    answer: str
    sources: list[dict]
    confidence: float
    guardrail_result: Optional[GuardrailResult] = None
    is_refusal: bool = False
    pipeline_result: Optional[PipelineResult] = None
    retrieval_results: Optional[list[SearchResult]] = None
    latency: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "is_refusal": self.is_refusal,
            "latency": self.latency,
            "pipeline": self.pipeline_result.to_dict() if self.pipeline_result else None,
        }


class RAGPipeline:
    """
    Voice-enabled RAG pipeline.

    Orchestrates: STT → Chunking → Retrieval → Answer Generation → Guardrails
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        stt: Optional[ElevenLabsSTT | MockSTT] = None,
        answer_fn: Optional[Callable[[str, list[Chunk]], str]] = None,
    ):
        self._config = config or RAGConfig()
        self._stt = stt
        self._answer_fn = answer_fn or self._default_answer_fn

        # Initialize components
        self._vector_store = VectorStore(self._config.embedding_fn)
        self._guardrails = Guardrails(self._config.guardrails)
        self._harness = PipelineHarness(self._config.retry)

        # State
        self._is_ingested = False
        self._all_document_text = ""

    def ingest_documents(self, documents: list[str]) -> PipelineResult:
        """
        Ingest documents: chunk, embed, and store.

        Args:
            documents: List of document texts

        Returns:
            PipelineResult with ingestion metrics
        """
        self._harness.reset()

        # Step 1: Chunk documents
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self._harness.run(
                "chunking",
                chunk_document,
                doc,
                self._config.chunking,
                self._config.chunking_strategies,
            )
            if chunks.success and chunks.output:
                all_chunks.extend(chunks.output)

        # Step 2: Store in vector DB
        if all_chunks:
            self._harness.run(
                "embedding_and_indexing",
                self._vector_store.add_chunks,
                all_chunks,
            )

        # Step 3: Update guardrails topic index
        self._all_document_text = "\n\n".join(documents)
        self._guardrails.update_topic_index(self._all_document_text)

        self._is_ingested = True
        return self._harness.build_result()

    def query(
        self,
        text: str,
        return_sources: bool = True,
    ) -> RAGResponse:
        """
        Query the RAG pipeline with text.

        Args:
            text: The query text
            return_sources: Whether to include source chunks

        Returns:
            RAGResponse with answer, sources, and metrics
        """
        self._harness.reset()
        latency: dict[str, float] = {}

        # Step 1: Validate input
        valid, msg = StructuredIO.validate_query(text)
        if not valid:
            return RAGResponse(
                answer=f"Invalid query: {msg}",
                sources=[],
                confidence=0.0,
                is_refusal=True,
                latency={"validation_ms": 0},
            )

        # Step 2: Input guardrails
        t0 = time.perf_counter()
        input_check = self._harness.run(
            "input_guardrails",
            self._guardrails.check_input,
            text,
        )
        latency["guardrails_input_ms"] = input_check.duration_ms

        if input_check.success and input_check.output:
            result: GuardrailResult = input_check.output
            if result.verdict == GuardrailVerdict.BLOCK:
                return RAGResponse(
                    answer=self._guardrails.get_refusal_message(result),
                    sources=[],
                    confidence=0.0,
                    guardrail_result=result,
                    is_refusal=True,
                    pipeline_result=self._harness.build_result(),
                    latency=latency,
                )

        # Step 3: Retrieve relevant chunks
        retrieval = self._harness.run(
            "retrieval",
            self._vector_store.query,
            text,
            self._config.retrieval,
        )
        latency["retrieval_ms"] = retrieval.duration_ms

        search_results: list[SearchResult] = retrieval.output if retrieval.success and retrieval.output else []

        if not search_results:
            return RAGResponse(
                answer=self._guardrails.get_refusal_message(
                    GuardrailResult(
                        verdict=GuardrailVerdict.BLOCK,
                        reason="No relevant context found in retrieval",
                    )
                ),
                sources=[],
                confidence=0.0,
                is_refusal=True,
                pipeline_result=self._harness.build_result(),
                latency=latency,
            )

        # Step 4: Generate answer
        retrieved_chunks = [sr.chunk for sr in search_results]
        retrieval_scores = [sr.score for sr in search_results]

        answer_gen = self._harness.run(
            "answer_generation",
            self._answer_fn,
            text,
            retrieved_chunks,
        )
        latency["generation_ms"] = answer_gen.duration_ms

        answer = answer_gen.output if answer_gen.success and answer_gen.output else "Failed to generate answer."

        # Step 5: Output guardrails
        output_check = self._harness.run(
            "output_guardrails",
            self._guardrails.check_output,
            answer,
            [c.content for c in retrieved_chunks],
            retrieval_scores,
        )
        latency["guardrails_output_ms"] = output_check.duration_ms

        if output_check.success and output_check.output:
            out_result: GuardrailResult = output_check.output
            if out_result.verdict == GuardrailVerdict.BLOCK:
                return RAGResponse(
                    answer=self._guardrails.get_refusal_message(out_result),
                    sources=[],
                    confidence=0.0,
                    guardrail_result=out_result,
                    is_refusal=True,
                    retrieval_results=search_results,
                    pipeline_result=self._harness.build_result(),
                    latency=latency,
                )

        # Step 6: Format output
        confidence = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0

        sources = []
        if return_sources:
            for sr in search_results:
                sources.append({
                    "chunk_index": sr.chunk.index,
                    "strategy": sr.chunk.strategy,
                    "score": round(sr.score, 4),
                    "content_preview": sr.chunk.content[:200],
                    "metadata": sr.chunk.metadata,
                })

        latency["total_ms"] = sum(latency.values())

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            guardrail_result=output_check.output if output_check.success else None,
            is_refusal=False,
            retrieval_results=search_results,
            pipeline_result=self._harness.build_result(),
            latency=latency,
        )

    def voice_query(self, audio_bytes: bytes) -> RAGResponse:
        """
        Full voice query: transcribe → query pipeline.

        Args:
            audio_bytes: Raw audio data

        Returns:
            RAGResponse with answer and full metrics
        """
        self._harness.reset()
        latency: dict[str, float] = {}

        # Step 1: Transcribe
        if self._stt is None:
            return RAGResponse(
                answer="No speech-to-text service configured.",
                sources=[],
                confidence=0.0,
                is_refusal=True,
                latency={},
            )

        transcription = self._harness.run(
            "transcription",
            self._stt.transcribe,
            audio_bytes,
        )
        latency["stt_ms"] = transcription.duration_ms

        if not transcription.success or not transcription.output:
            return RAGResponse(
                answer=f"Transcription failed: {transcription.error}",
                sources=[],
                confidence=0.0,
                is_refusal=True,
                pipeline_result=self._harness.build_result(),
                latency=latency,
            )

        query_text = transcription.output.text
        if not query_text.strip():
            return RAGResponse(
                answer="No speech detected in the audio.",
                sources=[],
                confidence=0.0,
                is_refusal=True,
                latency=latency,
            )

        # Step 2: Run text query pipeline
        text_response = self.query(query_text)

        # Merge latencies
        text_response.latency = {**latency, **text_response.latency}
        text_response.latency["total_ms"] = sum(
            v for v in text_response.latency.values() if isinstance(v, (int, float))
        )

        return text_response

    @staticmethod
    def _default_answer_fn(query: str, chunks: list[Chunk]) -> str:
        """
        Default answer generation: extractive approach.
        Returns the most relevant chunk content as the answer.
        In production, this would call an LLM.
        """
        if not chunks:
            return "No relevant information found."

        # Simple extractive: return the top chunk with query highlighting
        best = chunks[0]
        answer = best.content

        # Add context from additional chunks if available
        if len(chunks) > 1:
            additional = chunks[1].content
            # Truncate if too long
            if len(answer) + len(additional) < 1000:
                answer = f"{answer}\n\nAdditionally: {additional}"

        return answer

    @property
    def vector_store_stats(self) -> dict:
        """Get vector store statistics."""
        return self._vector_store.stats
