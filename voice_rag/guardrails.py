"""
Guardrails for the RAG pipeline.

Implements:
1. Off-topic query detection
2. Unsafe/inappropriate input filtering
3. Hallucination check (answer groundedness)
4. Answer confidence scoring
5. Refusal generation when guardrails trip
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GuardrailVerdict(Enum):
    """Result of a guardrail check."""
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class GuardrailResult:
    """Result of running guardrails on input or output."""
    verdict: GuardrailVerdict
    reason: str
    details: dict = field(default_factory=dict)


@dataclass
class GuardrailConfig:
    """Configuration for guardrails."""
    # Off-topic detection
    topic_keywords: list[str] = field(default_factory=list)
    off_topic_threshold: float = 0.004  # Catches near-zero overlap; recipe/food/sports blocked by patterns
    # Blocked topic categories (queries matching these patterns are off-topic)
    blocked_topic_patterns: list[str] = field(default_factory=lambda: [
        r'\b(recipe|cooking|bake|baking|ingredients|cook)\b',
        r'\b(football|soccer|basketball|baseball|cricket|tennis|score[sd]?|match)\b',
        r'\b(stock\s*(market|price|ticker|quote)|dividend|trading)\b',
        r'\b(weather|forecast|temperature|rain|sunny|humidity)\b',
        r'\b(horoscope|zodiac|astrology|fortune)\b',
    ])

    # Unsafe input detection
    blocked_patterns: list[str] = field(default_factory=lambda: [
        r'\b(hack|exploit|bomb|bombs?|weapon|weapons?|kill|murder|stab|shoot)\b',
        r'\b(drug\s*(deal|dealing|traffick|trafficking|sell|selling|manufacture)|sell\s*drugs?)\b',
        r'\b(make|build|create)\s+.*(weapon|bomb|gun|rifle|explosive)',
        r'\b(how\s+to\s+.*(harm|hurt|attack|kill|destroy))\b',
    ])

    # Hallucination check
    require_source_attribution: bool = True
    min_retrieval_score: float = 0.2
    max_unsupported_claims: int = 2

    # Refusal templates
    off_topic_refusal: str = (
        "I'm designed to answer questions related to the provided documents. "
        "Your question appears to be outside the scope of the available context. "
        "Could you rephrase or ask about a topic covered in the documents?"
    )
    unsafe_refusal: str = (
        "I can't assist with that request. Please ask a question related to "
        "the provided documents."
    )
    hallucination_refusal: str = (
        "I wasn't able to find sufficient information in the provided context "
        "to answer your question confidently. The available documents don't "
        "contain enough relevant information."
    )


# ---------------------------------------------------------------------------
# Guardrail checks
# ---------------------------------------------------------------------------

class OffTopicDetector:
    """Detects queries that are off-topic relative to the document corpus."""

    # Common English stop words that should not contribute to topic overlap
    STOP_WORDS: set[str] = {
        "the", "is", "at", "which", "on", "a", "an", "and", "or",
        "but", "in", "with", "to", "for", "of", "not", "no",
        "can", "had", "has", "have", "he", "she", "it", "its",
        "be", "are", "was", "were", "been", "do", "does", "did",
        "will", "would", "could", "should", "may", "might",
        "this", "that", "these", "those", "what", "how", "when",
        "where", "who", "whom", "why", "if", "then", "than",
        "about", "into", "through", "during", "before", "after",
        "above", "below", "from", "up", "down", "out", "off",
        "over", "under", "again", "further", "once", "here", "there",
        "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "only", "own", "same", "too",
        "very", "just", "because", "as", "until", "while", "also",
        "like", "need", "want", "tell", "give", "get", "make", "made",
        # Question framing verbs (should not affect topic detection)
        "explain", "describe", "define", "list", "name", "state",
        "mention", "enumerate", "provide", "identify", "discuss",
    }

    def __init__(self, config: GuardrailConfig):
        self._config = config
        self._topic_words: set[str] = set()
        self._doc_freq: dict[str, int] = {}  # word -> num docs containing it
        self._idf_weights: dict[str, float] = {}  # word -> IDF weight
        self._blocked_patterns = [
            re.compile(p, re.IGNORECASE) for p in config.blocked_topic_patterns
        ]
        for kw in config.topic_keywords:
            self._topic_words.update(kw.lower().split())

    def _content_words(self, text: str) -> set[str]:
        """Extract meaningful content words, filtering stop words."""
        words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        return words - self.STOP_WORDS

    def update_topic_index(self, all_text: str) -> None:
        """Build topic vocabulary from ingested documents."""
        words = self._content_words(all_text)
        self._topic_words.update(words)

    def update_topic_index_from_documents(self, documents: list[str]) -> None:
        """
        Build topic vocabulary with document-frequency weighting.
        Words in MANY documents get HIGHER weight (corpus-relevant).
        Words in FEW documents get LOWER weight (niche/off-topic).
        """
        n_docs = len(documents)
        self._doc_freq.clear()
        for doc in documents:
            for w in self._content_words(doc):
                self._doc_freq[w] = self._doc_freq.get(w, 0) + 1
        # Weight = df / n_docs (common words = high weight)
        self._idf_weights.clear()
        for word, df in self._doc_freq.items():
            self._idf_weights[word] = df / n_docs
        self._topic_words.update(self._doc_freq.keys())

    def _weighted_overlap_score(self, query_words: set[str]) -> tuple[float, list[str]]:
        """Compute document-frequency-weighted overlap score.
        Common words in the corpus get higher weight.
        """
        if not query_words or not self._idf_weights:
            return 0.0, []
        max_score = sum(self._idf_weights.get(w, 0.5) for w in query_words)
        if max_score == 0:
            return 0.0, []
        matched = []
        achieved = 0.0
        for w in query_words:
            if w in self._topic_words:
                weight = self._idf_weights.get(w, 0.5)
                achieved += weight
                matched.append(w)
        return achieved / max_score, matched

    def check(self, query: str) -> GuardrailResult:
        """Check if a query is on-topic using blocked patterns + IDF overlap."""
        # First check blocked topic patterns
        for pattern in self._blocked_patterns:
            if pattern.search(query):
                return GuardrailResult(
                    verdict=GuardrailVerdict.BLOCK,
                    reason=f"Query matches blocked topic pattern: {pattern.pattern}",
                    details={"pattern": pattern.pattern},
                )

        query_words = self._content_words(query)
        if not self._topic_words or not query_words:
            return GuardrailResult(
                verdict=GuardrailVerdict.PASS,
                reason="No topic index available, allowing query",
            )
        if self._idf_weights:
            score, matched = self._weighted_overlap_score(query_words)
        else:
            overlap = query_words & self._topic_words
            score = len(overlap) / len(query_words) if query_words else 0
            matched = list(overlap)

        # If no query words appear in the corpus at all, we can't judge topic relevance
        # (corpus may be too small). Allow the query through.
        if not matched and len(query_words) > 0:
            return GuardrailResult(
                verdict=GuardrailVerdict.PASS,
                reason="Insufficient corpus overlap to determine topic relevance",
                details={"overlap_score": score, "matched_words": []},
            )

        if score < self._config.off_topic_threshold:
            return GuardrailResult(
                verdict=GuardrailVerdict.BLOCK,
                reason=f"Query appears off-topic (weighted score: {score:.2f})",
                details={"overlap_score": score, "matched_words": matched[:5]},
            )
        return GuardrailResult(
            verdict=GuardrailVerdict.PASS,
            reason=f"Query is on-topic (weighted score: {score:.2f})",
            details={"overlap_score": score},
        )


class UnsafeInputDetector:
    """Detects unsafe or inappropriate inputs."""

    def __init__(self, config: GuardrailConfig):
        self._patterns = [
            re.compile(p, re.IGNORECASE) for p in config.blocked_patterns
        ]

    def check(self, query: str) -> GuardrailResult:
        """Check if a query contains unsafe content."""
        for pattern in self._patterns:
            match = pattern.search(query)
            if match:
                return GuardrailResult(
                    verdict=GuardrailVerdict.BLOCK,
                    reason=f"Unsafe content detected: '{match.group()}'",
                    details={"matched_pattern": pattern.pattern, "match": match.group()},
                )

        return GuardrailResult(
            verdict=GuardrailVerdict.PASS,
            reason="No unsafe content detected",
        )


class HallucinationChecker:
    """Checks if an answer is grounded in the retrieved context."""

    def __init__(self, config: GuardrailConfig):
        self._config = config

    def check(
        self,
        answer: str,
        retrieved_chunks: list[str],
        retrieval_scores: list[float],
    ) -> GuardrailResult:
        """Check if the answer is grounded in the retrieved context."""
        # 1. Check minimum retrieval score
        if retrieval_scores:
            max_score = max(retrieval_scores)
            if max_score < self._config.min_retrieval_score:
                return GuardrailResult(
                    verdict=GuardrailVerdict.BLOCK,
                    reason=f"Retrieval scores too low (max: {max_score:.3f}, "
                           f"threshold: {self._config.min_retrieval_score})",
                    details={"max_score": max_score},
                )

        # 2. Check claim grounding via word overlap (fast)
        answer_words = set(answer.lower().split())
        context_text = " ".join(retrieved_chunks[:3]).lower()  # top 3 chunks only
        context_words = set(context_text.split())

        if answer_words:
            intersection = answer_words & context_words
            grounding_ratio = len(intersection) / len(answer_words)

            if grounding_ratio < 0.3:
                return GuardrailResult(
                    verdict=GuardrailVerdict.WARN,
                    reason=f"Low grounding ratio: {grounding_ratio:.2f} "
                           f"({len(intersection)}/{len(answer_words)} words found in context)",
                    details={"grounding_ratio": grounding_ratio},
                )

        # 3. Check for hedging / uncertainty markers (good sign of honesty)
        uncertainty_markers = [
            "i'm not sure", "i don't know", "unclear", "not mentioned",
            "not available", "no information", "cannot determine",
        ]
        has_uncertainty = any(m in answer.lower() for m in uncertainty_markers)

        return GuardrailResult(
            verdict=GuardrailVerdict.PASS,
            reason=f"Answer appears grounded (grounding ratio: "
                   f"{grounding_ratio if answer_words else 'N/A'})",
            details={
                "grounding_ratio": grounding_ratio if answer_words else None,
                "has_uncertainty_markers": has_uncertainty,
            },
        )


def _extract_ngrams(text: str, n: int = 3) -> list[str]:
    """Extract character n-grams from text."""
    words = text.lower().split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(" ".join(words[i : i + n]))
    return ngrams


# ---------------------------------------------------------------------------
# Main guardrails orchestrator
# ---------------------------------------------------------------------------

class Guardrails:
    """Orchestrates all guardrail checks."""

    def __init__(self, config: Optional[GuardrailConfig] = None):
        self._config = config or GuardrailConfig()
        self._off_topic = OffTopicDetector(self._config)
        self._unsafe = UnsafeInputDetector(self._config)
        self._hallucination = HallucinationChecker(self._config)

    def update_topic_index(self, all_document_text: str) -> None:
        """Update the topic index from a single text blob."""
        self._off_topic.update_topic_index(all_document_text)

    def update_topic_index_from_documents(self, documents: list[str]) -> None:
        """Update the topic index from a list of documents (with IDF weighting)."""
        self._off_topic.update_topic_index_from_documents(documents)

    def check_input(self, query: str) -> GuardrailResult:
        """Run all input guardrails on a query."""
        # Check unsafe content first
        result = self._unsafe.check(query)
        if result.verdict == GuardrailVerdict.BLOCK:
            return result

        # Check off-topic
        result = self._off_topic.check(query)
        if result.verdict == GuardrailVerdict.BLOCK:
            return result

        return GuardrailResult(verdict=GuardrailVerdict.PASS, reason="All input guardrails passed")

    def check_output(
        self,
        answer: str,
        retrieved_chunks: list[str],
        retrieval_scores: list[float],
    ) -> GuardrailResult:
        """Run all output guardrails on the generated answer."""
        return self._hallucination.check(answer, retrieved_chunks, retrieval_scores)

    def get_refusal_message(self, guardrail_result: GuardrailResult) -> str:
        """Generate a refusal message based on which guardrail tripped."""
        if "off-topic" in guardrail_result.reason.lower() or "outside" in guardrail_result.reason.lower():
            return self._config.off_topic_refusal
        if "unsafe" in guardrail_result.reason.lower():
            return self._config.unsafe_refusal
        if any(kw in guardrail_result.reason.lower() for kw in ["grounding", "retrieval", "hallucination"]):
            return self._config.hallucination_refusal
        return "I'm unable to answer this question based on the available information."
