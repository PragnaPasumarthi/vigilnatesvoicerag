"""
Pipeline harness for structured orchestration.

Provides:
- Retry logic with exponential backoff
- Structured input/output handling
- Error recovery and fallback paths
- Latency telemetry
- Pipeline step tracking
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class StepResult:
    """Result of a single pipeline step."""
    name: str
    success: bool
    duration_ms: float
    output: Any = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Aggregated result of the entire pipeline."""
    steps: list[StepResult] = field(default_factory=list)
    final_output: Any = None
    total_duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "steps": [
                {
                    "name": s.name,
                    "success": s.success,
                    "duration_ms": round(s.duration_ms, 2),
                    "error": s.error,
                }
                for s in self.steps
            ],
            "total_duration_ms": round(self.total_duration_ms, 2),
            "success": self.success,
            "error": self.error,
        }


class RetryConfig:
    """Configuration for retry behavior."""
    def __init__(
        self,
        max_retries: int = 3,
        base_delay_ms: float = 100,
        backoff_factor: float = 2.0,
        max_delay_ms: float = 2000,
    ):
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.backoff_factor = backoff_factor
        self.max_delay_ms = max_delay_ms


class PipelineHarness:
    """
    Orchestrates pipeline steps with retries, telemetry, and error recovery.

    Usage:
        harness = PipelineHarness()
        result = harness.run("transcribe", transcribe_fn, audio_bytes)
        result = harness.run("retrieve", retrieve_fn, query)
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        fallback_handlers: Optional[dict[str, Callable]] = None,
    ):
        self._retry_config = retry_config or RetryConfig()
        self._fallback_handlers = fallback_handlers or {}
        self._results: list[StepResult] = []

    def run(
        self,
        step_name: str,
        fn: Callable[..., T],
        *args: Any,
        retry: bool = True,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs: Any,
    ) -> StepResult:
        """Run a single step with optional retry and fallback."""
        t0 = time.perf_counter()
        last_error: Optional[str] = None
        max_attempts = self._retry_config.max_retries + 1 if retry else 1

        for attempt in range(max_attempts):
            try:
                output = fn(*args, **kwargs)
                duration_ms = (time.perf_counter() - t0) * 1000

                result = StepResult(
                    name=step_name,
                    success=True,
                    duration_ms=duration_ms,
                    output=output,
                    metadata={"attempt": attempt + 1},
                )
                self._results.append(result)
                return result

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < max_attempts - 1:
                    delay_ms = min(
                        self._retry_config.base_delay_ms * (
                            self._retry_config.backoff_factor ** attempt
                        ),
                        self._retry_config.max_delay_ms,
                    )
                    time.sleep(delay_ms / 1000)

        # All retries failed — try fallback
        if fallback:
            try:
                t0_fb = time.perf_counter()
                output = fallback(*args, **kwargs)
                duration_ms = (time.perf_counter() - t0_fb) * 1000

                result = StepResult(
                    name=step_name,
                    success=True,
                    duration_ms=duration_ms,
                    output=output,
                    metadata={"attempt": max_attempts, "fallback": True},
                )
                self._results.append(result)
                return result
            except Exception as e2:
                last_error = f"Fallback also failed: {type(e2).__name__}: {e2}"

        # Check registered fallback handlers
        if step_name in self._fallback_handlers:
            try:
                t0_fh = time.perf_counter()
                output = self._fallback_handlers[step_name](*args, **kwargs)
                duration_ms = (time.perf_counter() - t0_fh) * 1000

                result = StepResult(
                    name=step_name,
                    success=True,
                    duration_ms=duration_ms,
                    output=output,
                    metadata={"fallback_handler": True},
                )
                self._results.append(result)
                return result
            except Exception as e3:
                last_error = f"Handler failed: {type(e3).__name__}: {e3}"

        # Complete failure
        duration_ms = (time.perf_counter() - t0) * 1000
        result = StepResult(
            name=step_name,
            success=False,
            duration_ms=duration_ms,
            error=last_error,
            metadata={"attempts": max_attempts},
        )
        self._results.append(result)
        return result

    def build_result(self) -> PipelineResult:
        """Build the aggregated pipeline result."""
        total_ms = sum(s.duration_ms for s in self._results)
        success = all(s.success for s in self._results)
        errors = [s.error for s in self._results if s.error]

        return PipelineResult(
            steps=self._results,
            total_duration_ms=total_ms,
            success=success,
            error=errors[0] if errors else None,
        )

    def reset(self) -> None:
        """Reset the harness for a new pipeline run."""
        self._results = []


class StructuredIO:
    """Handles structured input/output validation."""

    @staticmethod
    def validate_transcription_input(audio_bytes: bytes) -> tuple[bool, str]:
        """Validate audio input for transcription."""
        if not audio_bytes:
            return False, "Empty audio input"
        if len(audio_bytes) < 100:
            return False, "Audio input too short (< 100 bytes)"
        if len(audio_bytes) > 50 * 1024 * 1024:  # 50MB
            return False, "Audio input too large (> 50MB)"
        return True, "Valid"

    @staticmethod
    def validate_query(query: str) -> tuple[bool, str]:
        """Validate a text query."""
        if not query or not query.strip():
            return False, "Empty query"
        if len(query.strip()) < 3:
            return False, "Query too short (< 3 chars)"
        if len(query) > 10000:
            return False, "Query too long (> 10000 chars)"
        return True, "Valid"

    @staticmethod
    def format_answer(
        answer: str,
        sources: list[dict],
        confidence: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Format a structured answer output."""
        return {
            "answer": answer.strip(),
            "sources": sources,
            "confidence": round(confidence, 3),
            "metadata": metadata or {},
        }
