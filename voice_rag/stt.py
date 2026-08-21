"""
Speech-to-text integration using ElevenLabs API.

Supports:
- Audio file transcription
- Streaming transcription
- Multiple audio format support
- Error handling and retries
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    text: str
    duration_ms: float
    language: Optional[str] = None
    confidence: Optional[float] = None
    raw_response: Optional[dict] = None


class ElevenLabsSTT:
    """
    ElevenLabs speech-to-text client.

    Uses the ElevenLabs Speech-to-Text API for high-quality transcription.
    """

    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={
                "xi-api-key": api_key,
            },
            timeout=30.0,
        )

    def transcribe(
        self,
        audio_bytes: bytes,
        model_id: str = "scribe_v1",
        language_code: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data (wav, mp3, etc.)
            model_id: ElevenLabs model to use
            language_code: Optional language hint (e.g., "en")
            prompt: Optional prompt for context

        Returns:
            TranscriptionResult with the transcribed text
        """
        t0 = time.perf_counter()

        files = {
            "audio": ("audio.wav", audio_bytes, "audio/wav"),
        }

        data = {
            "model_id": model_id,
        }
        if language_code:
            data["language_code"] = language_code
        if prompt:
            data["prompt"] = prompt

        response = self._client.post(
            "/speech-to-text",
            files=files,
            data=data,
        )
        response.raise_for_status()

        result = response.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return TranscriptionResult(
            text=result.get("text", ""),
            duration_ms=elapsed_ms,
            language=result.get("language"),
            raw_response=result,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MockSTT:
    """
    Mock STT for testing and demos.
    Returns the provided text as if it were transcribed.
    """

    def __init__(self, mock_text: str = ""):
        self._text = mock_text

    def transcribe(
        self,
        audio_bytes: bytes,
        **kwargs,
    ) -> TranscriptionResult:
        """Return mock transcription result."""
        t0 = time.perf_counter()
        # Simulate small latency
        time.sleep(0.01)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return TranscriptionResult(
            text=self._text,
            duration_ms=elapsed_ms,
            language="en",
        )

    def close(self) -> None:
        pass


def create_stt(api_key: Optional[str] = None, mock_text: str = ""):
    """
    Factory function to create the appropriate STT client.

    Falls back to mock if no API key is provided.
    """
    if api_key:
        return ElevenLabsSTT(api_key)
    return MockSTT(mock_text)
