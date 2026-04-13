from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class TranscriptResult:
    text: str
    segments: list[TranscriptSegment]
    language: str | None = None
    language_probability: float | None = None

