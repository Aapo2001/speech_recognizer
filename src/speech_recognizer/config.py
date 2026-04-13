from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RecognizerConfig:
    model: str = "turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None
    beam_size: int = 1
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500
    sample_rate: int = 16_000
    file_batch_size: int = 16
    energy_threshold: float = 0.015
    silence_seconds: float = 0.45
    min_speech_seconds: float = 0.30
    max_speech_seconds: float = 12.0
    preroll_seconds: float = 0.20
    mic_block_ms: int = 100
