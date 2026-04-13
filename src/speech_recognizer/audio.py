from __future__ import annotations

import time
from collections import deque

import numpy as np


def pcm16_to_float32(samples: np.ndarray) -> np.ndarray:
    if samples.dtype != np.int16:
        samples = samples.astype(np.int16, copy=False)
    return samples.astype(np.float32) / 32768.0


def rms_energy(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float32), dtype=np.float32)))


class AudioChunkSegmenter:
    """Splits continuous audio into speech chunks with a lightweight RMS VAD."""

    def __init__(
        self,
        sample_rate: int,
        energy_threshold: float,
        silence_seconds: float,
        min_speech_seconds: float,
        max_speech_seconds: float,
        preroll_seconds: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_seconds = silence_seconds
        self.min_speech_seconds = min_speech_seconds
        self.max_speech_seconds = max_speech_seconds
        self.preroll_seconds = preroll_seconds

        self._in_speech = False
        self._last_voice_at = 0.0
        self._speech_chunks: list[np.ndarray] = []
        self._preroll_chunks: deque[np.ndarray] = deque()
        self._preroll_samples = 0
        self._speech_samples = 0

    def feed(self, chunk: np.ndarray, now: float | None = None) -> list[np.ndarray]:
        now = time.monotonic() if now is None else now
        energy = rms_energy(chunk)
        is_speech = energy >= self.energy_threshold
        completed: list[np.ndarray] = []

        if self._in_speech:
            self._speech_chunks.append(chunk)
            self._speech_samples += int(chunk.size)
            if is_speech:
                self._last_voice_at = now
            if self.duration_seconds >= self.max_speech_seconds:
                flushed = self._flush()
                if flushed is not None:
                    completed.append(flushed)
            elif not is_speech and now - self._last_voice_at >= self.silence_seconds:
                flushed = self._flush()
                if flushed is not None:
                    completed.append(flushed)
            return completed

        self._append_preroll(chunk)
        if is_speech:
            self._in_speech = True
            self._last_voice_at = now
            self._speech_chunks = list(self._preroll_chunks)
            self._speech_samples = sum(int(part.size) for part in self._speech_chunks)
        return completed

    def flush_if_idle(self, now: float | None = None) -> list[np.ndarray]:
        now = time.monotonic() if now is None else now
        if self._in_speech and now - self._last_voice_at >= self.silence_seconds:
            flushed = self._flush()
            if flushed is not None:
                return [flushed]
        return []

    def finalize(self) -> list[np.ndarray]:
        flushed = self._flush()
        return [flushed] if flushed is not None else []

    @property
    def duration_seconds(self) -> float:
        return self._speech_samples / float(self.sample_rate)

    def _append_preroll(self, chunk: np.ndarray) -> None:
        self._preroll_chunks.append(chunk)
        self._preroll_samples += int(chunk.size)
        max_preroll_samples = int(self.sample_rate * self.preroll_seconds)
        while self._preroll_samples > max_preroll_samples and self._preroll_chunks:
            removed = self._preroll_chunks.popleft()
            self._preroll_samples -= int(removed.size)

    def _flush(self) -> np.ndarray | None:
        if not self._speech_chunks:
            self._reset()
            return None

        audio = np.concatenate(self._speech_chunks)
        duration = audio.size / float(self.sample_rate)
        self._reset()
        if duration < self.min_speech_seconds:
            return None
        return audio

    def _reset(self) -> None:
        self._in_speech = False
        self._speech_chunks = []
        self._speech_samples = 0
        self._preroll_chunks.clear()
        self._preroll_samples = 0

