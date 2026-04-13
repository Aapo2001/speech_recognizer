from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

import numpy as np

from speech_recognizer.engine import RecognizerEngine
from speech_recognizer.types import TranscriptResult

LOG = logging.getLogger(__name__)


class TranscriptWorker:
    def __init__(
        self,
        engine: RecognizerEngine,
        on_result: Callable[[TranscriptResult], None],
    ) -> None:
        self._engine = engine
        self._on_result = on_result
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, audio: np.ndarray) -> None:
        if audio.size:
            self._queue.put(audio)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                result = self._engine.transcribe_audio(
                    item, sample_rate=self._engine.config.sample_rate
                )
            except Exception:
                LOG.exception("Transcription failed")
                continue
            if result.text:
                self._on_result(result)


class MicrophoneStreamer:
    def __init__(
        self,
        sample_rate: int,
        block_ms: int,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        blocksize = int(self.sample_rate * (self.block_ms / 1000.0))

        def callback(indata, frames, time_info, status) -> None:
            if status:
                LOG.warning("Sounddevice status: %s", status)
            chunk = np.frombuffer(bytes(indata), dtype=np.int16).copy()
            self._queue.put(chunk)

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=blocksize,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_nowait(self) -> np.ndarray | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

