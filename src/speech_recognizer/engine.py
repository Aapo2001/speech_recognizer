from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel

from speech_recognizer.config import RecognizerConfig
from speech_recognizer.types import TranscriptResult, TranscriptSegment


class RecognizerEngine:
    def __init__(self, config: RecognizerConfig) -> None:
        self.config = config
        self._validate_runtime()
        self.model = WhisperModel(
            config.model,
            device=config.device,
            compute_type=config.compute_type,
        )
        self._batched_model = BatchedInferencePipeline(model=self.model)

    def transcribe_file(self, audio_path: str | Path) -> TranscriptResult:
        path = str(audio_path)
        segments_iter, info = self._batched_model.transcribe(
            path,
            batch_size=self.config.file_batch_size,
            beam_size=self.config.beam_size,
            language=self.config.language,
            condition_on_previous_text=False,
            vad_filter=self.config.vad_filter,
            vad_parameters=self._vad_parameters(),
        )
        segments = [
            TranscriptSegment(start=segment.start, end=segment.end, text=segment.text.strip())
            for segment in segments_iter
            if segment.text.strip()
        ]
        return TranscriptResult(
            text=" ".join(segment.text for segment in segments),
            segments=segments,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
        )

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> TranscriptResult:
        if sample_rate != self.config.sample_rate:
            raise ValueError(
                f"Expected sample rate {self.config.sample_rate}, received {sample_rate}."
            )
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)
        segments_iter, info = self.model.transcribe(
            mono,
            beam_size=self.config.beam_size,
            language=self.config.language,
            condition_on_previous_text=False,
            vad_filter=self.config.vad_filter,
            vad_parameters=self._vad_parameters(),
        )
        segments = [
            TranscriptSegment(start=segment.start, end=segment.end, text=segment.text.strip())
            for segment in segments_iter
            if segment.text.strip()
        ]
        return TranscriptResult(
            text=" ".join(segment.text for segment in segments),
            segments=segments,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
        )

    def _validate_runtime(self) -> None:
        if self.config.device != "cuda":
            return
        try:
            import ctranslate2
        except ImportError as exc:
            raise RuntimeError("ctranslate2 is required for CUDA transcription.") from exc

        cuda_devices = ctranslate2.get_cuda_device_count()
        if cuda_devices < 1:
            raise RuntimeError(
                "CUDA was requested but no CUDA device was detected. "
                "Check your NVIDIA driver and CUDA/cuDNN runtime libraries."
            )
        self._validate_cuda_libraries()

    def _validate_cuda_libraries(self) -> None:
        if platform.system() != "Windows":
            return

        self._augment_windows_dll_search_path()

        required_dlls = (
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cudnn64_9.dll",
        )
        missing: list[str] = []
        for dll_name in required_dlls:
            try:
                ctypes.WinDLL(dll_name)
            except OSError:
                missing.append(dll_name)

        if missing:
            path_value = os.environ.get("PATH", "")
            raise RuntimeError(
                "CUDA runtime DLLs were not found on PATH. Missing: "
                f"{', '.join(missing)}. "
                "Install cuBLAS for CUDA 12 and cuDNN 9, then add their bin directory to PATH. "
                "A practical Windows option is the Purfview faster-whisper runtime bundle "
                "documented in the faster-whisper README. "
                f"Current PATH starts with: {path_value[:300]}"
            )

    def _augment_windows_dll_search_path(self) -> None:
        candidate_dirs: list[str] = []

        for env_name in ("WHISPER_CUDA_DIR", "CUDA_BIN_DIR"):
            value = os.environ.get(env_name)
            if value:
                candidate_dirs.append(value)

        for env_name in ("CUDA_PATH", "CUDA_HOME"):
            value = os.environ.get(env_name)
            if value:
                candidate_dirs.append(os.path.join(value, "bin"))

        for directory in candidate_dirs:
            if not directory or not os.path.isdir(directory):
                continue
            try:
                os.add_dll_directory(directory)
            except (AttributeError, FileNotFoundError, OSError):
                pass
            current_path = os.environ.get("PATH", "")
            entries = current_path.split(os.pathsep) if current_path else []
            if directory not in entries:
                os.environ["PATH"] = directory + os.pathsep + current_path

    def _vad_parameters(self) -> dict[str, int]:
        return {
            "min_silence_duration_ms": self.config.vad_min_silence_duration_ms,
        }
