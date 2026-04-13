from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from speech_recognizer.audio import AudioChunkSegmenter, pcm16_to_float32
from speech_recognizer.config import RecognizerConfig
from speech_recognizer.engine import RecognizerEngine
from speech_recognizer.live import MicrophoneStreamer, TranscriptWorker
from speech_recognizer.types import TranscriptResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast local CUDA speech recognition.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Validate the runtime and model load.")
    add_common_model_args(doctor)

    file_cmd = subparsers.add_parser("transcribe-file", help="Transcribe an audio file.")
    add_common_model_args(file_cmd)
    file_cmd.add_argument("path", type=Path)
    file_cmd.add_argument("--json", action="store_true", help="Print JSON output.")

    mic_cmd = subparsers.add_parser("transcribe-mic", help="Transcribe the default microphone.")
    add_common_model_args(mic_cmd)
    add_live_args(mic_cmd)
    mic_cmd.add_argument("--mic-device", default=None, help="Sounddevice input device.")

    return parser


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default=None)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--vad-filter", dest="vad_filter", action="store_true", default=True)
    parser.add_argument("--no-vad-filter", dest="vad_filter", action="store_false")
    parser.add_argument("--vad-min-silence-ms", type=int, default=500)


def add_live_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--energy-threshold", type=float, default=0.015)
    parser.add_argument("--silence-seconds", type=float, default=0.45)
    parser.add_argument("--min-speech-seconds", type=float, default=0.30)
    parser.add_argument("--max-speech-seconds", type=float, default=12.0)
    parser.add_argument("--preroll-seconds", type=float, default=0.20)
    parser.add_argument("--block-ms", type=int, default=100)


def config_from_args(args: argparse.Namespace) -> RecognizerConfig:
    return RecognizerConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        vad_min_silence_duration_ms=args.vad_min_silence_ms,
        sample_rate=getattr(args, "sample_rate", 16_000),
        energy_threshold=getattr(args, "energy_threshold", 0.015),
        silence_seconds=getattr(args, "silence_seconds", 0.45),
        min_speech_seconds=getattr(args, "min_speech_seconds", 0.30),
        max_speech_seconds=getattr(args, "max_speech_seconds", 12.0),
        preroll_seconds=getattr(args, "preroll_seconds", 0.20),
        mic_block_ms=getattr(args, "block_ms", 100),
    )


def print_result(result: TranscriptResult) -> None:
    if result.language:
        print(
            f"# language={result.language} "
            f"prob={result.language_probability if result.language_probability is not None else 0.0:.3f}"
        )
    for segment in result.segments:
        print(f"[{segment.start:7.2f}s -> {segment.end:7.2f}s] {segment.text}")
    if not result.segments and result.text:
        print(result.text)


def command_doctor(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    engine = RecognizerEngine(config)
    # Force one actual inference path so delayed CUDA DLL issues fail here.
    engine.transcribe_audio(np.zeros(config.sample_rate, dtype=np.float32), config.sample_rate)
    payload = {
        "status": "ok",
        "model": config.model,
        "device": config.device,
        "compute_type": config.compute_type,
        "sample_rate": config.sample_rate,
        "engine": type(engine.model).__name__,
    }
    print(json.dumps(payload, indent=2))
    return 0


def command_transcribe_file(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    engine = RecognizerEngine(config)
    result = engine.transcribe_file(args.path)
    if args.json:
        print(
            json.dumps(
                {
                    "text": result.text,
                    "language": result.language,
                    "language_probability": result.language_probability,
                    "segments": [
                        {"start": item.start, "end": item.end, "text": item.text}
                        for item in result.segments
                    ],
                },
                indent=2,
            )
        )
    else:
        print_result(result)
    return 0


def command_transcribe_mic(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    engine = RecognizerEngine(config)
    segmenter = AudioChunkSegmenter(
        sample_rate=config.sample_rate,
        energy_threshold=config.energy_threshold,
        silence_seconds=config.silence_seconds,
        min_speech_seconds=config.min_speech_seconds,
        max_speech_seconds=config.max_speech_seconds,
        preroll_seconds=config.preroll_seconds,
    )
    mic = MicrophoneStreamer(
        sample_rate=config.sample_rate,
        block_ms=config.mic_block_ms,
        device=args.mic_device,
    )
    worker = TranscriptWorker(engine=engine, on_result=print_result)

    stop = False

    def request_stop(signum, frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    mic.start()
    print("Listening. Press Ctrl+C to stop.", file=sys.stderr)

    try:
        while not stop:
            chunk = mic.read(timeout=0.1)
            if chunk is not None:
                for utterance in segmenter.feed(pcm16_to_float32(chunk)):
                    worker.submit(utterance)
            for utterance in segmenter.flush_if_idle():
                worker.submit(utterance)
            time.sleep(0.01)
        for utterance in segmenter.finalize():
            worker.submit(utterance)
    finally:
        mic.close()
        worker.close()
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "doctor":
        return command_doctor(args)
    if args.command == "transcribe-file":
        return command_transcribe_file(args)
    if args.command == "transcribe-mic":
        return command_transcribe_mic(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
