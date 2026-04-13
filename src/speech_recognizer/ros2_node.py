from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, String

from speech_recognizer.audio import AudioChunkSegmenter, pcm16_to_float32
from speech_recognizer.config import RecognizerConfig
from speech_recognizer.engine import RecognizerEngine
from speech_recognizer.live import MicrophoneStreamer, TranscriptWorker
from speech_recognizer.types import TranscriptResult


class SpeechRecognizerNode(Node):
    def __init__(self) -> None:
        super().__init__("speech_recognizer")

        self.declare_parameter("model", "turbo")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("compute_type", "float16")
        self.declare_parameter("language", "")
        self.declare_parameter("beam_size", 1)
        self.declare_parameter("vad_filter", True)
        self.declare_parameter("vad_min_silence_duration_ms", 500)
        self.declare_parameter("sample_rate", 16_000)
        self.declare_parameter("energy_threshold", 0.015)
        self.declare_parameter("silence_seconds", 0.45)
        self.declare_parameter("min_speech_seconds", 0.30)
        self.declare_parameter("max_speech_seconds", 12.0)
        self.declare_parameter("preroll_seconds", 0.20)
        self.declare_parameter("mic_block_ms", 100)
        self.declare_parameter("input_mode", "microphone")
        self.declare_parameter("audio_topic", "/speech_recognizer/audio_pcm16")
        self.declare_parameter("text_topic", "/speech_recognizer/text")
        self.declare_parameter("mic_device", "")

        self.config = RecognizerConfig(
            model=self.get_parameter("model").value,
            device=self.get_parameter("device").value,
            compute_type=self.get_parameter("compute_type").value,
            language=self._nullable(self.get_parameter("language").value),
            beam_size=int(self.get_parameter("beam_size").value),
            vad_filter=bool(self.get_parameter("vad_filter").value),
            vad_min_silence_duration_ms=int(
                self.get_parameter("vad_min_silence_duration_ms").value
            ),
            sample_rate=int(self.get_parameter("sample_rate").value),
            energy_threshold=float(self.get_parameter("energy_threshold").value),
            silence_seconds=float(self.get_parameter("silence_seconds").value),
            min_speech_seconds=float(self.get_parameter("min_speech_seconds").value),
            max_speech_seconds=float(self.get_parameter("max_speech_seconds").value),
            preroll_seconds=float(self.get_parameter("preroll_seconds").value),
            mic_block_ms=int(self.get_parameter("mic_block_ms").value),
        )
        self.input_mode = self.get_parameter("input_mode").value
        self.audio_topic = self.get_parameter("audio_topic").value
        self.text_topic = self.get_parameter("text_topic").value
        mic_device = self._nullable(self.get_parameter("mic_device").value)

        self.publisher = self.create_publisher(String, self.text_topic, 10)
        self.segmenter = AudioChunkSegmenter(
            sample_rate=self.config.sample_rate,
            energy_threshold=self.config.energy_threshold,
            silence_seconds=self.config.silence_seconds,
            min_speech_seconds=self.config.min_speech_seconds,
            max_speech_seconds=self.config.max_speech_seconds,
            preroll_seconds=self.config.preroll_seconds,
        )
        self.engine = RecognizerEngine(self.config)
        self.worker = TranscriptWorker(engine=self.engine, on_result=self._publish_result)
        self.mic: MicrophoneStreamer | None = None

        if self.input_mode == "topic":
            self.subscription = self.create_subscription(
                Int16MultiArray,
                self.audio_topic,
                self._on_audio_message,
                10,
            )
            self.get_logger().info(
                f"Listening to {self.audio_topic} as Int16MultiArray at {self.config.sample_rate} Hz."
            )
        elif self.input_mode == "microphone":
            self.mic = MicrophoneStreamer(
                sample_rate=self.config.sample_rate,
                block_ms=self.config.mic_block_ms,
                device=mic_device,
            )
            self.mic.start()
            self.get_logger().info("Listening to the local microphone.")
        else:
            raise ValueError("input_mode must be either 'microphone' or 'topic'.")

        self.timer = self.create_timer(0.05, self._tick)

    def destroy_node(self) -> bool:
        if self.mic is not None:
            self.mic.close()
        self.worker.close()
        return super().destroy_node()

    def _on_audio_message(self, msg: Int16MultiArray) -> None:
        chunk = np.asarray(msg.data, dtype=np.int16)
        for utterance in segmenter_output(self.segmenter, chunk):
            self.worker.submit(utterance)

    def _tick(self) -> None:
        if self.mic is not None:
            while True:
                chunk = self.mic.read_nowait()
                if chunk is None:
                    break
                for utterance in segmenter_output(self.segmenter, chunk):
                    self.worker.submit(utterance)

        for utterance in self.segmenter.flush_if_idle(time.monotonic()):
            self.worker.submit(utterance)

    def _publish_result(self, result: TranscriptResult) -> None:
        msg = String()
        msg.data = result.text
        self.publisher.publish(msg)
        self.get_logger().info(f"Transcript: {result.text}")

    @staticmethod
    def _nullable(value: str) -> str | None:
        return value or None


def segmenter_output(segmenter: AudioChunkSegmenter, chunk: np.ndarray) -> list[np.ndarray]:
    audio = pcm16_to_float32(chunk)
    return segmenter.feed(audio, now=time.monotonic())


def main() -> None:
    rclpy.init()
    node = SpeechRecognizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for utterance in node.segmenter.finalize():
            node.worker.submit(utterance)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
