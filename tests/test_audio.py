import numpy as np

from speech_recognizer.audio import AudioChunkSegmenter, pcm16_to_float32, rms_energy


def test_pcm16_to_float32_scales_samples() -> None:
    pcm = np.array([-32768, 0, 32767], dtype=np.int16)
    converted = pcm16_to_float32(pcm)
    assert converted.dtype == np.float32
    assert np.isclose(converted[0], -1.0)
    assert np.isclose(converted[1], 0.0)
    assert converted[2] > 0.99


def test_rms_energy_detects_non_silence() -> None:
    silence = np.zeros(1600, dtype=np.float32)
    speech = np.ones(1600, dtype=np.float32) * 0.1
    assert rms_energy(silence) == 0.0
    assert rms_energy(speech) > 0.01


def test_segmenter_flushes_after_silence() -> None:
    segmenter = AudioChunkSegmenter(
        sample_rate=16_000,
        energy_threshold=0.01,
        silence_seconds=0.2,
        min_speech_seconds=0.1,
        max_speech_seconds=5.0,
        preroll_seconds=0.1,
    )
    speech = np.ones(1600, dtype=np.float32) * 0.1
    silence = np.zeros(1600, dtype=np.float32)

    assert segmenter.feed(speech, now=0.0) == []
    assert segmenter.feed(speech, now=0.1) == []
    completed = segmenter.feed(silence, now=0.35)
    assert len(completed) == 1
    assert completed[0].size >= speech.size * 2
