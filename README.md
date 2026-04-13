# Local CUDA Speech Recognizer for Python + ROS 2 Humble

Fast local speech recognition with:

- `faster-whisper` on CUDA for low-latency transcription
- Pixi for reproducible Windows and Linux environments
- ROS 2 Humble support through RoboStack packages inside Pixi
- A standalone CLI and an `rclpy` ROS node

## What this project does

- Transcribes local audio files on GPU
- Transcribes microphone audio live on GPU
- Publishes transcripts into ROS 2 topics
- Can read audio either from the microphone or from a ROS 2 PCM topic

## Why this stack

- `faster-whisper` uses CTranslate2 and is much faster than the original Whisper implementation for the same accuracy.
- Pixi can manage a ROS 2 Humble environment directly, including cross-platform activation and task execution.

## Prerequisites

1. Install `pixi`.
2. Use an NVIDIA GPU with current drivers.
3. Make CUDA 12 and cuDNN 9 runtime libraries visible to the process.

Notes:

- Linux: the `faster-whisper` README documents `nvidia-cublas-cu12` and `nvidia-cudnn-cu12==9.*` as a valid runtime path.
- Windows and Linux: the same README documents using the Purfview CUDA runtime bundle by adding its directory to `PATH`.
- On Windows, `pixi run doctor` now checks real inference. If it reports missing `cublas64_12.dll`, `cublasLt64_12.dll`, or `cudnn64_9.dll`, your CUDA runtime directory is still not on `PATH`.
- On Windows, the app also checks `WHISPER_CUDA_DIR`, `CUDA_BIN_DIR`, `CUDA_PATH\\bin`, and `CUDA_HOME\\bin` automatically before failing.

## Quick start

```powershell
pixi install
pixi run doctor
```

### Transcribe a file

```powershell
pixi run file -- audio.wav
```

### Live microphone transcription

```powershell
pixi run mic
```

### ROS 2 node using the local microphone

```powershell
pixi run ros-node
```

The node publishes transcripts to:

- `/speech_recognizer/text`

Inspect them with:

```powershell
pixi run ros2 topic echo /speech_recognizer/text
```

### ROS 2 node using a PCM topic instead of the microphone

```powershell
pixi run ros-node --ros-args -p input_mode:=topic -p audio_topic:=/audio_pcm16
```

Expected topic format:

- `std_msgs/msg/Int16MultiArray`
- Mono PCM16
- `16000 Hz` by default

## Tasks

- `pixi run doctor`
- `pixi run file -- <path>`
- `pixi run mic`
- `pixi run ros-node`
- `pixi run test`

## Tuning

Useful overrides:

```powershell
pixi run mic -- --model tiny.en --compute-type float16
pixi run ros-node --ros-args -p model:=tiny.en -p energy_threshold:=0.012
```

Defaults are aimed at fast local inference:

- model: `turbo`
- device: `cuda`
- compute type: `float16`
- beam size: `1`
- VAD filter: `on`
- VAD minimum silence: `500 ms`
- live sample rate: `16000`

If you want the smallest latency above all else, try `tiny.en`.

## Layout

- `src/speech_recognizer/engine.py`: CUDA `faster-whisper` wrapper
- `src/speech_recognizer/cli.py`: file and microphone CLI
- `src/speech_recognizer/ros2_node.py`: ROS 2 Humble node
- `src/speech_recognizer/audio.py`: PCM helpers and speech chunking
