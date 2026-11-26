# crepe-predictor

A dependency-light reimplementation of [CREPE](https://github.com/marl/crepe) pitch estimation, exported to ONNX for inference without PyTorch.

- `crepe_predictor.py` — the runtime package: framing and Viterbi-decoded pitch prediction from an ONNX session, wrapped in a `CrepePredictor` class, and Kaldi-compatible (NCCF, pitch) conversion in `to_kaldi_pitch`.
- `export_torchcrepe_to_onnx.py` — script to (re-)generate the ONNX checkpoints.

## Installation

```sh
pip install crepe-predictor
```

Inference only depends on `numpy`, `onnxruntime`, and `scipy`.
Exporting checkpoints additionally requires `torch` and `onnxscript`, listed as script dependencies at the top of `export_torchcrepe_to_onnx.py`.

## API

Inference goes through `crepe_predictor.CrepePredictor`; `crepe_predictor.to_kaldi_pitch` converts its output for Kaldi.

### `CrepePredictor(capacity="full", *, checkpoint=None, providers=None, session_options=None)`

Resolves a checkpoint and opens an ONNX Runtime session for it.

- `capacity`: `"tiny"`, `"small"`, `"medium"`, `"large"`, or `"full"` — model size, trading accuracy for speed. Ignored if `checkpoint` is given.
- `checkpoint`: path to a local `.onnx` file. If omitted, the checkpoint matching `capacity` is downloaded and cached under `$CREPE_CACHE_DIR`, `$XDG_CACHE_HOME`, or `~/.cache/crepe_predictor/checkpoints`.
- `providers`: ONNX Runtime execution providers, in order of preference, e.g. `["CUDAExecutionProvider", "CPUExecutionProvider"]`. Each entry is a provider name, or a `(name, options)` tuple to configure it, e.g. `("CUDAExecutionProvider", {"device_id": 1})`. Defaults to `["CPUExecutionProvider"]`.
- `session_options`: an `onnxruntime.SessionOptions`, e.g. to limit threads with `intra_op_num_threads` when running several predictors in parallel. Defaults to ONNX Runtime's own defaults.

### `predictor(audio, *, viterbi=True, center=True, frame_shift=0.01, batch_size=128) -> np.ndarray`

Calling a `CrepePredictor` estimates pitch from 16 kHz mono `audio` (a 1-D array), returning an `(n_frames, 2)` array of `(POV, pitch)`: probability of voicing in `[0, 1]`, and pitch in Hz.

- `viterbi`: decode pitch bins along a Viterbi path enforcing pitch continuity, instead of a per-frame argmax.
- `center`: pad `audio` so each frame is centered on its timestamp.
- `frame_shift`: frame spacing in seconds (the hop), rounded to a whole number of samples.
- `batch_size`: maximum number of frames passed to the model at once. Peak memory grows with it, not with the length of `audio`. The result does not depend on it, except for last-bit rounding when ONNX Runtime runs on several threads, which already varied with the length of `audio`. Larger values can be faster on GPU.

### `to_kaldi_pitch(pitch) -> np.ndarray`

Converts an `(n_frames, 2)` array of `(POV, pitch)`, from a `CrepePredictor` or another pitch extractor, into `(NCCF, pitch)` compatible with Kaldi's `process-pitch`: unvoiced frames are detected with a voicing HMM, pitch is interpolated over them, and POV is converted to an NCCF value. Raises `ValueError` if no frame is voiced.

## Usage

Estimate pitch from a synthetic tone:

```python
import numpy as np
from crepe_predictor import CrepePredictor

predictor = CrepePredictor("full")  # "tiny", "small", "medium", "large", or "full"

t = np.arange(16000) / 16000  # 1 second at 16 kHz
audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)  # a 220 Hz tone

pov, pitch = predictor(audio).T
print(pitch[pov > 0.5])  # pitch in Hz for confidently voiced frames
```

Process a recording and produce Kaldi-compatible pitch features, running on GPU when available:

```python
from scipy.io import wavfile
from crepe_predictor import CrepePredictor, to_kaldi_pitch

predictor = CrepePredictor(
    "full",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

sample_rate, audio = wavfile.read("speech.wav")
assert sample_rate == 16000
audio = audio.astype("float32") / 32768.0  # int16 PCM -> float32 in [-1, 1]

nccf, pitch = to_kaldi_pitch(predictor(audio)).T
```
