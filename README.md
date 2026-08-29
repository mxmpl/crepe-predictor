# crepe-predictor

A dependency-light reimplementation of [CREPE](https://github.com/marl/crepe) pitch estimation, exported to ONNX for inference without PyTorch.

- `crepe_predictor.py` — the runtime package: framing, Viterbi-decoded pitch prediction from an ONNX session, and Kaldi-compatible (NCCF, pitch) postprocessing, wrapped in a `CrepePredictor` class.
- `export_torchcrepe_to_onnx.py` — script to (re-)generate the ONNX checkpoints.

## Installation

```sh
pip install crepe-predictor
```

Inference only depends on `numpy`, `onnxruntime`, and `scipy`.
Exporting checkpoints additionally requires `torch` and `onnxscript`, listed as script dependencies at the top of `export_torchcrepe_to_onnx.py`.

## API

Everything is exposed through `crepe_predictor.CrepePredictor`.

### `CrepePredictor(capacity="full", *, checkpoint=None, onnx_providers=None)`

Resolves a checkpoint and opens an ONNX Runtime session for it.

- `capacity`: `"tiny"`, `"small"`, `"medium"`, `"large"`, or `"full"` — model size, trading accuracy for speed.
- `checkpoint`: path to a local `.onnx` file. If omitted, the checkpoint matching `capacity` is downloaded and cached under `$CREPE_CACHE_DIR`, `$XDG_CACHE_HOME`, or `~/.cache/crepe_predictor/checkpoints`.
- `onnx_providers`: ONNX Runtime execution providers, e.g. `["CUDAExecutionProvider", "CPUExecutionProvider"]`. Defaults to `["CPUExecutionProvider"]`.

### `predict(audio, *, viterbi=True, center=True, frame_shift=0.01, frame_length=0.025) -> np.ndarray`

Estimates pitch from 16 kHz mono `audio`, returning an `(n_frames, 2)` array of `(POV, pitch)`: probability of voicing in `[0, 1]`, and pitch in Hz.

- `viterbi`: decode pitch bins along a Viterbi path enforcing pitch continuity, instead of a per-frame argmax.
- `center`: pad `audio` so each frame is centered on its timestamp.
- `frame_shift`, `frame_length`: frame spacing and length in seconds, used to resample the output to the frame count they imply.

### `predict_kaldi(audio, *, viterbi=True, center=True, frame_shift=0.01, frame_length=0.025) -> np.ndarray`

Same arguments as `predict`, but returns `(n_frames, 2)` of `(NCCF, pitch)`, compatible with Kaldi's `process-pitch`: unvoiced frames are detected with a voicing HMM, pitch is interpolated over them, and POV is converted to an NCCF value. Raises `ValueError` if no frame is voiced.

## Usage

Estimate pitch from a synthetic tone:

```python
import numpy as np
from crepe_predictor import CrepePredictor

predictor = CrepePredictor("full")  # "tiny", "small", "medium", "large", or "full"

t = np.arange(16000) / 16000  # 1 second at 16 kHz
audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)  # a 220 Hz tone

pov, pitch = predictor.predict(audio).T
print(pitch[pov > 0.5])  # pitch in Hz for confidently voiced frames
```

Process a recording and produce Kaldi-compatible pitch features, running on GPU when available:

```python
from scipy.io import wavfile
from crepe_predictor import CrepePredictor

predictor = CrepePredictor(
    "full",
    onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

sample_rate, audio = wavfile.read("speech.wav")
assert sample_rate == 16000
audio = audio.astype("float32") / 32768.0  # int16 PCM -> float32 in [-1, 1]

nccf, pitch = predictor.predict_kaldi(audio).T
```
