import hashlib
import os
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import onnxruntime as ort
import scipy.interpolate

__all__ = ["WINDOW_SIZE", "Capacity", "CrepePredictor", "to_kaldi_pitch"]

Capacity = Literal["tiny", "small", "medium", "large", "full"]

# bin-number-to-cents mapping used by the CREPE classifier (360 pitch bins)
_CENTS_MAPPING = np.linspace(0, 7180, 360) + 1997.3794084376191
_SAMPLE_RATE = 16000
_TIMEOUT = 30
_REMOTE_URLS = {
    "tiny": "https://media.githubusercontent.com/media/mxmpl/crepe-predictor/main/checkpoints/tiny.onnx",
    "small": "https://media.githubusercontent.com/media/mxmpl/crepe-predictor/main/checkpoints/small.onnx",
    "medium": "https://media.githubusercontent.com/media/mxmpl/crepe-predictor/main/checkpoints/medium.onnx",
    "large": "https://media.githubusercontent.com/media/mxmpl/crepe-predictor/main/checkpoints/large.onnx",
    "full": "https://media.githubusercontent.com/media/mxmpl/crepe-predictor/main/checkpoints/full.onnx",
}
_CHECKSUMS = {
    "tiny": "345b8ed787dc94236f237f234c6fb3f3f389291315b44909643c011b7a16f8c7",
    "small": "762e975d7717d5c47265344d3588dfac5bdee7e9a66d666a40f888a20eddbfa8",
    "medium": "00e25a2fbf0b141a2c739609fd1587cb8923daf78162d7dd3404413cc7fbf985",
    "large": "bff31dfecdaca02141cf3c1c3bc984e2ea2bef7de98d9cd6a36bbbe851fb12c4",
    "full": "9046c78f1cf40ebdbad1a2b3d9dc154dab36ef51ab55c6cd4d43776b9f5948ce",
}

WINDOW_SIZE = 1024  # samples per analysis window, fixed by the CREPE architecture


def _frame(audio: np.ndarray, hop_length: int, center: bool) -> np.ndarray:
    """Split ``audio`` into normalized 1024-sample frames expected by CREPE."""
    audio = np.asarray(audio, dtype=np.float32)
    if center:
        # pad so frames are centered on their timestamps (first frame is zero-centered)
        audio = np.pad(audio, WINDOW_SIZE // 2)
    if len(audio) < WINDOW_SIZE:
        raise ValueError(f"audio is too short to form a single {WINDOW_SIZE}-sample frame: got {len(audio)} samples")
    frames = np.lib.stride_tricks.sliding_window_view(audio, WINDOW_SIZE)[::hop_length].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    frames /= np.clip(frames.std(axis=1, keepdims=True), 1e-8, None)  # avoid /0 on constant (silent) frames
    return frames


def _local_average_cents(salience: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Weighted average of the cents mapping over a +/-4 bin window around each center."""
    salience = np.pad(salience, ((0, 0), (4, 4)))
    mapping = np.pad(_CENTS_MAPPING, (4, 4))
    index = centers[:, None] + np.arange(9)  # window [center-4, center+4] in padded coords
    window = np.take_along_axis(salience, index, axis=1)
    return (window * mapping[index]).sum(axis=1) / window.sum(axis=1)


def _viterbi(log_start: np.ndarray, log_trans: np.ndarray, log_emit: np.ndarray) -> np.ndarray:
    """Generic Viterbi decode. ``log_emit`` has shape (n_frames, n_states)."""
    n_frames = log_emit.shape[0]
    score = log_start + log_emit[0]
    backpointers = np.empty_like(log_emit, dtype=int)
    for t in range(1, n_frames):
        candidates = score[:, None] + log_trans
        backpointers[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_emit[t]
    path = np.empty(n_frames, dtype=int)
    path[-1] = score.argmax()
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = backpointers[t, path[t]]
    return path


def _viterbi_centers(salience: np.ndarray) -> np.ndarray:
    """Viterbi path over the 360 pitch bins with a transition prior enforcing continuity."""
    n = 360
    transition = np.maximum(12 - np.abs(np.subtract.outer(np.arange(n), np.arange(n))), 0.0)
    transition /= transition.sum(axis=1, keepdims=True)
    emission = np.eye(n) * 0.1 + 0.9 / n  # fixed self-probability, uniform otherwise
    observations = salience.argmax(axis=1)
    with np.errstate(divide="ignore"):
        return _viterbi(np.full(n, -np.log(n)), np.log(transition), np.log(emission)[:, observations].T)


def _predict(
    session: ort.InferenceSession,
    audio: np.ndarray,
    viterbi: bool = True,
    center: bool = True,
    frame_shift: float = 0.01,
) -> np.ndarray:
    """Extract the (POV, pitch) per frame from a 16 kHz mono ``audio`` signal.

    ``session`` runs the CREPE model exported to ONNX (see ``model.export_onnx``). The first
    output column is the probability of voicing, the second the estimated pitch in Hz.

    As in the original CREPE, one estimate is produced per ``frame_shift``: the result has
    ``1 + len(audio) // hop_length`` rows when ``center`` is set, and
    ``1 + (len(audio) - WINDOW_SIZE) // hop_length`` otherwise.
    """
    hop_length = round(_SAMPLE_RATE * frame_shift)
    frames = _frame(audio, hop_length, center)
    salience = np.asarray(session.run(None, {session.get_inputs()[0].name: frames})[0])  # activation matrix, (T, 360)
    confidence = salience.max(axis=1)  # heuristic voicing probability
    centers = _viterbi_centers(salience) if viterbi else salience.argmax(axis=1)
    cents = _local_average_cents(salience, centers)
    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0
    return np.stack([confidence, frequency], axis=1)


def _predict_voicing(confidence: np.ndarray) -> np.ndarray:
    """Viterbi path over voiced (1) vs unvoiced (0) frames from the voicing confidence."""
    means, variance = np.array([0.0, 1.0]), 0.25  # unvoiced and voiced states
    log_emit = -((confidence[:, None] - means) ** 2) / (2 * variance)  # gaussian, up to a constant
    log_start = np.log([0.5, 0.5])
    log_trans = np.log([[0.99, 0.01], [0.01, 0.99]])  # prior on continuous voicing state
    return _viterbi(log_start, log_trans, log_emit)


def _nccf_to_pov(nccf: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation to probability of voicing (Povey, ICASSP 2014)."""
    y = -5.2 + 5.4 * np.exp(7.5 * (nccf - 1)) + 4.8 * nccf - 2 * np.exp(-10 * nccf) + 4.2 * np.exp(20 * (nccf - 1))
    return 1 / (1 + np.exp(-y))


def to_kaldi_pitch(pitch: np.ndarray) -> np.ndarray:
    """Convert (POV, pitch) per frame into (NCCF, pitch) for Kaldi's ``process-pitch``.

    Unvoiced frames are detected with a voicing HMM and their pitch interpolated from the
    surrounding voiced frames, then the POV is converted back to an NCCF.

    Args:
        pitch: An ``(n_frames, 2)`` array of (POV, pitch), as returned by calling a
            :class:`CrepePredictor`: the probability of voicing in ``[0, 1]`` and the pitch in Hz.

    Returns:
        An ``(n_frames, 2)`` array of (NCCF, pitch): the NCCF in ``[0, 1]`` and the pitch in Hz,
        interpolated over unvoiced frames so that it is positive everywhere.

    Raises:
        ValueError: If no frame is voiced, or if the interpolated pitch is not positive everywhere.
    """
    to_remove = _predict_voicing(pitch[:, 0]) == 0  # interpolate pitch values over the unvoiced frames
    if np.all(to_remove):
        raise ValueError("No voiced frames")
    data = pitch[:, 1].copy()
    keep = np.where(~to_remove)[0]
    first, last = keep[0], keep[-1]
    interp = scipy.interpolate.interp1d(keep, data[keep], fill_value="extrapolate")
    data[to_remove] = interp(np.where(to_remove)[0])
    data[:first] = data[first]
    data[last:] = data[last]
    if not np.all(data > 0):
        raise ValueError("Not all pitch values are positive after interpolation")
    grid = np.linspace(0, 1, 4096)
    nccf = np.interp(pitch[:, 0], _nccf_to_pov(grid), grid, left=0.0, right=1.0)
    return np.stack([nccf, data], axis=1)


def _cache_dir() -> Path:
    base = os.environ.get("CREPE_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "crepe_predictor" / "checkpoints"


def _download(capacity: Capacity, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp_dir:
        tmp = Path(tmp_dir) / destination.name
        digest = hashlib.sha256()
        with urllib.request.urlopen(_REMOTE_URLS[capacity], timeout=_TIMEOUT) as response, tmp.open("wb") as f:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                f.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != _CHECKSUMS[capacity]:
            raise ValueError(
                f"checksum mismatch for the {capacity!r} checkpoint: "
                f"expected {_CHECKSUMS[capacity]}, got {digest.hexdigest()}"
            )
        tmp.replace(destination)


def _resolve_checkpoint(capacity: Capacity, checkpoint: str | Path | None) -> Path:
    """Resolve a checkpoint path, downloading and caching the remote one if needed."""
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint at {path}")
        return path
    path = _cache_dir() / f"{capacity}.onnx"
    if not path.is_file():
        _download(capacity, path)
    return path


class CrepePredictor:
    """Load a CREPE ONNX checkpoint and run pitch inference."""

    def __init__(
        self,
        capacity: Capacity = "full",
        *,
        checkpoint: str | Path | None = None,
        providers: Sequence[str | tuple[str, dict[str, Any]]] | None = None,
        session_options: ort.SessionOptions | None = None,
    ) -> None:
        """Resolve a checkpoint and open an ONNX Runtime session for it.

        Args:
            capacity: Model size, trading accuracy for speed.
            checkpoint: Path to a local ``.onnx`` file. If omitted, the checkpoint matching
                ``capacity`` is downloaded and cached under ``$CREPE_CACHE_DIR``,
                ``$XDG_CACHE_HOME``, or ``~/.cache/crepe_predictor/checkpoints``.
            providers: ONNX Runtime execution providers, in order of preference. Each is a
                provider name, or a ``(name, options)`` tuple to configure it, e.g.
                ``[("CUDAExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"]``.
                Defaults to ``["CPUExecutionProvider"]``.
            session_options: ONNX Runtime session options, e.g. to set the number of threads
                with ``intra_op_num_threads``. Defaults to ONNX Runtime's own defaults.

        Raises:
            FileNotFoundError: If ``checkpoint`` is given but does not exist.
            ValueError: If the downloaded checkpoint does not match its expected checksum.
        """
        self.capacity = capacity
        path = _resolve_checkpoint(capacity, checkpoint)
        self.session = ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )

    def __call__(
        self,
        audio: np.ndarray,
        *,
        viterbi: bool = True,
        center: bool = True,
        frame_shift: float = 0.01,
    ) -> np.ndarray:
        """Estimate the pitch of a 16 kHz mono ``audio`` signal, frame by frame.

        Args:
            audio: 1-D array of audio samples at 16 kHz, typically float32 in ``[-1, 1]``.
                Each frame is normalized independently, so the overall scale does not matter.
            viterbi: If true, decode pitch bins along a Viterbi path that enforces pitch
                continuity across frames. Otherwise, take the most salient bin of each frame.
            center: If true, pad ``audio`` by half a window on both sides so that each frame
                is centered on its timestamp, the first one at time 0.
            frame_shift: Spacing between consecutive frames, in seconds.

        Returns:
            An ``(n_frames, 2)`` array of (POV, pitch): the probability of voicing in ``[0, 1]``
            and the pitch in Hz. Pass it to :func:`to_kaldi_pitch` to get (NCCF, pitch) for Kaldi.

            There is one row per ``frame_shift``, as in the original CREPE. With
            ``hop_length = round(16000 * frame_shift)``, ``n_frames`` is
            ``1 + len(audio) // hop_length`` if ``center`` is true, and
            ``1 + (len(audio) - WINDOW_SIZE) // hop_length`` otherwise.

        Raises:
            ValueError: If ``audio`` is too short to form a single frame, i.e. fewer than
                ``WINDOW_SIZE`` samples when ``center`` is false.
        """
        return _predict(self.session, audio, viterbi, center, frame_shift)
