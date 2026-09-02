import hashlib
import os
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
import pytest
import torch
import torchcrepe
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from typing_extensions import Self

import crepe_predictor
from crepe_predictor import CrepePredictor, _frame, _predict, _resolve_checkpoint, postprocess_pitch
from export_torchcrepe_to_onnx import CREPE, convert_from_torchcrepe, export_crepe_to_onnx


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """ONNX export of an untrained tiny CREPE model, for exercising CrepePredictor without a download."""
    path = str(tmp_path_factory.mktemp("onnx") / "tiny.onnx")
    export_crepe_to_onnx(CREPE("tiny"), path)
    return path


@pytest.fixture(scope="module", params=["tiny", "full"])
def models(request, tmp_path_factory):
    """torchcrepe reference, our torch CREPE, and its ONNX session, all sharing the same weights."""
    capacity = request.param
    reference = torchcrepe.Crepe(capacity)
    weights = os.path.join(os.path.dirname(torchcrepe.__file__), "assets", f"{capacity}.pth")
    reference.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    reference.eval()

    crepe = CREPE(capacity)
    crepe.load_state_dict(convert_from_torchcrepe(reference.state_dict()))
    crepe.eval()

    path = str(tmp_path_factory.mktemp("onnx") / f"{capacity}.onnx")
    export_crepe_to_onnx(crepe, path)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return reference, crepe, session


@settings(deadline=None)
@given(
    audio=arrays(
        np.float32,
        array_shapes(min_dims=1, max_dims=1, min_side=1024, max_side=4096),
        elements=st.floats(-1, 1, width=32),
    )
)
def test_crepe_matches_torchcrepe(models, audio):
    """Our torch and ONNX CREPE produce the same activation matrix as torchcrepe."""
    reference, crepe, session = models
    frames = _frame(audio, 160, center=True)
    with torch.no_grad():
        expected = reference(torch.from_numpy(frames)).numpy()
        torch_out = crepe(torch.from_numpy(frames)).numpy()
    onnx_out = session.run(None, {session.get_inputs()[0].name: frames})[0]

    np.testing.assert_allclose(torch_out, expected, atol=1e-5)
    np.testing.assert_allclose(onnx_out, expected, atol=1e-5)


def test_crepe_predictor_predict_shape(tiny_checkpoint):
    predictor = CrepePredictor("tiny", checkpoint=tiny_checkpoint)
    audio = np.zeros(16000, dtype=np.float32)
    pitch = predictor.predict(audio)
    assert pitch.ndim == 2
    assert pitch.shape[1] == 2
    assert np.all((pitch[:, 0] >= 0) & (pitch[:, 0] <= 1))  # POV is a probability
    assert np.all(pitch[:, 1] >= 0)  # pitch in Hz


def test_resolve_checkpoint_uses_explicit_path(tiny_checkpoint):
    assert _resolve_checkpoint("tiny", tiny_checkpoint) == Path(tiny_checkpoint)


def test_resolve_checkpoint_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_checkpoint("tiny", tmp_path / "missing.onnx")


def test_postprocess_interpolates_unvoiced_and_converts_pov_to_nccf():
    # runs must be long enough to overcome the voicing HMM's continuity prior (see _predict_voicing)
    pov = np.concatenate([np.zeros(15), np.full(15, 0.95), np.zeros(15)])
    pitch = np.concatenate([np.full(15, 100.0), np.full(15, 150.0), np.full(15, 100.0)])
    nccf_pitch = postprocess_pitch(np.stack([pov, pitch], axis=1))
    assert nccf_pitch.shape == (45, 2)
    assert np.all(nccf_pitch[:, 1] > 0)
    assert np.all((nccf_pitch[:, 0] >= 0) & (nccf_pitch[:, 0] <= 1))


def test_postprocess_raises_when_no_frame_is_voiced():
    pitch = np.zeros((10, 2))
    with pytest.raises(ValueError, match="No voiced frames"):
        postprocess_pitch(pitch)


def test_postprocess_raises_when_pitch_not_positive_after_interpolation():
    pov = np.full(20, 0.95)  # long constant run so the voicing HMM keeps every frame voiced
    pitch = np.full(20, 100.0)
    pitch[5] = -1.0  # invalid pitch value smuggled into an otherwise-voiced frame
    with pytest.raises(ValueError, match="Not all pitch values are positive"):
        postprocess_pitch(np.stack([pov, pitch], axis=1))


def test_predict_raises_on_audio_shorter_than_one_frame(tiny_checkpoint):
    predictor = CrepePredictor("tiny", checkpoint=tiny_checkpoint)
    audio = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError, match="audio is too short"):
        predictor.predict(audio, center=False)


class _FakeSession:
    """Minimal stand-in for an ONNX session returning a fixed activation matrix."""

    def __init__(self, salience: np.ndarray) -> None:
        self.salience = salience

    def get_inputs(self) -> list[object]:
        return [type("_Input", (), {"name": "frames"})]

    def run(self, output_names: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [self.salience]


def test_predict_frame_count_matches_the_original_crepe():
    """One estimate per hop over the whole signal, as marl/crepe and torchcrepe do."""
    n_frames, hop = 1000, 160
    salience = np.full((n_frames + 1, 360), 1e-6, dtype=np.float32)
    salience[:, 180] = 1.0
    audio = np.zeros(hop * n_frames, dtype=np.float32)

    session = cast(ort.InferenceSession, _FakeSession(salience))
    centered = _predict(session, audio, viterbi=False, center=True)
    assert len(centered) == 1 + len(audio) // hop

    salience = salience[: 1 + (len(audio) - crepe_predictor.WINDOW_SIZE) // hop]
    session = cast(ort.InferenceSession, _FakeSession(salience))
    uncentered = _predict(session, audio, viterbi=False, center=False)
    assert len(uncentered) == 1 + (len(audio) - crepe_predictor.WINDOW_SIZE) // hop


def test_predict_never_returns_a_negative_pitch():
    """A step-shaped contour used to ring through an FFT resample of the output."""
    n_frames = 1000
    salience = np.full((n_frames, 360), 1e-6, dtype=np.float32)
    # alternate between a low and a high pitch bin, so the contour has sharp steps
    salience[np.arange(n_frames), np.where((np.arange(n_frames) // 100) % 2 == 0, 20, 340)] = 1.0

    audio = np.zeros(160 * (n_frames - 1), dtype=np.float32)
    pitch = _predict(cast(ort.InferenceSession, _FakeSession(salience)), audio, viterbi=False)
    assert np.all(pitch[:, 1] > 0)


def test_frame_without_centering():
    audio = np.zeros(2048, dtype=np.float32)
    frames = _frame(audio, 160, center=False)
    assert frames.shape[1] == 1024
    assert len(frames) == (len(audio) - crepe_predictor.WINDOW_SIZE) // 160 + 1


def test_frame_raises_on_audio_shorter_than_one_frame():
    audio = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError, match="audio is too short"):
        _frame(audio, 160, center=False)


def test_crepe_predictor_predict_kaldi(tiny_checkpoint, monkeypatch):
    predictor = CrepePredictor("tiny", checkpoint=tiny_checkpoint)
    pov = np.full(20, 0.95)
    pitch = np.full(20, 150.0)
    monkeypatch.setattr(predictor, "predict", lambda *args, **kwargs: np.stack([pov, pitch], axis=1))
    nccf_pitch = predictor.predict_kaldi(np.zeros(16000, dtype=np.float32))
    assert nccf_pitch.shape == (20, 2)


def test_cache_dir_uses_crepe_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CREPE_CACHE_DIR", str(tmp_path))
    assert crepe_predictor._cache_dir() == tmp_path / "crepe_predictor" / "checkpoints"


def test_cache_dir_falls_back_to_home(monkeypatch):
    monkeypatch.delenv("CREPE_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert crepe_predictor._cache_dir() == Path.home() / ".cache" / "crepe_predictor" / "checkpoints"


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, size: int) -> bytes:
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass


def test_download_moves_temp_file_to_destination(tmp_path, monkeypatch):
    destination = tmp_path / "sub" / "tiny.onnx"
    data = b"data"
    monkeypatch.setitem(crepe_predictor._CHECKSUMS, "tiny", hashlib.sha256(data).hexdigest())
    captured = {}

    def fake_urlopen(url, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResponse(data)

    monkeypatch.setattr(crepe_predictor.urllib.request, "urlopen", fake_urlopen)
    crepe_predictor._download("tiny", destination)
    assert destination.read_bytes() == data
    assert captured["url"] == crepe_predictor._REMOTE_URLS["tiny"]
    assert captured["timeout"] == crepe_predictor._TIMEOUT


def test_download_raises_and_leaves_no_file_on_checksum_mismatch(tmp_path, monkeypatch):
    destination = tmp_path / "tiny.onnx"
    monkeypatch.setattr(
        crepe_predictor.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse(b"corrupted")
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        crepe_predictor._download("tiny", destination)
    assert not destination.exists()


def test_resolve_checkpoint_downloads_when_missing_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(crepe_predictor, "_cache_dir", lambda: tmp_path)
    calls = []

    def fake_download(capacity, destination):
        calls.append((capacity, destination))
        destination.touch()

    monkeypatch.setattr(crepe_predictor, "_download", fake_download)
    path = _resolve_checkpoint("tiny", None)
    assert path == tmp_path / "tiny.onnx"
    assert calls == [("tiny", tmp_path / "tiny.onnx")]


def test_resolve_checkpoint_skips_download_when_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(crepe_predictor, "_cache_dir", lambda: tmp_path)
    (tmp_path / "tiny.onnx").touch()

    def fail_download(capacity, destination):
        raise AssertionError("should not download when the checkpoint is already cached")

    monkeypatch.setattr(crepe_predictor, "_download", fail_download)
    path = _resolve_checkpoint("tiny", None)
    assert path == tmp_path / "tiny.onnx"
