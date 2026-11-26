import hashlib
import os
from pathlib import Path
from typing import cast

import numpy as np

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

from typing import Self

import onnxruntime as ort
import pytest
import torch
import torchcrepe
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

import crepe_predictor
from crepe_predictor import (
    CrepePredictor,
    _frame_view,
    _normalize,
    _predict,
    _resolve_checkpoint,
    _viterbi,
    _viterbi_centers,
    to_kaldi_pitch,
)
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
    frames = _normalize(_frame_view(audio, 160, center=True))
    with torch.no_grad():
        expected = reference(torch.from_numpy(frames)).numpy()
        torch_out = crepe(torch.from_numpy(frames)).numpy()
    onnx_out = session.run(None, {session.get_inputs()[0].name: frames})[0]

    np.testing.assert_allclose(torch_out, expected, atol=1e-5)
    np.testing.assert_allclose(onnx_out, expected, atol=1e-5)


@pytest.mark.parametrize("frequency", [110.0, 220.0, 440.0, 880.0])
@pytest.mark.parametrize("viterbi", [True, False])
def test_predict_recovers_the_pitch_of_a_pure_tone(models, frequency, viterbi):
    """With the real weights, a sine is voiced and its pitch is found within 10 cents."""
    _, _, session = models
    t = np.arange(16000) / 16000
    pov, pitch = _predict(session, np.sin(2 * np.pi * frequency * t).astype(np.float32), viterbi=viterbi).T
    inner = slice(5, -5)  # frames near the edges overlap the zero padding
    assert np.mean(pov[inner] > 0.5) > 0.9
    cents = 1200 * np.log2(pitch[inner] / frequency)
    assert np.abs(cents).max() < 10


def test_crepe_predictor_call_shape(tiny_checkpoint):
    predictor = CrepePredictor("tiny", checkpoint=tiny_checkpoint)
    audio = np.zeros(16000, dtype=np.float32)
    pitch = predictor(audio)
    assert pitch.ndim == 2
    assert pitch.shape[1] == 2
    assert np.all((pitch[:, 0] >= 0) & (pitch[:, 0] <= 1))  # POV is a probability
    assert np.all(pitch[:, 1] >= 0)  # pitch in Hz


def test_crepe_predictor_accepts_provider_options_and_session_options(tiny_checkpoint):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    predictor = CrepePredictor(
        "tiny",
        checkpoint=tiny_checkpoint,
        providers=[("CPUExecutionProvider", {})],
        session_options=options,
    )
    assert predictor.session.get_providers() == ["CPUExecutionProvider"]
    assert predictor.session.get_session_options().intra_op_num_threads == 1


def _allocate_cores(monkeypatch: pytest.MonkeyPatch, platform: str, allocated: int, machine: int) -> None:
    """Simulate a process allowed on ``allocated`` of the ``machine``'s cores, seen through ``platform``'s API."""
    for name in ["process_cpu_count", "sched_getaffinity"]:
        monkeypatch.delattr(os, name, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: machine)
    if platform == "process_cpu_count":
        monkeypatch.setattr(os, "process_cpu_count", lambda: allocated, raising=False)
    elif platform == "sched_getaffinity":
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(allocated)), raising=False)


@pytest.mark.parametrize("platform", ["process_cpu_count", "sched_getaffinity", "cpu_count"])
def test_default_session_options_use_one_thread_per_allocated_core(monkeypatch, platform):
    _allocate_cores(monkeypatch, platform, allocated=10, machine=64)
    threads = crepe_predictor._default_session_options().intra_op_num_threads
    assert threads == (64 if platform == "cpu_count" else 10)  # os.cpu_count() cannot see the allocation


def test_crepe_predictor_defaults_to_one_thread_per_allocated_core(tiny_checkpoint, monkeypatch):
    _allocate_cores(monkeypatch, "sched_getaffinity", allocated=3, machine=64)
    predictor = CrepePredictor("tiny", checkpoint=tiny_checkpoint)
    assert predictor.session.get_session_options().intra_op_num_threads == 3


def test_crepe_predictor_uses_given_session_options_as_is(tiny_checkpoint, monkeypatch):
    _allocate_cores(monkeypatch, "sched_getaffinity", allocated=3, machine=64)
    predictor = CrepePredictor("tiny", checkpoint=tiny_checkpoint, session_options=ort.SessionOptions())
    assert predictor.session.get_session_options().intra_op_num_threads == 0  # ONNX Runtime's default


def test_crepe_predictor_defaults_to_cpu(tiny_checkpoint):
    predictor = CrepePredictor("tiny", checkpoint=tiny_checkpoint)
    assert predictor.session.get_providers() == ["CPUExecutionProvider"]


def test_crepe_predictor_result_does_not_depend_on_batch_size(tiny_checkpoint):
    # a single thread, since ONNX Runtime splits work across threads differently depending
    # on the batch shape, which can change the last bit of the activations
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    predictor = CrepePredictor("tiny", checkpoint=tiny_checkpoint, session_options=options)
    audio = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    expected = predictor(audio, batch_size=10_000)  # 101 frames, all in one batch
    for batch_size in [1, 7, 100]:
        np.testing.assert_array_equal(predictor(audio, batch_size=batch_size), expected)


def test_crepe_predictor_raises_on_multichannel_audio(tiny_checkpoint):
    predictor = CrepePredictor("tiny", checkpoint=tiny_checkpoint)
    with pytest.raises(ValueError, match="must be a 1-D mono signal"):
        predictor(np.zeros((16000, 2), dtype=np.float32))


def test_crepe_predictor_raises_on_frame_shift_shorter_than_one_sample(tiny_checkpoint):
    predictor = CrepePredictor("tiny", checkpoint=tiny_checkpoint)
    with pytest.raises(ValueError, match="frame_shift must be at least one sample"):
        predictor(np.zeros(16000, dtype=np.float32), frame_shift=1e-5)


def test_crepe_predictor_raises_on_non_positive_batch_size(tiny_checkpoint):
    predictor = CrepePredictor("tiny", checkpoint=tiny_checkpoint)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        predictor(np.zeros(16000, dtype=np.float32), batch_size=0)


def _dense_viterbi(log_start: np.ndarray, log_trans: np.ndarray, log_emit: np.ndarray) -> np.ndarray:
    """Textbook Viterbi decode over all transitions, as a reference for the banded one."""
    score = log_start + log_emit[0]
    backpointers = []
    for t in range(1, len(log_emit)):
        candidates = score[:, None] + log_trans
        backpointers.append(candidates.argmax(axis=0))
        score = candidates.max(axis=0) + log_emit[t]
    path = [int(score.argmax())]
    for pointers in reversed(backpointers):
        path.append(int(pointers[path[-1]]))
    return np.array(path[::-1])


@pytest.mark.parametrize("seed", range(3))
def test_banded_viterbi_matches_the_dense_one(seed):
    rng = np.random.default_rng(seed)
    n_states, max_jump = 50, 3
    log_trans = np.log(rng.random((n_states, n_states)))
    log_trans[np.abs(np.subtract.outer(np.arange(n_states), np.arange(n_states))) > max_jump] = -np.inf
    log_start, log_emit = np.log(rng.random(n_states)), np.log(rng.random((200, n_states)))
    expected = _dense_viterbi(log_start, log_trans, log_emit)
    np.testing.assert_array_equal(_viterbi(log_start, log_trans, log_emit, max_jump), expected)
    np.testing.assert_array_equal(_viterbi(log_start, log_trans, log_emit, n_states - 1), expected)


def test_viterbi_centers_matches_a_dense_decode(monkeypatch):
    """The pitch decode restricted to its transition band finds the same path as a full one."""
    rng = np.random.default_rng(0)
    contour = np.clip(180 + np.cumsum(rng.integers(-15, 16, 500)), 0, 359)  # includes jumps beyond the band
    salience = rng.random((500, 360)) * 0.5
    salience[np.arange(500), contour] = 1.0
    banded = _viterbi_centers(salience)
    monkeypatch.setattr(
        crepe_predictor, "_viterbi", lambda start, trans, emit, max_jump: _dense_viterbi(start, trans, emit)
    )
    np.testing.assert_array_equal(banded, _viterbi_centers(salience))


def test_resolve_checkpoint_uses_explicit_path(tiny_checkpoint):
    assert _resolve_checkpoint("tiny", tiny_checkpoint) == Path(tiny_checkpoint)


def test_resolve_checkpoint_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_checkpoint("tiny", tmp_path / "missing.onnx")


def test_to_kaldi_pitch_interpolates_unvoiced_and_converts_pov_to_nccf():
    # runs must be long enough to overcome the voicing HMM's continuity prior (see _predict_voicing)
    pov = np.concatenate([np.zeros(15), np.full(15, 0.95), np.zeros(15)])
    pitch = np.concatenate([np.full(15, 100.0), np.full(15, 150.0), np.full(15, 100.0)])
    nccf_pitch = to_kaldi_pitch(np.stack([pov, pitch], axis=1))
    assert nccf_pitch.shape == (45, 2)
    assert np.all(nccf_pitch[:, 1] > 0)
    assert np.all((nccf_pitch[:, 0] >= 0) & (nccf_pitch[:, 0] <= 1))


def test_to_kaldi_pitch_raises_when_no_frame_is_voiced():
    pitch = np.zeros((10, 2))
    with pytest.raises(ValueError, match="No voiced frames"):
        to_kaldi_pitch(pitch)


def test_to_kaldi_pitch_raises_when_pitch_not_positive_after_interpolation():
    pov = np.full(20, 0.95)  # long constant run so the voicing HMM keeps every frame voiced
    pitch = np.full(20, 100.0)
    pitch[5] = -1.0  # invalid pitch value smuggled into an otherwise-voiced frame
    with pytest.raises(ValueError, match="Not all pitch values are positive"):
        to_kaldi_pitch(np.stack([pov, pitch], axis=1))


def test_predict_raises_on_audio_shorter_than_one_frame(tiny_checkpoint):
    predictor = CrepePredictor("tiny", checkpoint=tiny_checkpoint)
    audio = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError, match="audio is too short"):
        predictor(audio, center=False)


class _FakeSession:
    """Minimal stand-in for an ONNX session returning a fixed activation matrix, batch by batch."""

    def __init__(self, salience: np.ndarray) -> None:
        self.salience = salience
        self.position = 0

    def get_inputs(self) -> list[object]:
        return [type("_Input", (), {"name": "frames"})]

    def run(self, output_names: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        n_frames = len(feeds["frames"])
        batch = self.salience[self.position : self.position + n_frames]
        assert len(batch) == n_frames, "more frames than rows in the activation matrix"
        self.position += n_frames
        return [batch]


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
    frames = _frame_view(audio, 160, center=False)
    assert frames.shape[1] == 1024
    assert len(frames) == (len(audio) - crepe_predictor.WINDOW_SIZE) // 160 + 1


def test_frame_raises_on_audio_shorter_than_one_frame():
    audio = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError, match="audio is too short"):
        _frame_view(audio, 160, center=False)


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
