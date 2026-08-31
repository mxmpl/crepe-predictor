# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "onnx>=1.22.0",
#     "onnxscript>=0.7.1",
#     "torch>=2.13.0",
# ]
# ///
"""Convert torchcrepe checkpoints to ONNX for use with :class:`crepe_predictor.CrepePredictor`.

Defines :class:`CREPE`, a torch reimplementation of the architecture used by
`torchcrepe <https://github.com/maxrmorrison/torchcrepe>`_, remaps a torchcrepe state
dict onto it with :func:`convert_from_torchcrepe`, and exports the result to a single
ONNX file with :func:`export_crepe_to_onnx`.

Run as a script to convert a torchcrepe ``.pth`` checkpoint to ``.onnx``::

    uv run export_torchcrepe_to_onnx.py <checkpoint> <output.onnx> <capacity>

where ``<capacity>`` is one of ``tiny``, ``small``, ``medium``, ``large``, ``full`` and
must match the checkpoint being converted. The exported model has a dynamic frame
(batch) dimension, so any number of 1024-sample frames can be passed to the resulting
ONNX session in a single call.
"""

from typing import assert_never

import torch
from torch import nn
from torch.nn import functional as F

from crepe_predictor import Capacity


def crepe_multiplier(capacity: Capacity) -> int:
    match capacity:
        case "tiny":
            return 4
        case "small":
            return 8
        case "medium":
            return 16
        case "large":
            return 24
        case "full":
            return 32
        case _:
            assert_never(capacity)


class CREPELayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), stride)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, self.padding[0], self.padding[1]))
        x = self.batch_norm(F.relu(self.conv(x)))
        return F.max_pool2d(x, (2, 1), (2, 1))


class CREPE(nn.Module):
    def __init__(self, capacity: Capacity = "full") -> None:
        super().__init__()
        self.capacity = capacity
        mult = crepe_multiplier(self.capacity)
        self.layers = nn.Sequential(
            CREPELayer(1, mult * 32, 512, 4, (254, 254)),
            CREPELayer(mult * 32, mult * 4, 64, 1, (31, 32)),
            CREPELayer(mult * 4, mult * 4, 64, 1, (31, 32)),
            CREPELayer(mult * 4, mult * 4, 64, 1, (31, 32)),
            CREPELayer(mult * 4, mult * 8, 64, 1, (31, 32)),
            CREPELayer(mult * 8, mult * 16, 64, 1, (31, 32)),
        )
        self.classifier = nn.Linear(mult * 64, 360)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1).unsqueeze(3)
        x = self.layers(x)
        x = x.permute(0, 2, 1, 3).flatten(1)
        return self.classifier(x).sigmoid()


def convert_from_torchcrepe(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap a torchcrepe checkpoint to this module's :class:`CREPE` state dict."""
    converted = {}
    for k in range(6):
        converted[f"layers.{k}.conv.weight"] = state_dict[f"conv{k + 1}.weight"]
        converted[f"layers.{k}.conv.bias"] = state_dict[f"conv{k + 1}.bias"]
        converted[f"layers.{k}.batch_norm.running_mean"] = state_dict[f"conv{k + 1}_BN.running_mean"]
        converted[f"layers.{k}.batch_norm.running_var"] = state_dict[f"conv{k + 1}_BN.running_var"]
        converted[f"layers.{k}.batch_norm.weight"] = state_dict[f"conv{k + 1}_BN.weight"]
        converted[f"layers.{k}.batch_norm.bias"] = state_dict[f"conv{k + 1}_BN.bias"]
    converted["classifier.weight"] = state_dict["classifier.weight"]
    converted["classifier.bias"] = state_dict["classifier.bias"]
    return converted


def export_crepe_to_onnx(model: CREPE, path: str) -> None:
    """Export a CREPE model to a single ONNX file with a dynamic frame (batch) dimension."""
    model.eval()
    n_frames = torch.export.Dim("n_frames")
    torch.onnx.export(
        model,
        (torch.zeros(1, 1024),),
        path,
        input_names=["frames"],
        output_names=["salience"],
        dynamic_shapes=({0: n_frames},),
        opset_version=18,
        dynamo=True,
        external_data=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a torchcrepe checkpoint to ONNX.")
    parser.add_argument("checkpoint", help="path to the torchcrepe .pth checkpoint")
    parser.add_argument("output", help="path to the output .onnx file")
    parser.add_argument("capacity", choices=["tiny", "small", "medium", "large", "full"], help="the model size")
    args = parser.parse_args()

    model = CREPE(args.capacity)
    model.load_state_dict(convert_from_torchcrepe(torch.load(args.checkpoint, map_location="cpu", weights_only=True)))
    export_crepe_to_onnx(model, args.output)
