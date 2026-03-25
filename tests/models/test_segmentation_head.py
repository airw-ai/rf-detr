# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from contextlib import contextmanager

import pytest
import torch

from rfdetr.models.heads.segmentation import DepthwiseConvBlock


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            id="gpu",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is not available",
                ),
            ],
        ),
    ],
)
def test_depthwise_conv_block_forward(device: str) -> None:
    """DepthwiseConvBlock forward pass produces correct output shape without error."""
    block = DepthwiseConvBlock(dim=8).to(device)
    x = torch.randn(1, 8, 4, 4, device=device)
    y = block(x)
    assert y.shape == x.shape


def test_depthwise_conv_block_disables_cudnn_on_cpu(monkeypatch) -> None:
    """On CPU (and pre-Ampere CUDA), depthwise conv should execute with cuDNN disabled."""
    block = DepthwiseConvBlock(dim=8)
    cudnn_enabled = True

    class _MockDepthwiseConv(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            # On CPU the cuDNN flags context is still entered (with enabled=False),
            # but cuDNN itself is a no-op on CPU.
            return x

    fallback_dwconv = _MockDepthwiseConv()
    block.dwconv = fallback_dwconv

    enabled_calls: list[bool] = []

    @contextmanager
    def _fake_cudnn_flags(*, enabled: bool):
        nonlocal cudnn_enabled
        previous = cudnn_enabled
        cudnn_enabled = enabled
        enabled_calls.append(enabled)
        try:
            yield
        finally:
            cudnn_enabled = previous

    monkeypatch.setattr(torch.backends.cudnn, "flags", _fake_cudnn_flags)

    # CPU tensor: compute capability check is skipped, so the cuDNN-disabled fallback path runs.
    x = torch.randn(1, 8, 4, 4)  # CPU tensor
    block(x)

    assert fallback_dwconv.calls == 1
    assert enabled_calls == [False], "cuDNN should be disabled for non-CUDA (CPU/MPS) inputs"


def test_depthwise_conv_block_enables_cudnn_on_ampere(monkeypatch) -> None:
    """On Ampere+ GPUs (compute ≥ 8.0), depthwise conv should run WITH cuDNN enabled."""
    block = DepthwiseConvBlock(dim=8)

    cudnn_flags_entered = []

    @contextmanager
    def _fake_cudnn_flags(*, enabled: bool):
        cudnn_flags_entered.append(enabled)
        yield

    monkeypatch.setattr(torch.backends.cudnn, "flags", _fake_cudnn_flags)

    # Simulate an Ampere GPU (compute capability 8.0) by patching get_device_capability.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))

    class _MockCUDATensor:
        """Minimal stand-in for a CUDA tensor to exercise the capability branch."""

        @property
        def device(self):
            class _Device:
                type = "cuda"
                index = 0

            return _Device()

    class _MockDepthwiseConv(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x) -> torch.Tensor:
            self.calls += 1
            return torch.zeros(1, 8, 4, 4)

    fallback_dwconv = _MockDepthwiseConv()
    block.dwconv = fallback_dwconv

    mock_x = _MockCUDATensor()
    block._depthwise_conv(mock_x)

    assert fallback_dwconv.calls == 1
    # cuDNN flags context should NOT have been entered for Ampere+.
    assert cudnn_flags_entered == [], "cuDNN should NOT be disabled on Ampere+ GPUs"


def test_depthwise_conv_block_disables_cudnn_on_pre_ampere(monkeypatch) -> None:
    """On pre-Ampere CUDA (compute < 8.0, e.g. T4=7.5), cuDNN must be disabled."""
    block = DepthwiseConvBlock(dim=8)

    enabled_calls: list[bool] = []

    @contextmanager
    def _fake_cudnn_flags(*, enabled: bool):
        enabled_calls.append(enabled)
        yield

    monkeypatch.setattr(torch.backends.cudnn, "flags", _fake_cudnn_flags)
    # Simulate a T4 GPU (compute 7.5).
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))

    class _MockDepthwiseConv(torch.nn.Module):
        def forward(self, x) -> torch.Tensor:
            return torch.zeros(1, 8, 4, 4)

    block.dwconv = _MockDepthwiseConv()

    class _MockCUDATensor:
        @property
        def device(self):
            class _Device:
                type = "cuda"
                index = 0

            return _Device()

    block._depthwise_conv(_MockCUDATensor())
    assert enabled_calls == [False], "cuDNN must be disabled on pre-Ampere GPUs (e.g. T4)"
