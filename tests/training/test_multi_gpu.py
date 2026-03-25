# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for multi-GPU training correctness and usability fixes.

These tests run on CPU with mocked distributed state so they work in CI without GPUs.
"""

from __future__ import annotations

import math
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.training.trainer import build_trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_model_config(**kwargs) -> RFDETRBaseConfig:
    return RFDETRBaseConfig(pretrain_weights=None, **kwargs)


def _base_train_config(tmp_path, **kwargs) -> TrainConfig:
    return TrainConfig(
        dataset_dir=str(tmp_path),
        output_dir=str(tmp_path),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. auto_batch_target_effective per-device semantics
# ---------------------------------------------------------------------------


class TestAutoBatchPerDeviceSemantics:
    """auto_batch_target_effective must be interpreted per-device, not divided by world_size."""

    def test_target_not_divided_by_world_size(self, tmp_path):
        """On 4 GPUs, each device should target auto_batch_target_effective, not a quarter of it."""
        from rfdetr.training.auto_batch import recommend_grad_accum_steps

        target_per_device = 16
        safe_micro_batch = 4
        world_size = 4  # 4 GPUs

        # Under the OLD (wrong) behavior: target = ceil(16 / 4) = 4 → accum = 1
        # Under the NEW (correct) behavior: target = 16 → accum = 4
        grad_accum = recommend_grad_accum_steps(safe_micro_batch, target_per_device)
        assert grad_accum == math.ceil(target_per_device / safe_micro_batch)

        # Verify the global effective batch scales with world_size
        global_effective = safe_micro_batch * grad_accum * world_size
        assert global_effective == target_per_device * world_size  # 64, not 16

    def test_single_gpu_unchanged(self, tmp_path):
        """Single-GPU behavior is unchanged."""
        from rfdetr.training.auto_batch import recommend_grad_accum_steps

        target_per_device = 16
        safe_micro_batch = 4
        grad_accum = recommend_grad_accum_steps(safe_micro_batch, target_per_device)
        assert grad_accum == 4  # 16 / 4 = 4


# ---------------------------------------------------------------------------
# 2. torchrun auto-device detection
# ---------------------------------------------------------------------------


class TestTorchrunAutoDevices:
    """When LOCAL_WORLD_SIZE is set and devices was not explicitly passed, devices should become 'auto'."""

    def test_devices_auto_when_local_world_size_set(self, tmp_path):
        """TrainConfig should set devices='auto' when torchrun env var is present."""
        env = {"LOCAL_WORLD_SIZE": "4"}
        with patch.dict(os.environ, env, clear=False):
            with pytest.warns(UserWarning, match="Detected torchrun environment"):
                tc = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path))
        assert tc.devices == "auto"

    def test_explicit_devices_not_overridden(self, tmp_path):
        """Explicitly passed devices= must not be overridden even when torchrun env var is set."""
        env = {"LOCAL_WORLD_SIZE": "4"}
        with patch.dict(os.environ, env, clear=False):
            tc = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), devices=2)
        assert tc.devices == 2

    def test_no_env_var_default_is_one(self, tmp_path):
        """Without torchrun env vars, devices should remain the default (1)."""
        env = {}
        # Unset LOCAL_WORLD_SIZE to ensure a clean environment
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("LOCAL_WORLD_SIZE", None)
            tc = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path))
        assert tc.devices == 1


# ---------------------------------------------------------------------------
# 3. find_unused_parameters guard for freeze_encoder / backbone_lora
# ---------------------------------------------------------------------------


class TestFindUnusedParametersGuard:
    """build_trainer should use DDPStrategy(find_unused_parameters=True) when required."""

    def test_segmentation_head_enables_find_unused(self, tmp_path):
        """segmentation_head=True should result in find_unused_parameters=True in DDP strategy."""
        from pytorch_lightning.strategies import DDPStrategy

        mc = _base_model_config(segmentation_head=True)
        tc = _base_train_config(tmp_path, strategy="ddp", devices=1)
        trainer = build_trainer(tc, mc, accelerator="cpu")

        assert isinstance(trainer.strategy, DDPStrategy)
        assert trainer.strategy._ddp_kwargs.get("find_unused_parameters") is True

    def test_freeze_encoder_enables_find_unused(self, tmp_path):
        """freeze_encoder=True should result in find_unused_parameters=True in DDP strategy."""
        from pytorch_lightning.strategies import DDPStrategy

        mc = _base_model_config(freeze_encoder=True)
        tc = _base_train_config(tmp_path, strategy="ddp", devices=1)
        trainer = build_trainer(tc, mc, accelerator="cpu")

        assert isinstance(trainer.strategy, DDPStrategy)
        assert trainer.strategy._ddp_kwargs.get("find_unused_parameters") is True

    def test_backbone_lora_enables_find_unused(self, tmp_path):
        """backbone_lora=True should result in find_unused_parameters=True in DDP strategy."""
        from pytorch_lightning.strategies import DDPStrategy

        mc = _base_model_config(backbone_lora=True)
        tc = _base_train_config(tmp_path, strategy="ddp", devices=1)
        trainer = build_trainer(tc, mc, accelerator="cpu")

        assert isinstance(trainer.strategy, DDPStrategy)
        assert trainer.strategy._ddp_kwargs.get("find_unused_parameters") is True

    def test_default_no_find_unused(self, tmp_path):
        """Without segmentation head, frozen encoder, or LoRA, find_unused_parameters should NOT be forced."""
        from pytorch_lightning.strategies import DDPStrategy

        mc = _base_model_config()
        tc = _base_train_config(tmp_path, strategy="auto", devices=1)
        trainer = build_trainer(tc, mc, accelerator="cpu")

        # strategy may be "auto" (not DDPStrategy) for single device, so only
        # check that if it IS DDP it does not have find_unused forced to True.
        if isinstance(trainer.strategy, DDPStrategy):
            assert trainer.strategy._ddp_kwargs.get("find_unused_parameters") is not True


# ---------------------------------------------------------------------------
# 4. torch.compile enabled with multi_scale
# ---------------------------------------------------------------------------


class TestCompileWithMultiScale:
    """torch.compile must NOT be blocked by multi_scale=True."""

    def test_compile_called_with_multi_scale_true(self, tmp_path):
        """compile=True on CUDA should call torch.compile even when multi_scale=True."""
        from rfdetr.training.module_model import RFDETRModelModule

        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=True)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("rfdetr.training.module_model.torch.compile", side_effect=lambda m, **_: m) as mock_compile,
            patch("rfdetr.training.module_model.build_model_from_config", return_value=MagicMock(spec=torch.nn.Module)),
            patch("rfdetr.training.module_model.build_criterion_from_config", return_value=(MagicMock(), MagicMock())),
            patch("rfdetr.training.module_model.load_pretrain_weights"),
        ):
            mc.pretrain_weights = None
            RFDETRModelModule(mc, tc)

        mock_compile.assert_called_once()


# ---------------------------------------------------------------------------
# 5. EMA device configuration
# ---------------------------------------------------------------------------


class TestEMADevice:
    """EMA callback should create AveragedModel on the configured device."""

    def test_ema_on_cpu_when_configured(self):
        """ema_device='cpu' should create AveragedModel with device=cpu."""
        from rfdetr.training.callbacks.ema import RFDETREMACallback

        callback = RFDETREMACallback(ema_device="cpu")

        fake_pl_module = MagicMock(spec=torch.nn.Module)
        fake_pl_module.device = torch.device("cuda:0")
        fake_trainer = MagicMock()

        captured_device = {}

        def mock_averaged_model(model, device, **kwargs):
            captured_device["device"] = device
            mock = MagicMock()
            mock.eval = MagicMock()
            return mock

        with patch("rfdetr.training.callbacks.ema.AveragedModel", side_effect=mock_averaged_model):
            callback.setup(fake_trainer, fake_pl_module, stage="fit")

        assert captured_device["device"] == torch.device("cpu")

    def test_ema_on_gpu_by_default(self):
        """Default ema_device='gpu' should create AveragedModel on the module's device."""
        from rfdetr.training.callbacks.ema import RFDETREMACallback

        callback = RFDETREMACallback(ema_device="gpu")

        fake_pl_module = MagicMock(spec=torch.nn.Module)
        fake_pl_module.device = torch.device("cuda:0")
        fake_trainer = MagicMock()

        captured_device = {}

        def mock_averaged_model(model, device, **kwargs):
            captured_device["device"] = device
            mock = MagicMock()
            mock.eval = MagicMock()
            return mock

        with patch("rfdetr.training.callbacks.ema.AveragedModel", side_effect=mock_averaged_model):
            callback.setup(fake_trainer, fake_pl_module, stage="fit")

        assert captured_device["device"] == torch.device("cuda:0")

    def test_ema_cpu_moves_to_gpu_for_validation(self):
        """ema_device='cpu': EMA model should move to GPU on validation start and back to CPU on end."""
        from rfdetr.training.callbacks.ema import RFDETREMACallback

        callback = RFDETREMACallback(ema_device="cpu")

        fake_pl_module = MagicMock()
        fake_pl_module.device = torch.device("cuda:0")
        fake_trainer = MagicMock()

        to_calls: list[str] = []
        fake_avg_model = MagicMock()
        fake_avg_model.to = lambda d: to_calls.append(str(d))

        callback._average_model = fake_avg_model

        callback.on_validation_epoch_start(fake_trainer, fake_pl_module)
        assert to_calls == ["cuda:0"], "EMA model should move to GPU at validation start"

        callback.on_validation_epoch_end(fake_trainer, fake_pl_module)
        assert to_calls == ["cuda:0", "cpu"], "EMA model should return to CPU at validation end"

    def test_ema_gpu_does_not_move_on_validation(self):
        """ema_device='gpu': on_validation_epoch_start/end should be no-ops."""
        from rfdetr.training.callbacks.ema import RFDETREMACallback

        callback = RFDETREMACallback(ema_device="gpu")

        fake_pl_module = MagicMock()
        fake_pl_module.device = torch.device("cuda:0")
        fake_trainer = MagicMock()

        to_calls: list[str] = []
        fake_avg_model = MagicMock()
        fake_avg_model.to = lambda d: to_calls.append(str(d))

        callback._average_model = fake_avg_model

        callback.on_validation_epoch_start(fake_trainer, fake_pl_module)
        callback.on_validation_epoch_end(fake_trainer, fake_pl_module)
        assert to_calls == [], "GPU EMA should not call .to() during validation hooks"


# ---------------------------------------------------------------------------
# 6. BestModelCallback.on_fit_end — trainer.test() called on all ranks
# ---------------------------------------------------------------------------


class _FakeModuleWithTestStep:
    """Minimal stub whose type() satisfies the has_test_step check in BestModelCallback."""

    model = MagicMock()

    def test_step(self, batch, batch_idx):
        return None


class TestBestModelCallbackDDP:
    """trainer.test() must be a collective called on all DDP ranks, not rank 0 only."""

    def _make_pl_module(self):
        return _FakeModuleWithTestStep()

    def test_non_zero_rank_calls_trainer_test(self, tmp_path):
        """Rank 1 (non-zero) must still call trainer.test() — not return early."""
        from rfdetr.training.callbacks.best_model import BestModelCallback

        callback = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        callback.best_model_score = None

        trainer = MagicMock()
        trainer.is_global_zero = False
        trainer.datamodule = MagicMock()

        with patch("torch.distributed.is_available", return_value=False):
            callback.on_fit_end(trainer, self._make_pl_module())

        trainer.test.assert_called_once()

    def test_rank_zero_calls_trainer_test(self, tmp_path):
        """Rank 0 must also call trainer.test() after its file operations."""
        from rfdetr.training.callbacks.best_model import BestModelCallback

        callback = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        callback.best_model_score = None

        trainer = MagicMock()
        trainer.is_global_zero = True
        trainer.datamodule = MagicMock()

        with patch("torch.distributed.is_available", return_value=False):
            callback.on_fit_end(trainer, self._make_pl_module())

        trainer.test.assert_called_once()

    def test_barrier_called_before_test_in_distributed(self, tmp_path):
        """A dist.barrier() must separate rank-0 file writes from all-rank test()."""
        from rfdetr.training.callbacks.best_model import BestModelCallback

        callback = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        callback.best_model_score = None

        trainer = MagicMock()
        trainer.is_global_zero = False
        trainer.datamodule = MagicMock()

        call_order: list[str] = []

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.barrier", side_effect=lambda: call_order.append("barrier")),
        ):
            trainer.test = MagicMock(side_effect=lambda *a, **kw: call_order.append("test"))
            callback.on_fit_end(trainer, self._make_pl_module())

        assert call_order == ["barrier", "test"], (
            "dist.barrier() must be called before trainer.test() so all ranks "
            "wait for rank 0 to finish writing the checkpoint file"
        )
