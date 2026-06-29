"""Callback for tracking wall-clock time spent in training phases."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class _PhaseAccumulator:
    """Running totals for one training epoch (seconds)."""

    data_wait: float = 0.0
    forward: float = 0.0
    backward: float = 0.0
    optimizer: float = 0.0
    validation: float = 0.0

    def total_tracked(self) -> float:
        return self.data_wait + self.forward + self.backward + self.optimizer + self.validation


class TrainingPhaseTimer(Callback):
    """Track wall-clock time spent in major training phases.

    Phases:
        - ``data_wait``: time waiting for the next training batch (dataloader + host overhead)
        - ``forward``: ``training_step`` up to ``loss.backward()``
        - ``backward``: ``loss.backward()`` through gradient clipping
        - ``optimizer``: ``optimizer.step()`` and related sync
        - ``validation``: full validation loop each epoch
        - ``other``: remaining epoch wall time (checkpointing, logging, etc.)

    Timing uses ``time.perf_counter()`` and optionally synchronizes CUDA before each
    measurement so GPU kernels are included in the elapsed time.

    Logged metrics (per epoch, seconds and fraction of epoch wall time):
        ``time/data_wait_sec``, ``time/forward_sec``, ``time/backward_sec``,
        ``time/optimizer_sec``, ``time/validation_sec``, ``time/other_sec``,
        ``time/epoch_sec``, and matching ``time/*_pct`` keys.
    """

    def __init__(self, sync_cuda: bool = True, log_fractions: bool = True) -> None:
        super().__init__()
        self.sync_cuda = sync_cuda
        self.log_fractions = log_fractions

        self._epoch: _PhaseAccumulator = _PhaseAccumulator()
        self._fit_totals: Dict[str, float] = {
            "data_wait": 0.0,
            "forward": 0.0,
            "backward": 0.0,
            "optimizer": 0.0,
            "validation": 0.0,
            "other": 0.0,
            "epoch": 0.0,
        }

        self._epoch_start: Optional[float] = None
        self._last_train_batch_end: Optional[float] = None
        self._forward_start: Optional[float] = None
        self._backward_start: Optional[float] = None
        self._optimizer_start: Optional[float] = None
        self._validation_start: Optional[float] = None
        self._in_validation: bool = False
        self._epoch_logged: bool = False

    def _now(self) -> float:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _reset_epoch_state(self) -> None:
        self._epoch = _PhaseAccumulator()
        self._epoch_start = None
        self._last_train_batch_end = None
        self._forward_start = None
        self._backward_start = None
        self._optimizer_start = None
        self._validation_start = None
        self._in_validation = False
        self._epoch_logged = False

    def _finalize_epoch(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._epoch_logged or self._epoch_start is None:
            return
        epoch_wall = self._now() - self._epoch_start
        self._log_epoch_metrics(trainer, pl_module, epoch_wall)
        self._epoch_logged = True

    @rank_zero_only
    def _log_epoch_metrics(self, trainer: Trainer, pl_module: LightningModule, epoch_wall: float) -> None:
        tracked = self._epoch.total_tracked()
        other = max(epoch_wall - tracked, 0.0)

        metrics = {
            "time/data_wait_sec": self._epoch.data_wait,
            "time/forward_sec": self._epoch.forward,
            "time/backward_sec": self._epoch.backward,
            "time/optimizer_sec": self._epoch.optimizer,
            "time/validation_sec": self._epoch.validation,
            "time/other_sec": other,
            "time/epoch_sec": epoch_wall,
        }

        if self.log_fractions and epoch_wall > 0:
            for phase, seconds in metrics.items():
                if phase == "time/epoch_sec":
                    continue
                key = phase.replace("_sec", "_pct")
                metrics[key] = seconds / epoch_wall

        for name, value in metrics.items():
            pl_module.log(name, value, on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

        self._fit_totals["data_wait"] += self._epoch.data_wait
        self._fit_totals["forward"] += self._epoch.forward
        self._fit_totals["backward"] += self._epoch.backward
        self._fit_totals["optimizer"] += self._epoch.optimizer
        self._fit_totals["validation"] += self._epoch.validation
        self._fit_totals["other"] += other
        self._fit_totals["epoch"] += epoch_wall

        log.info(
            "Epoch %d timing (s): data_wait=%.2f forward=%.2f backward=%.2f "
            "optimizer=%.2f validation=%.2f other=%.2f total=%.2f",
            trainer.current_epoch,
            self._epoch.data_wait,
            self._epoch.forward,
            self._epoch.backward,
            self._epoch.optimizer,
            self._epoch.validation,
            other,
            epoch_wall,
        )

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._reset_epoch_state()
        self._epoch_start = self._now()

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: object, batch_idx: int
    ) -> None:
        now = self._now()
        if self._last_train_batch_end is not None:
            self._epoch.data_wait += now - self._last_train_batch_end
        elif self._epoch_start is not None:
            # First batch of the epoch: includes fetching the first batch.
            self._epoch.data_wait += now - self._epoch_start
        self._forward_start = now

    def on_before_backward(self, trainer: Trainer, pl_module: LightningModule, loss: torch.Tensor) -> None:
        now = self._now()
        if self._forward_start is not None:
            self._epoch.forward += now - self._forward_start
            self._forward_start = None
        self._backward_start = now

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        now = self._now()
        if self._backward_start is not None:
            self._epoch.backward += now - self._backward_start
            self._backward_start = None

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: torch.optim.Optimizer
    ) -> None:
        self._optimizer_start = self._now()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        now = self._now()
        if self._optimizer_start is not None:
            self._epoch.optimizer += now - self._optimizer_start
            self._optimizer_start = None
        self._last_train_batch_end = now

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._in_validation = True
        self._validation_start = self._now()

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or not self._in_validation or self._validation_start is None:
            return
        self._epoch.validation += self._now() - self._validation_start
        self._validation_start = None
        self._in_validation = False
        self._finalize_epoch(trainer, pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        should_validate = (
            trainer.enable_validation
            and trainer.val_dataloaders is not None
            and (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch == 0
        )
        if not should_validate:
            self._finalize_epoch(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        total = self._fit_totals["epoch"]
        if total <= 0:
            return

        log.info(
            "Training timing summary (s): data_wait=%.1f forward=%.1f backward=%.1f "
            "optimizer=%.1f validation=%.1f other=%.1f total=%.1f",
            self._fit_totals["data_wait"],
            self._fit_totals["forward"],
            self._fit_totals["backward"],
            self._fit_totals["optimizer"],
            self._fit_totals["validation"],
            self._fit_totals["other"],
            total,
        )

        for phase in ("data_wait", "forward", "backward", "optimizer", "validation", "other"):
            pl_module.log(
                f"time/fit_{phase}_sec",
                self._fit_totals[phase],
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        pl_module.log("time/fit_total_sec", total, on_step=False, on_epoch=True, sync_dist=False)
