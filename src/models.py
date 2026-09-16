"""Model objects used by the article's HuPin and median-QR workflows."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class HuberPinballLoss(nn.Module):
    """Submitted elementwise Huber plus weighted pinball loss."""

    def __init__(self, beta: float, tau: float, lam: float):
        super().__init__()
        self.beta = float(beta)
        self.tau = float(tau)
        self.lam = float(lam)
        self.huber = nn.SmoothL1Loss(beta=self.beta, reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber_term = self.huber(pred, target)
        residual = target - pred
        pinball_term = torch.where(
            residual >= 0,
            self.tau * residual,
            (self.tau - 1.0) * residual,
        )
        return huber_term.mean() + self.lam * pinball_term.mean()


def build_article_lstm(
    *,
    lookback: int,
    hidden_dim: int,
    n_rnn_layers: int,
    dropout: float,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    random_state: int,
    loss_fn: nn.Module,
    early_patience: int | None = None,
    early_min_delta: float = 1e-6,
    progress_bar: bool = False,
) -> Any:
    """Construct the corrected article LSTM for tuning or final refitting.

    Passing ``early_patience`` enables validation-based early stopping during
    tuning.  Passing ``None`` removes validation callbacks so that a selected
    configuration can be refit on the complete training panel for a fixed
    number of epochs.
    """

    from darts.models import BlockRNNModel

    callbacks = []
    if early_patience is not None:
        from pytorch_lightning.callbacks import EarlyStopping

        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=int(early_patience),
                min_delta=float(early_min_delta),
                mode="min",
            )
        )

    return BlockRNNModel(
        model="LSTM",
        input_chunk_length=int(lookback),
        output_chunk_length=1,
        hidden_dim=int(hidden_dim),
        n_rnn_layers=int(n_rnn_layers),
        dropout=float(dropout),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        optimizer_kwargs={"lr": float(learning_rate)},
        random_state=int(random_state),
        likelihood=None,
        loss_fn=loss_fn,
        pl_trainer_kwargs={
            "enable_progress_bar": bool(progress_bar),
            "logger": False,
            "log_every_n_steps": 1,
            "accelerator": "cpu",
            "devices": 1,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "enable_model_summary": False,
            "deterministic": True,
        },
    )


def build_submitted_quantile_regression(
    *,
    lookback: int,
    alpha: float,
    tau: float,
) -> Any:
    """Construct the article's joint median quantile-regression model."""

    from darts.models import SKLearnModel
    from sklearn.linear_model import QuantileRegressor

    estimator = QuantileRegressor(
        quantile=tau,
        alpha=alpha,
        fit_intercept=True,
        solver="highs",
    )
    return SKLearnModel(
        model=estimator,
        lags=lookback,
        lags_past_covariates=[-1],
        output_chunk_length=1,
    )


__all__ = [
    "HuberPinballLoss",
    "build_article_lstm",
    "build_submitted_quantile_regression",
]
