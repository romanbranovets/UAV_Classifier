"""Optional MLflow experiment tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from dataset import CLASS_NAMES


@dataclass
class EpochMetrics:
    acc: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]

    def as_log_dict(self, *, prefix: str = "val") -> dict[str, float]:
        out = {
            f"{prefix}_acc": self.acc,
            f"{prefix}_macro_precision": self.macro_precision,
            f"{prefix}_macro_recall": self.macro_recall,
            f"{prefix}_macro_f1": self.macro_f1,
        }
        for name in CLASS_NAMES:
            out[f"{prefix}_{name}_precision"] = self.per_class_precision[name]
            out[f"{prefix}_{name}_recall"] = self.per_class_recall[name]
            out[f"{prefix}_{name}_f1"] = self.per_class_f1[name]
        return out


class TrainingTracker:
    """No-op tracker; swap for ``MlflowTracker`` when MLflow is enabled."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def __enter__(self) -> TrainingTracker:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def log_params(self, params: dict[str, Any]) -> None:
        return None

    def log_epoch(
        self,
        *,
        phase: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        metrics: EpochMetrics,
    ) -> None:
        return None

    def log_checkpoint(self, path: Path) -> None:
        return None

    def save_plots(self, path: Path) -> None:
        return None


class MlflowTracker(TrainingTracker):
    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        run_name: Optional[str] = None,
    ) -> None:
        super().__init__(enabled=True)
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        self._history: list[dict[str, float]] = []
        self._step = 0

    def __exit__(self, *args: object) -> None:
        if self._history:
            self._write_plot()
        self._mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        flat = {key: value for key, value in params.items() if value is not None}
        self._mlflow.log_params(flat)

    def log_epoch(
        self,
        *,
        phase: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        metrics: EpochMetrics,
    ) -> None:
        self._step += 1
        row = {
            "step": float(self._step),
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            **metrics.as_log_dict(prefix="val"),
            "phase_head": float(phase == "head"),
            "phase_enc": float(phase == "enc"),
        }
        self._history.append(row)
        self._mlflow.log_metrics(row, step=self._step)
        self._mlflow.set_tag("last_phase", phase)

    def log_checkpoint(self, path: Path) -> None:
        if path.is_file():
            self._mlflow.log_artifact(str(path), artifact_path="checkpoints")

    def save_plots(self, path: Path) -> None:
        if not self._history:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_training_plot(self._history, path)
        self._mlflow.log_artifact(str(path), artifact_path="plots")

    def _write_plot(self) -> None:
        plot_path = Path("training_curves.png")
        _save_training_plot(self._history, plot_path)
        self._mlflow.log_artifact(str(plot_path), artifact_path="plots")


def create_tracker(
    *,
    enabled: bool,
    tracking_uri: str,
    experiment_name: str,
    run_name: Optional[str] = None,
) -> TrainingTracker:
    if not enabled:
        return TrainingTracker(enabled=False)
    return MlflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
    )


def _save_training_plot(history: list[dict[str, float]], path: Path) -> None:
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_f1 = [row["val_macro_f1"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(steps, train_loss, label="train")
    axes[0].plot(steps, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("step")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, val_f1, label="val macro F1", color="tab:green")
    for name in CLASS_NAMES:
        key = f"val_{name}_f1"
        if key in history[0]:
            axes[1].plot(steps, [row[key] for row in history], label=f"{name} F1")
    axes[1].set_title("Validation F1")
    axes[1].set_xlabel("step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def dataclass_params(obj: Any) -> dict[str, Any]:
    return {key: value for key, value in asdict(obj).items() if not key.startswith("_")}
