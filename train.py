"""Training pipeline entry point."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BEATS_CHECKPOINT,
    CONFIG,
    DEVICE,
    NUM_CLASSES,
    DataLoaderConfig,
    TrainConfig,
)
from dataloader import make_dataloaders
from dataset import CLASS_NAMES, ListenChannelDataset, class_weights_from_indices, prepare_train_split
from model import ListenChannelBeatsClassifier
from tracking import EpochMetrics, TrainingTracker, create_tracker, dataclass_params


def _unwrap(model: nn.Module) -> ListenChannelBeatsClassifier:
    return model.module if isinstance(model, nn.DataParallel) else model


def _progress_mode() -> str:
    """``auto`` | ``full`` | ``epoch`` | ``off`` — set via ``TRAIN_PROGRESS`` env var."""
    return os.environ.get("TRAIN_PROGRESS", "auto").lower()


def _show_batch_progress() -> bool:
    mode = _progress_mode()
    if mode in ("off", "epoch"):
        return False
    if mode == "full":
        return True
    return sys.stderr.isatty()


def _show_epoch_progress() -> bool:
    mode = _progress_mode()
    if mode == "off":
        return False
    if mode == "full":
        return True
    return sys.stderr.isatty()


def _batch_progress(iterable, *, desc: str):
    if not _show_batch_progress():
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        leave=False,
        mininterval=1.0,
        dynamic_ncols=True,
    )


def _update_batch_postfix(pbar, *, loss: float) -> None:
    if hasattr(pbar, "set_postfix"):
        pbar.set_postfix(loss=f"{loss:.4f}")


def _epoch_progress(iterable, *, desc: str, unit: str = "epoch"):
    if not _show_epoch_progress():
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, leave=True, mininterval=1.0, dynamic_ncols=True)


class EarlyStopping:
    """Stop when ``val_loss`` does not improve for ``patience`` epochs; restore best weights."""

    def __init__(self, *, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.epochs_without_improvement = 0

    def step(self, model: nn.Module, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(_unwrap(model).state_dict())
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            _unwrap(model).load_state_dict(self.best_state)


class BestCheckpointSaver:
    """Persist best weights to disk whenever validation loss improves."""

    def __init__(self, path: Path, *, min_delta: float = 0.0) -> None:
        self.path = path
        self.min_delta = min_delta
        self.best_loss = float("inf")

    def maybe_save(
        self,
        model: nn.Module,
        val_loss: float,
        *,
        phase: str,
        epoch: int,
        metrics: str,
    ) -> bool:
        if val_loss >= self.best_loss - self.min_delta:
            return False

        self.best_loss = val_loss
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": _unwrap(model).state_dict(),
                "class_names": CLASS_NAMES,
                "val_loss": val_loss,
                "phase": phase,
                "epoch": epoch,
                "metrics": metrics,
            },
            self.path,
        )
        print(f"  saved best  val_loss={val_loss:.4f}  -> {self.path}")
        return True

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> EpochMetrics:
    labels = list(range(NUM_CLASSES))
    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return EpochMetrics(
        acc=acc,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        per_class_precision={name: float(prec[i]) for i, name in enumerate(CLASS_NAMES)},
        per_class_recall={name: float(rec[i]) for i, name in enumerate(CLASS_NAMES)},
        per_class_f1={name: float(f1[i]) for i, name in enumerate(CLASS_NAMES)},
    )


def _format_metrics(metrics: EpochMetrics) -> str:
    per_class = " ".join(
        f"{name}:P={metrics.per_class_precision[name]:.2f}"
        f"/R={metrics.per_class_recall[name]:.2f}"
        f"/F1={metrics.per_class_f1[name]:.2f}"
        for name in CLASS_NAMES
    )
    return (
        f"acc={metrics.acc:.3f}  "
        f"P={metrics.macro_precision:.3f}  "
        f"R={metrics.macro_recall:.3f}  "
        f"F1={metrics.macro_f1:.3f}  "
        f"({per_class})"
    )


def _format_lrs(optimizer: torch.optim.Optimizer) -> str:
    return "/".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)


def _make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.lr_scheduler_min_lr,
    )


@torch.no_grad()
def _eval_epoch(
    model: ListenChannelBeatsClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    desc: str = "val",
) -> tuple[float, EpochMetrics]:
    model.eval()
    total_loss = 0.0
    total = 0
    labels_all: list[int] = []
    preds_all: list[int] = []
    pbar = _batch_progress(loader, desc=desc)
    for batch in pbar:
        fbank = batch["fbank"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        logits = model(fbank)["logits"]
        total_loss += float(criterion(logits, labels).item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        labels_all.extend(labels.cpu().tolist())
        preds_all.extend(preds.cpu().tolist())
        total += labels.numel()
        _update_batch_postfix(pbar, loss=total_loss / total)
    y_true = np.asarray(labels_all, dtype=np.int64)
    y_pred = np.asarray(preds_all, dtype=np.int64)
    metrics = _compute_metrics(y_true, y_pred)
    return total_loss / max(total, 1), metrics


def _train_epoch(
    model: ListenChannelBeatsClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    desc: str = "train",
) -> float:
    model.train()
    base = _unwrap(model)
    if not any(p.requires_grad for p in base.encoder.parameters()):
        base.encoder.eval()

    total_loss = 0.0
    total = 0
    pbar = _batch_progress(loader, desc=desc)
    for batch in pbar:
        fbank = batch["fbank"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(fbank)["logits"]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)
        _update_batch_postfix(pbar, loss=total_loss / total)
    return total_loss / max(total, 1)


def _run_phase(
    model: ListenChannelBeatsClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    phase_name: str,
    max_epochs: int,
    early_stopping: EarlyStopping,
    checkpoint_saver: BestCheckpointSaver | None = None,
    tracker: TrainingTracker | None = None,
) -> None:
    for epoch in _epoch_progress(range(1, max_epochs + 1), desc=phase_name, unit="epoch"):
        train_loss = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            desc=f"{phase_name} e{epoch:02d} train",
        )
        val_loss, val_metrics = _eval_epoch(
            model,
            val_loader,
            criterion,
            desc=f"{phase_name} e{epoch:02d} val",
        )
        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        if tracker is not None and tracker.enabled:
            tracker.log_epoch(
                phase=phase_name,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=lr,
                metrics=val_metrics,
            )
        if checkpoint_saver is not None:
            saved = checkpoint_saver.maybe_save(
                model,
                val_loss,
                phase=phase_name,
                epoch=epoch,
                metrics=_format_metrics(val_metrics),
            )
            if saved and tracker is not None and tracker.enabled and checkpoint_saver.path.is_file():
                tracker.log_checkpoint(checkpoint_saver.path)
        stop = early_stopping.step(model, val_loss)
        print(
            f"  {phase_name} epoch {epoch:02d}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={_format_lrs(optimizer)}  {_format_metrics(val_metrics)}"
            + ("  [stop]" if stop else "")
        )
        if stop:
            break

    early_stopping.restore_best(model)
    best_loss, best_metrics = _eval_epoch(
        model,
        val_loader,
        criterion,
        desc=f"{phase_name} best val",
    )
    print(f"  {phase_name} best  val_loss={best_loss:.4f}  {_format_metrics(best_metrics)}")


def train_model(
    model: ListenChannelBeatsClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    config: TrainConfig,
    class_weights: torch.Tensor | None = None,
    output: Path | None = None,
    tracker: TrainingTracker | None = None,
) -> ListenChannelBeatsClassifier:
    model = model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    checkpoint_saver = (
        BestCheckpointSaver(output, min_delta=config.min_delta) if output is not None else None
    )
    active_tracker = tracker or TrainingTracker(enabled=False)

    print(
        f"phase 1: head only, lr={config.head_lr}, "
        f"max_epochs={config.head_max_epochs}, patience={config.patience}"
    )
    head_stop = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    optimizer = torch.optim.AdamW(
        [_unwrap(model).head_parameter_group(lr=config.head_lr)],
        weight_decay=config.weight_decay,
    )
    head_scheduler = _make_lr_scheduler(optimizer, config)
    _run_phase(
        model,
        train_loader,
        val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=head_scheduler,
        phase_name="head",
        max_epochs=config.head_max_epochs,
        early_stopping=head_stop,
        checkpoint_saver=checkpoint_saver,
        tracker=active_tracker,
    )

    print(
        f"phase 2: unfreeze last {config.unfreeze_last_n_layers} encoder layers, "
        f"head_lr={config.head_lr}, encoder_lr={config.encoder_lr}, "
        f"max_epochs={config.encoder_max_epochs}, patience={config.patience}"
    )
    _unwrap(model).begin_encoder_finetune(last_n_layers=config.unfreeze_last_n_layers)
    encoder_stop = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    optimizer = torch.optim.AdamW(
        _unwrap(model).parameter_groups(
            encoder_lr=config.encoder_lr,
            head_lr=config.head_lr,
        ),
        weight_decay=config.weight_decay,
    )
    encoder_scheduler = _make_lr_scheduler(optimizer, config)
    _run_phase(
        model,
        train_loader,
        val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=encoder_scheduler,
        phase_name="enc",
        max_epochs=config.encoder_max_epochs,
        early_stopping=encoder_stop,
        checkpoint_saver=checkpoint_saver,
        tracker=active_tracker,
    )

    return _unwrap(model)


def _cmd_inspect(args: argparse.Namespace) -> None:
    from collections import Counter

    dataset = ListenChannelDataset(args.dataset)
    print(f"sessions: {dataset.num_sessions}")
    print(f"clips:    {dataset.num_clips}")
    print(f"windows:  {len(dataset)}")

    clip_labels = Counter(int(clip.label) for clip in dataset.clips)
    windows_per_clip = Counter(window.clip_index for window in dataset.windows)
    window_labels: Counter[int] = Counter()
    for clip_idx, count in windows_per_clip.items():
        window_labels[int(dataset.clips[clip_idx].label)] += count
    print(f"clip labels:   {dict(sorted(clip_labels.items()))}")
    print(f"window labels: {dict(sorted(window_labels.items()))}")

    if args.val_ratio > 0:
        train_ds, val_ds = prepare_train_split(dataset, val_ratio=args.val_ratio)
        print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}")
        print(f"augment: on for {len(train_ds.indices)} train windows")
        print(f"background pool: {len(dataset._train_background_indices)} train windows")

    if len(dataset) > 0:
        sample = dataset[0]
        clip = dataset.clips[dataset.windows[0].clip_index]
        print(
            f"sample: fbank={tuple(sample['fbank'].shape)} "
            f"label={CLASS_NAMES[int(sample['label'])]} "
            f"session={clip.session.session_id} channel={clip.channel_key}"
        )


def _cmd_train(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BEATs checkpoint not found: {checkpoint}")

    train_cfg = TrainConfig(
        head_max_epochs=args.head_max_epochs,
        encoder_max_epochs=args.encoder_max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        head_lr=args.head_lr,
        encoder_lr=args.encoder_lr,
        unfreeze_last_n_layers=args.unfreeze_layers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        mlflow_enabled=args.mlflow,
        mlflow_tracking_uri=str(args.mlflow_uri),
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
        plot_path=str(args.plot_path),
        output_checkpoint=str(args.output),
        loader=DataLoaderConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_clips=args.cache_clips,
        ),
    )

    train_loader, val_loader, dataset, train_ds, _ = make_dataloaders(
        args.dataset,
        loader_config=train_cfg.loader,
        val_ratio=train_cfg.val_ratio,
        seed=train_cfg.seed,
        cache_clips=train_cfg.loader.cache_clips,
    )
    class_weights = class_weights_from_indices(dataset, train_ds.indices).to(DEVICE)

    model = ListenChannelBeatsClassifier.from_checkpoint(checkpoint)
    with create_tracker(
        enabled=train_cfg.mlflow_enabled,
        tracking_uri=train_cfg.mlflow_tracking_uri,
        experiment_name=train_cfg.mlflow_experiment,
        run_name=train_cfg.mlflow_run_name,
        plot_path=train_cfg.plot_path,
    ) as tracker:
        tracker.log_params(
            {
                **dataclass_params(train_cfg),
                **dataclass_params(train_cfg.loader),
                "dataset": str(args.dataset),
                "beats_checkpoint": str(checkpoint),
                "output": str(args.output),
                "device": DEVICE,
            }
        )
        train_model(
            model,
            train_loader,
            val_loader,
            config=train_cfg,
            class_weights=class_weights,
            output=args.output,
            tracker=tracker,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="x502 listen-channel training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="summarize ListenChannelDataset")
    p_inspect.add_argument("dataset", type=Path, help="Dataset root (session folders)")
    p_inspect.add_argument("--val-ratio", type=float, default=CONFIG.val_ratio)
    p_inspect.set_defaults(func=_cmd_inspect)

    p_train = sub.add_parser("train", help="two-phase BEATs fine-tune (head → encoder)")
    p_train.add_argument("dataset", type=Path)
    p_train.add_argument("--checkpoint", type=Path, default=Path(BEATS_CHECKPOINT))
    p_train.add_argument("--output", type=Path, default=Path(CONFIG.output_checkpoint))
    p_train.add_argument("--batch-size", type=int, default=CONFIG.loader.batch_size)
    p_train.add_argument("--num-workers", type=int, default=CONFIG.loader.num_workers)
    p_train.add_argument(
        "--cache-clips",
        action="store_true",
        default=CONFIG.loader.cache_clips,
        help="keep decoded WAV in RAM (~9 GiB train; duplicates per DataLoader worker)",
    )
    p_train.add_argument("--head-max-epochs", type=int, default=CONFIG.head_max_epochs)
    p_train.add_argument("--encoder-max-epochs", type=int, default=CONFIG.encoder_max_epochs)
    p_train.add_argument("--patience", type=int, default=CONFIG.patience)
    p_train.add_argument("--min-delta", type=float, default=CONFIG.min_delta)
    p_train.add_argument("--head-lr", type=float, default=CONFIG.head_lr)
    p_train.add_argument("--encoder-lr", type=float, default=CONFIG.encoder_lr)
    p_train.add_argument("--unfreeze-layers", type=int, default=CONFIG.unfreeze_last_n_layers)
    p_train.add_argument("--val-ratio", type=float, default=CONFIG.val_ratio)
    p_train.add_argument("--seed", type=int, default=CONFIG.seed)
    p_train.add_argument(
        "--mlflow",
        action="store_true",
        default=CONFIG.mlflow_enabled,
        help="log metrics/plots to MLflow",
    )
    p_train.add_argument("--mlflow-uri", default=CONFIG.mlflow_tracking_uri)
    p_train.add_argument("--mlflow-experiment", default=CONFIG.mlflow_experiment)
    p_train.add_argument("--mlflow-run-name", default=CONFIG.mlflow_run_name)
    p_train.add_argument("--plot-path", type=Path, default=Path(CONFIG.plot_path))
    p_train.set_defaults(func=_cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
