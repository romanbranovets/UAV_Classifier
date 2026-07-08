"""Training pipeline entry point."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import replace
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
    DADS_HEAD_LR,
    DADS_HF_DATASET,
    DADS_INDEX_CACHE_DIR,
    DADS_PRETRAIN_CHECKPOINT,
    DADS_VAL_RATIO,
    DEVICE,
    FINETUNE_HEAD_LR,
    NUM_CLASSES,
    BeatsClassifierConfig,
    MODEL_CONFIG,
    DataLoaderConfig,
    TrainConfig,
)
from dataloader import make_dads_dataloaders, make_dataloaders
from dataset import CLASS_NAMES, Label, ListenChannelDataset, prepare_train_split
from model import ListenChannelBeatsClassifier, SupConLoss
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
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    pos_p, pos_r, pos_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=int(Label.UAV),
        zero_division=0,
    )
    return EpochMetrics(
        acc=acc,
        precision=float(pos_p),
        recall=float(pos_r),
        f1=float(pos_f1),
        per_class_precision={name: float(prec[i]) for i, name in enumerate(CLASS_NAMES)},
        per_class_recall={name: float(rec[i]) for i, name in enumerate(CLASS_NAMES)},
        per_class_f1={name: float(f1[i]) for i, name in enumerate(CLASS_NAMES)},
    )


def _format_metrics(metrics: EpochMetrics) -> str:
    per_class = " ".join(
        f"{name}:P={metrics.per_class_precision[name]:.2f}"
        f"/R={metrics.per_class_recall[name]:.2f}"
        for name in CLASS_NAMES
    )
    return (
        f"acc={metrics.acc:.3f}  "
        f"P={metrics.precision:.3f}  "
        f"R={metrics.recall:.3f}  "
        f"F1={metrics.f1:.3f}  "
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


def _forward_loss(
    model: nn.Module,
    fbank: torch.Tensor,
    labels: torch.Tensor,
    criterion: SupConLoss,
    *,
    update_prototypes: bool,
) -> torch.Tensor:
    base = _unwrap(model)
    out = model(fbank)
    projection = out["projection"]
    if update_prototypes:
        base.prototypes.update(projection.detach(), labels)
    return criterion(projection, labels)


@torch.no_grad()
def _eval_epoch(
    model: ListenChannelBeatsClassifier,
    loader: DataLoader,
    criterion: SupConLoss,
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
        out = model(fbank)
        total_loss += float(criterion(out["projection"], labels).item()) * labels.size(0)
        preds = out["logits"].argmax(dim=1)
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
    criterion: SupConLoss,
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
        loss = _forward_loss(model, fbank, labels, criterion, update_prototypes=True)
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
    criterion: SupConLoss,
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


def load_classifier_checkpoint(model: ListenChannelBeatsClassifier, path: Path) -> dict:
    """Load classifier weights saved by ``BestCheckpointSaver``."""
    if not path.is_file():
        raise FileNotFoundError(f"classifier checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if state is None:
        raise ValueError(f"checkpoint missing 'model' state: {path}")
    model.load_state_dict(state)
    return payload


def train_model(
    model: ListenChannelBeatsClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    config: TrainConfig,
    output: Path | None = None,
    tracker: TrainingTracker | None = None,
    encoder_finetune: bool = True,
) -> ListenChannelBeatsClassifier:
    model = model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    base = _unwrap(model)
    criterion = SupConLoss(temperature=base.config.supcon_temperature)
    print(
        f"loss: supcon  proj_dim={base.config.proj_dim}  "
        f"temperature={base.config.supcon_temperature}"
    )

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

    if not encoder_finetune:
        return _unwrap(model)

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


def _build_model(
    checkpoint: Path,
    *,
    proj_dim: int,
    supcon_temperature: float,
    prototype_momentum: float,
    init_checkpoint: Path | None = None,
) -> ListenChannelBeatsClassifier:
    model_cfg = BeatsClassifierConfig(
        proj_dim=proj_dim,
        supcon_temperature=supcon_temperature,
        prototype_momentum=prototype_momentum,
    )
    model = ListenChannelBeatsClassifier.from_checkpoint(checkpoint, config=model_cfg)
    if init_checkpoint is not None:
        meta = load_classifier_checkpoint(model, init_checkpoint)
        print(f"loaded init checkpoint: {init_checkpoint}  (phase={meta.get('phase')}, epoch={meta.get('epoch')})")
    return model


def _train_cfg_from_args(args: argparse.Namespace, *, default_head_lr: float) -> TrainConfig:
    head_lr = args.head_lr if getattr(args, "head_lr", None) is not None else default_head_lr
    return TrainConfig(
        head_max_epochs=args.head_max_epochs,
        encoder_max_epochs=args.encoder_max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        head_lr=head_lr,
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
            cache_clips=getattr(args, "cache_clips", False),
        ),
    )


def _add_train_hparams(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, default=Path(BEATS_CHECKPOINT), help="BEATs encoder .pt")
    parser.add_argument("--output", type=Path, default=Path(CONFIG.output_checkpoint))
    parser.add_argument("--batch-size", type=int, default=CONFIG.loader.batch_size)
    parser.add_argument("--num-workers", type=int, default=CONFIG.loader.num_workers)
    parser.add_argument("--head-max-epochs", type=int, default=CONFIG.head_max_epochs)
    parser.add_argument("--encoder-max-epochs", type=int, default=CONFIG.encoder_max_epochs)
    parser.add_argument("--patience", type=int, default=CONFIG.patience)
    parser.add_argument("--min-delta", type=float, default=CONFIG.min_delta)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--proj-dim", type=int, default=MODEL_CONFIG.proj_dim)
    parser.add_argument("--supcon-temperature", type=float, default=MODEL_CONFIG.supcon_temperature)
    parser.add_argument("--prototype-momentum", type=float, default=MODEL_CONFIG.prototype_momentum)
    parser.add_argument("--encoder-lr", type=float, default=CONFIG.encoder_lr)
    parser.add_argument("--unfreeze-layers", type=int, default=CONFIG.unfreeze_last_n_layers)
    parser.add_argument("--val-ratio", type=float, default=CONFIG.val_ratio)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--mlflow", action="store_true", default=CONFIG.mlflow_enabled)
    parser.add_argument("--mlflow-uri", default=CONFIG.mlflow_tracking_uri)
    parser.add_argument("--mlflow-experiment", default=CONFIG.mlflow_experiment)
    parser.add_argument("--mlflow-run-name", default=CONFIG.mlflow_run_name)
    parser.add_argument("--plot-path", type=Path, default=Path(CONFIG.plot_path))


def _encoder_finetune_enabled(args: argparse.Namespace, *, pretrain: bool = False) -> bool:
    if pretrain:
        return bool(getattr(args, "encoder_finetune", False))
    return not bool(getattr(args, "head_only", False))


def _cmd_pretrain(args: argparse.Namespace) -> None:
    beats_ckpt = args.checkpoint
    if not beats_ckpt.is_file():
        raise FileNotFoundError(f"BEATs checkpoint not found: {beats_ckpt}")

    train_cfg = replace(
        _train_cfg_from_args(args, default_head_lr=DADS_HEAD_LR),
        val_ratio=args.val_ratio,
        output_checkpoint=str(args.output),
    )

    train_loader, val_loader, dataset, train_ds, val_ds = make_dads_dataloaders(
        hf_id=args.dads_dataset,
        cache_dir=args.hf_cache_dir,
        index_cache_dir=args.index_cache_dir,
        loader_config=train_cfg.loader,
        val_ratio=train_cfg.val_ratio,
        seed=train_cfg.seed,
        max_clips=args.max_clips,
        balance_train=not args.no_balance,
    )
    print(
        f"DADS clips={dataset.num_clips} windows={len(dataset)} "
        f"train={len(train_ds)} val={len(val_ds)}"
    )

    model = _build_model(
        beats_ckpt,
        proj_dim=args.proj_dim,
        supcon_temperature=args.supcon_temperature,
        prototype_momentum=args.prototype_momentum,
    )
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
                "stage": "dads_pretrain",
                "dads_dataset": args.dads_dataset,
                "beats_checkpoint": str(beats_ckpt),
                "output": str(args.output),
                "device": DEVICE,
                "max_clips": args.max_clips,
            }
        )
        train_model(
            model,
            train_loader,
            val_loader,
            config=train_cfg,
            output=args.output,
            tracker=tracker,
            encoder_finetune=_encoder_finetune_enabled(args, pretrain=True),
        )


def _cmd_finetune(args: argparse.Namespace) -> None:
    beats_ckpt = args.checkpoint
    if not beats_ckpt.is_file():
        raise FileNotFoundError(f"BEATs checkpoint not found: {beats_ckpt}")
    if args.init_checkpoint is None:
        raise ValueError("finetune requires --init-checkpoint from DADS pretrain stage")

    train_cfg = _train_cfg_from_args(args, default_head_lr=FINETUNE_HEAD_LR)

    train_loader, val_loader, dataset, train_ds, val_ds = make_dataloaders(
        args.dataset,
        loader_config=train_cfg.loader,
        val_ratio=train_cfg.val_ratio,
        seed=train_cfg.seed,
        cache_clips=train_cfg.loader.cache_clips,
    )
    print(
        f"operational sessions={dataset.num_sessions} clips={dataset.num_clips} "
        f"windows={len(dataset)} train={len(train_ds)} val={len(val_ds)}"
    )

    model = _build_model(
        beats_ckpt,
        proj_dim=args.proj_dim,
        supcon_temperature=args.supcon_temperature,
        prototype_momentum=args.prototype_momentum,
        init_checkpoint=args.init_checkpoint,
    )
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
                "stage": "operational_finetune",
                "dataset": str(args.dataset),
                "beats_checkpoint": str(beats_ckpt),
                "init_checkpoint": str(args.init_checkpoint),
                "output": str(args.output),
                "device": DEVICE,
            }
        )
        train_model(
            model,
            train_loader,
            val_loader,
            config=train_cfg,
            output=args.output,
            tracker=tracker,
            encoder_finetune=_encoder_finetune_enabled(args),
        )


def _cmd_train(args: argparse.Namespace) -> None:
    """Single-stage training on operational data (no DADS pretrain)."""
    beats_ckpt = args.checkpoint
    if not beats_ckpt.is_file():
        raise FileNotFoundError(f"BEATs checkpoint not found: {beats_ckpt}")

    default_head_lr = CONFIG.head_lr
    train_cfg = _train_cfg_from_args(args, default_head_lr=default_head_lr)

    train_loader, val_loader, _, _, _ = make_dataloaders(
        args.dataset,
        loader_config=train_cfg.loader,
        val_ratio=train_cfg.val_ratio,
        seed=train_cfg.seed,
        cache_clips=train_cfg.loader.cache_clips,
    )

    model = _build_model(
        beats_ckpt,
        proj_dim=args.proj_dim,
        supcon_temperature=args.supcon_temperature,
        prototype_momentum=args.prototype_momentum,
        init_checkpoint=getattr(args, "init_checkpoint", None),
    )
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
                "stage": "operational_train",
                "dataset": str(args.dataset),
                "beats_checkpoint": str(beats_ckpt),
                "output": str(args.output),
                "device": DEVICE,
            }
        )
        train_model(
            model,
            train_loader,
            val_loader,
            config=train_cfg,
            output=args.output,
            tracker=tracker,
            encoder_finetune=_encoder_finetune_enabled(args),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="x502 listen-channel training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="summarize ListenChannelDataset")
    p_inspect.add_argument("dataset", type=Path, help="Dataset root (session folders)")
    p_inspect.add_argument("--val-ratio", type=float, default=CONFIG.val_ratio)
    p_inspect.set_defaults(func=_cmd_inspect)

    p_train = sub.add_parser("train", help="train on operational Dataset/ only")
    p_train.add_argument("dataset", type=Path)
    _add_train_hparams(p_train)
    p_train.add_argument(
        "--cache-clips",
        action="store_true",
        default=CONFIG.loader.cache_clips,
        help="keep decoded WAV in RAM (~9 GiB train; duplicates per DataLoader worker)",
    )
    p_train.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="optional classifier checkpoint (e.g. after DADS pretrain)",
    )
    p_train.add_argument(
        "--head-only",
        action="store_true",
        help="skip encoder fine-tune (phase 2)",
    )
    p_train.set_defaults(func=_cmd_train, head_only=False)

    p_pretrain = sub.add_parser("pretrain", help="stage 1: fine-tune on Hugging Face DADS")
    _add_train_hparams(p_pretrain)
    p_pretrain.set_defaults(
        output=Path(DADS_PRETRAIN_CHECKPOINT),
        val_ratio=DADS_VAL_RATIO,
        func=_cmd_pretrain,
    )
    p_pretrain.add_argument("--dads-dataset", default=DADS_HF_DATASET)
    p_pretrain.add_argument("--hf-cache-dir", type=Path, default=Path(".cache/huggingface"))
    p_pretrain.add_argument("--index-cache-dir", type=Path, default=Path(DADS_INDEX_CACHE_DIR))
    p_pretrain.add_argument("--max-clips", type=int, default=None, help="limit clips for debugging")
    p_pretrain.add_argument(
        "--no-balance",
        action="store_true",
        help="disable balanced sampling on DADS train windows",
    )
    p_pretrain.add_argument(
        "--encoder-finetune",
        action="store_true",
        help="also run encoder fine-tune on DADS (default: head only)",
    )

    p_finetune = sub.add_parser("finetune", help="stage 2: fine-tune on operational Dataset/")
    p_finetune.add_argument("dataset", type=Path)
    _add_train_hparams(p_finetune)
    p_finetune.set_defaults(
        val_ratio=CONFIG.val_ratio,
        func=_cmd_finetune,
    )
    p_finetune.add_argument(
        "--init-checkpoint",
        type=Path,
        required=True,
        help="classifier checkpoint from pretrain stage",
    )
    p_finetune.add_argument(
        "--cache-clips",
        action="store_true",
        default=CONFIG.loader.cache_clips,
    )
    p_finetune.add_argument(
        "--head-only",
        action="store_true",
        help="skip encoder fine-tune (phase 2)",
    )

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
