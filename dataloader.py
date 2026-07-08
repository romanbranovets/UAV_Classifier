"""DataLoader helpers for listen-channel and DADS training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from config import (
    AugmentConfig,
    DADS_HF_DATASET,
    DataLoaderConfig,
    PreprocessConfig,
)
from dads_dataset import DadsDataset, load_dads_hf, prepare_dads_train_split
from dataset import ListenChannelDataset, prepare_train_split


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    sampler: Optional[WeightedRandomSampler],
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def _balanced_sampler(subset: Subset, clip_labels: np.ndarray, window_clip_index: np.ndarray) -> WeightedRandomSampler:
    labels = [int(clip_labels[window_clip_index[i]]) for i in subset.indices]
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights))


def make_dataloaders(
    root: Path | str,
    *,
    preprocess_config: Optional[PreprocessConfig] = None,
    augment_config: Optional[AugmentConfig] = None,
    loader_config: Optional[DataLoaderConfig] = None,
    val_ratio: float = 0.15,
    seed: int = 0,
    cache_clips: bool = False,
) -> tuple[DataLoader, DataLoader, ListenChannelDataset, Subset, Subset]:
    """
    Build train/val DataLoaders with session split and train augmentations.

    Train loader is shuffled; val loader is sequential.
    """
    cfg = loader_config or DataLoaderConfig()

    dataset = ListenChannelDataset(
        root,
        config=preprocess_config,
        augment_config=augment_config,
        seed=seed,
        cache_clips=cache_clips,
    )
    train_ds, val_ds = prepare_train_split(dataset, val_ratio=val_ratio, seed=seed)

    train_loader = _make_loader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
    )
    val_loader = _make_loader(
        val_ds,
        batch_size=cfg.val_batch_size or cfg.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, dataset, train_ds, val_ds


def make_dads_dataloaders(
    *,
    hf_id: str = DADS_HF_DATASET,
    cache_dir: Optional[Path | str] = None,
    index_cache_dir: Optional[Path | str] = None,
    preprocess_config: Optional[PreprocessConfig] = None,
    augment_config: Optional[AugmentConfig] = None,
    loader_config: Optional[DataLoaderConfig] = None,
    val_ratio: float = 0.1,
    seed: int = 0,
    max_clips: Optional[int] = None,
    balance_train: bool = True,
) -> tuple[DataLoader, DataLoader, DadsDataset, Subset, Subset]:
    """Build train/val DataLoaders for Hugging Face DADS."""
    cfg = loader_config or DataLoaderConfig()
    hf_cache = Path(cache_dir) if cache_dir is not None else None
    hf_ds = load_dads_hf(hf_id=hf_id, cache_dir=hf_cache, max_clips=max_clips)

    dataset = DadsDataset(
        hf_ds,
        config=preprocess_config,
        augment_config=augment_config,
        seed=seed,
        index_cache_dir=index_cache_dir,
        hf_id=hf_id,
        max_clips=max_clips,
    )
    train_ds, val_ds = prepare_dads_train_split(dataset, val_ratio=val_ratio, seed=seed)

    sampler = None
    if balance_train:
        sampler = _balanced_sampler(train_ds, dataset.clip_labels, dataset._window_clip_index)

    train_loader = _make_loader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
    )
    val_loader = _make_loader(
        val_ds,
        batch_size=cfg.val_batch_size or cfg.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, dataset, train_ds, val_ds
