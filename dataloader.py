"""DataLoader helpers for listen-channel training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from config import AugmentConfig, DataLoaderConfig, PreprocessConfig
from dataset import ListenChannelDataset, prepare_train_split


def make_dataloaders(
    root: Path | str,
    *,
    preprocess_config: Optional[PreprocessConfig] = None,
    augment_config: Optional[AugmentConfig] = None,
    loader_config: Optional[DataLoaderConfig] = None,
    val_ratio: float = 0.15,
    seed: int = 0,
    cache_clips: bool = True,
) -> tuple[DataLoader, DataLoader, ListenChannelDataset, Subset, Subset]:
    """
    Build train/val DataLoaders with session split, augmentations, and ЭД oversampling.

    Train loader uses ``WeightedRandomSampler`` (no shuffle). Val loader is sequential.
    """
    cfg = loader_config or DataLoaderConfig()
    dataset = ListenChannelDataset(
        root,
        config=preprocess_config,
        augment_config=augment_config,
        seed=seed,
        cache_clips=cache_clips,
    )
    train_ds, val_ds, sampler = prepare_train_split(dataset, val_ratio=val_ratio, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batch_size or cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
    )
    return train_loader, val_loader, dataset, train_ds, val_ds
