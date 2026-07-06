"""Training, model, and preprocessing configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


# Microsoft OneDrive: https://github.com/microsoft/unilm/tree/master/beats
BEATS_CHECKPOINT = "checkpoints/BEATs_iter3_plus_AS2M.pt"

NUM_CLASSES = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MLflow 3.x: use sqlite backend (filesystem ./mlruns is deprecated).
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_MLFLOW_EXPERIMENT = "uav-listen"
DEFAULT_PLOT_PATH = "training_curves.png"

@dataclass(frozen=True)
class PreprocessConfig:
    """Kaldi fbank preprocessing for BEATs (listen-channel recordings)."""

    # Input WAV (listen channel from x502 ADC / beamformer).
    source_rate: int = 62500  # Hz, > 0
    window_stride_sec: float = 0.262144  # seconds, > 0; step between windows at source_rate

    # Model input (after resample).
    sample_rate: int = 16000  # Hz, > 0

    # Kaldi fbank — matches microsoft/unilm beats/BEATs.py preprocess().
    num_mel_bins: int = 128  # > 0
    frame_length_ms: float = 25.0  # Povey window length
    frame_shift_ms: float = 10.0  # hop between frames
    fbank_mean: float = 15.41663
    fbank_std: float = 6.55582

    # Fbank time bins — 112 = 7 × 16 (BEATs patch size), no pad in the model.
    beats_fbank_frames: int = 112

    @property
    def frame_length_samples(self) -> int:
        return int(round(self.sample_rate * self.frame_length_ms / 1000.0))

    @property
    def frame_shift_samples(self) -> int:
        return max(1, int(round(self.sample_rate * self.frame_shift_ms / 1000.0)))

    @property
    def model_samples(self) -> int:
        fl = self.frame_length_samples
        fs = self.frame_shift_samples
        return fl + (self.beats_fbank_frames - 1) * fs

    @property
    def input_sec(self) -> float:
        return self.model_samples / self.sample_rate

    @property
    def window_samples_source(self) -> int:
        return self.model_samples * self.source_rate // self.sample_rate

    @property
    def stride_samples_source(self) -> int:
        return max(1, int(round(self.source_rate * self.window_stride_sec)))


@dataclass(frozen=True)
class AugmentConfig:
    """Tier-1 training augmentations (SpecAugment, background noise mix)."""

    spec_augment_p: float = 0.3
    freq_mask_count: int = 2
    freq_mask_width: int = 24
    time_mask_count: int = 2
    time_mask_width: int = 32

    ed_freq_mask_count: int = 3
    ed_freq_mask_width: int = 30
    ed_time_mask_count: int = 3
    ed_time_mask_width: int = 40

    noise_mix_p: float = 0.3
    snr_min_db: float = 0.0
    snr_max_db: float = 20.0


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 256
    num_workers: int = os.cpu_count()
    pin_memory: bool = True
    drop_last: bool = False
    val_batch_size: Optional[int] = None


@dataclass(frozen=True)
class BeatsClassifierConfig:
    """MLP head on a frozen BEATs encoder (unfreeze top layers in training phase 2)."""

    num_classes: int = NUM_CLASSES
    head_hidden_dim: int = 256
    head_dropout: float = 0.3
    patch_size: int = 16


@dataclass(frozen=True)
class TrainConfig:
    head_max_epochs: int = 50
    encoder_max_epochs: int = 50
    patience: int = 10
    min_delta: float = 1e-4
    head_lr: float = 1e-3
    encoder_lr: float = 1e-5
    unfreeze_last_n_layers: int = 2
    weight_decay: float = 1e-3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 3
    lr_scheduler_min_lr: float = 1e-7
    val_ratio: float = 0.15
    seed: int = 0
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI
    mlflow_experiment: str = DEFAULT_MLFLOW_EXPERIMENT
    mlflow_run_name: Optional[str] = None
    plot_path: str = DEFAULT_PLOT_PATH
