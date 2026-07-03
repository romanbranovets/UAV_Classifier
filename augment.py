"""Tier-1 training augmentations (SpecAugment, background noise mix)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    # SpecAugment on log-mel [n_mels, time_bins].
    spec_augment_p: float = 0.5
    freq_mask_count: int = 2
    freq_mask_width: int = 24
    time_mask_count: int = 2
    time_mask_width: int = 32

    # Stronger masks for ЭД (label 2).
    ed_freq_mask_count: int = 3
    ed_freq_mask_width: int = 30
    ed_time_mask_count: int = 3
    ed_time_mask_width: int = 40

    # Mix random Фон window into ДВС/ЭД at random SNR.
    noise_mix_p: float = 0.3
    snr_min_db: float = 0.0
    snr_max_db: float = 20.0

    # WeightedRandomSampler boost for ЭД.
    ed_sample_weight: float = 4.0


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sig = signal.astype(np.float32, copy=False)
    n = noise.astype(np.float32, copy=False)
    if n.shape[0] < sig.shape[0]:
        n = np.pad(n, (0, sig.shape[0] - n.shape[0]))
    elif n.shape[0] > sig.shape[0]:
        n = n[: sig.shape[0]]

    sig_power = float(np.mean(sig * sig)) + 1e-12
    noise_power = float(np.mean(n * n)) + 1e-12
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_scale = np.sqrt(target_noise_power / noise_power)
    return sig + n * np.float32(noise_scale)


def spec_augment(
    log_mel: np.ndarray,
    *,
    freq_mask_count: int,
    freq_mask_width: int,
    time_mask_count: int,
    time_mask_width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = log_mel.copy()
    n_mels, n_time = out.shape

    for _ in range(freq_mask_count):
        width = min(freq_mask_width, n_mels)
        start = int(rng.integers(0, max(1, n_mels - width + 1)))
        out[start : start + width, :] = 0.0

    for _ in range(time_mask_count):
        width = min(time_mask_width, n_time)
        start = int(rng.integers(0, max(1, n_time - width + 1)))
        out[:, start : start + width] = 0.0

    return out


def apply_spec_augment(
    log_mel: np.ndarray,
    label: int,
    cfg: AugmentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if rng.random() >= cfg.spec_augment_p:
        return log_mel

    if label == 2:
        return spec_augment(
            log_mel,
            freq_mask_count=cfg.ed_freq_mask_count,
            freq_mask_width=cfg.ed_freq_mask_width,
            time_mask_count=cfg.ed_time_mask_count,
            time_mask_width=cfg.ed_time_mask_width,
            rng=rng,
        )

    return spec_augment(
        log_mel,
        freq_mask_count=cfg.freq_mask_count,
        freq_mask_width=cfg.freq_mask_width,
        time_mask_count=cfg.time_mask_count,
        time_mask_width=cfg.time_mask_width,
        rng=rng,
    )
