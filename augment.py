"""Tier-1 training augmentations (SpecAugment, background noise mix)."""

from __future__ import annotations

import numpy as np

from config import AugmentConfig

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
