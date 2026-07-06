"""PCM window -> Kaldi fbank preprocessing (used by ListenChannelDataset)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torchaudio.compliance.kaldi as ta_kaldi

from config import PreprocessConfig


def simple_resample_mono(pcm: np.ndarray, input_rate: int, target_rate: int) -> np.ndarray:
    if pcm.size == 0 or input_rate == target_rate:
        return pcm.astype(np.float32, copy=False)

    input_frames = pcm.shape[0]
    output_frames = int(input_frames * target_rate // input_rate)
    if output_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    pcm_f32 = pcm.astype(np.float32, copy=False)
    src_idx = np.linspace(0, input_frames - 1, num=output_frames, dtype=np.float64)
    idx0 = np.floor(src_idx).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, input_frames - 1)
    frac = (src_idx - idx0).astype(np.float32)
    frac = np.clip(frac, 0.0, 1.0)
    return pcm_f32[idx0] * (1.0 - frac) + pcm_f32[idx1] * frac


def mono_to_model_input(
    samples: np.ndarray,
    source_rate: int,
    required_samples: int,
    target_rate: int,
) -> np.ndarray:
    required = required_samples
    wave = np.zeros(required, dtype=np.float32)
    if samples.size == 0:
        return wave
    src = samples.astype(np.float32, copy=False)
    if source_rate != target_rate:
        src = simple_resample_mono(src, source_rate, target_rate)
    wave[:src.shape[0]] = src
    return wave


class ListenChannelPreprocessor:
    """1 s PCM @ source_rate -> Kaldi fbank [num_mel_bins, num_frames] for BEATs."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def compute_fbank(self, waveform: np.ndarray) -> np.ndarray:
        cfg = self.config
        wave = torch.from_numpy(waveform.astype(np.float32, copy=False)).unsqueeze(0)
        wave = wave * (2**15)
        fbank = ta_kaldi.fbank(
            wave,
            num_mel_bins=cfg.num_mel_bins,
            sample_frequency=cfg.sample_rate,
            frame_length=cfg.frame_length_ms,
            frame_shift=cfg.frame_shift_ms,
        )
        fbank = (fbank - cfg.fbank_mean) / (2.0 * cfg.fbank_std)
        return fbank.numpy().T.astype(np.float32, copy=False)

    def process_window(self, pcm: np.ndarray, source_rate: int) -> np.ndarray:
        cfg = self.config
        wave = mono_to_model_input(pcm, source_rate, cfg.model_samples, cfg.sample_rate)
        return self.compute_fbank(wave)
