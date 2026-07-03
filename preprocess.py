"""PCM window -> log-mel preprocessing (used by ListenChannelDataset)."""

from __future__ import annotations

from typing import Optional

import librosa
import numpy as np

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
    input_sec: float,
    target_rate: int,
) -> np.ndarray:
    required = int(round(input_sec * target_rate))
    wave = np.zeros(required, dtype=np.float32)
    if samples.size == 0:
        return wave
    src = samples.astype(np.float32, copy=False)
    if source_rate != target_rate:
        src = simple_resample_mono(src, source_rate, target_rate)
    wave[:src.shape[0]] = src
    return wave


def peak_normalize(wave: np.ndarray) -> np.ndarray:
    if wave.size == 0:
        return np.zeros(0, dtype=np.float32)
    max_abs = float(np.max(np.abs(wave)))
    if max_abs == 0.0:
        return np.zeros_like(wave, dtype=np.float32)
    return (wave / max_abs).astype(np.float32)


class ListenChannelPreprocessor:
    """1 s PCM @ source_rate -> log-mel [n_mels, time_bins]."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        cfg = self.config
        self._win_length = cfg.n_fft
        self._hop_length = cfg.hop_length
        self._n_frames = cfg.time_bins
        self._n_freq = self._win_length // 2 + 1
        self._hann = np.hanning(self._win_length).astype(np.float32)
        self._mel_basis = librosa.filters.mel(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin_hz,
            fmax=cfg.fmax_hz,
            norm="slaney",
            htk=False,
        ).astype(np.float32)

    def compute_log_mel(self, waveform: np.ndarray) -> np.ndarray:
        wave = waveform.astype(np.float32, copy=False)
        n = wave.shape[0]
        power = np.zeros((self._n_frames, self._n_freq), dtype=np.float32)

        for t in range(self._n_frames):
            offset = t * self._hop_length
            frame = np.zeros(self._win_length, dtype=np.float32)
            for i in range(self._win_length):
                idx = offset + i
                if idx < n:
                    frame[i] = wave[idx]
            frame *= self._hann
            spectrum = np.fft.rfft(frame, n=self._win_length)
            power[t, :] = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(
                np.float32
            )

        mel = power @ self._mel_basis.T
        return np.log1p(np.maximum(mel, 0.0), dtype=np.float32).T

    def process_window(self, pcm: np.ndarray, source_rate: int) -> np.ndarray:
        cfg = self.config
        wave = peak_normalize(
            mono_to_model_input(pcm, source_rate, cfg.input_sec, cfg.sample_rate)
        )
        return self.compute_log_mel(wave)
