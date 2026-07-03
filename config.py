"""Preprocessing parameters for listen-channel training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:
    """Log-mel preprocessing for ListenChannel recordings."""

    # Input WAV (listen channel from x502 ADC / beamformer).
    source_rate: int = 62500  # Hz, > 0
    input_sec: float = 1.0  # seconds, > 0
    window_stride_sec: float = 0.262144  # seconds, > 0; step between 1 s windows at source_rate

    # Model input (after resample).
    sample_rate: int = 16000  # Hz, > 0

    # Mel band — tune for your sensor / targets.
    fmin_hz: float = 200.0  # >= 0
    fmax_hz: float = 2500.0  # > fmin_hz, <= sample_rate / 2 (Nyquist)

    # Spectrogram shape [n_mels, time_bins] -> ONNX [1, 1, 224, 224].
    n_fft: int = 2048  # > 0, <= input_sec * sample_rate
    n_mels: int = 224  # > 0
    time_bins: int = 224  # > 1 (hop_length needs time_bins - 1)

    @property
    def window_samples_source(self) -> int:
        return int(round(self.source_rate * self.input_sec))

    @property
    def stride_samples_source(self) -> int:
        return max(1, int(round(self.source_rate * self.window_stride_sec)))

    @property
    def model_samples(self) -> int:
        return int(round(self.sample_rate * self.input_sec))

    @property
    def hop_length(self) -> int:
        n = self.model_samples
        if self.time_bins <= 1:
            return 0
        return int((n - self.n_fft) / (self.time_bins - 1))
