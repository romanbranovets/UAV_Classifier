"""Hugging Face DADS dataset (geronimobasso/drone-audio-detection-samples)."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Subset

from augment import apply_spec_augment, mix_at_snr
from config import AugmentConfig, DADS_HF_DATASET, DADS_SOURCE_RATE, PreprocessConfig
from dataset import Label
from preprocess import ListenChannelPreprocessor

DADS_LABEL_BKG = int(Label.BKG)
DADS_LABEL_UAV = int(Label.UAV)


@dataclass(frozen=True)
class DadsWindowRef:
    clip_index: int
    start_sample: int


def dads_preprocess_config(base: Optional[PreprocessConfig] = None) -> PreprocessConfig:
    """DADS is already 16 kHz mono — match BEATs windowing at native rate."""
    cfg = base or PreprocessConfig()
    return replace(cfg, source_rate=DADS_SOURCE_RATE)


def load_dads_hf(
    *,
    hf_id: str = DADS_HF_DATASET,
    cache_dir: Optional[Path | str] = None,
    max_clips: Optional[int] = None,
):
    from datasets import Audio, load_dataset

    split = f"train[:{max_clips}]" if max_clips is not None else "train"
    kwargs: dict = {"path": hf_id, "split": split}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    ds = load_dataset(**kwargs)
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def iter_dads_window_starts(
    num_samples: int,
    window_samples: int,
    stride_samples: int,
) -> Iterator[int]:
    """
    Window starts for DADS clips.

    Clips shorter than ``window_samples`` still yield one window at 0; trailing
    samples are zero-padded in ``mono_to_model_input`` (many drone clips are ~0.5–0.6 s).
    """
    if num_samples <= 0:
        return iter(())
    if num_samples < window_samples:
        return iter((0,))
    return iter(range(0, num_samples - window_samples + 1, stride_samples))


def _index_cache_path(cache_dir: Path, *, hf_id: str, max_clips: Optional[int], config: PreprocessConfig) -> Path:
    tag = hf_id.split("/")[-1]
    limit = max_clips if max_clips is not None else "all"
    key = (
        f"{tag}_clips{limit}_"
        f"sr{config.source_rate}_win{config.window_samples_source}_"
        f"stride{config.stride_samples_source}_padshort"
    )
    return cache_dir / f"{key}.npz"


def _audio_num_samples(audio_bytes: bytes) -> int:
    return int(sf.info(io.BytesIO(audio_bytes)).frames)


class DadsDataset(Dataset):
    """
    Windowed clips from DADS (label 0 = no drone, 1 = drone).

    Each Hugging Face row is one source clip. Windowing matches the listen-channel
    pipeline at 16 kHz; clips shorter than one model window are kept as a single
    zero-padded window (typical for short drone samples in DADS).
    """

    def __init__(
        self,
        hf_dataset,
        config: Optional[PreprocessConfig] = None,
        *,
        augment_config: Optional[AugmentConfig] = None,
        seed: int = 0,
        index_cache_dir: Optional[Path | str] = None,
        hf_id: str = DADS_HF_DATASET,
        max_clips: Optional[int] = None,
    ) -> None:
        self.hf = hf_dataset
        self.config = dads_preprocess_config(config)
        self.preprocessor = ListenChannelPreprocessor(self.config)
        self.augment_config = augment_config or AugmentConfig()
        self._rng = np.random.default_rng(seed)
        self.hf_id = hf_id
        self.max_clips = max_clips

        self.windows: list[DadsWindowRef] = []
        self._background_indices: list[int] = []
        self._train_background_indices: list[int] = []
        self._augment_indices: set[int] = set()
        self._audio_cache: dict[int, np.ndarray] = {}

        cache_dir = Path(index_cache_dir) if index_cache_dir is not None else None
        self._clip_labels = self._build_index(cache_dir)

    def _build_index(self, cache_dir: Optional[Path]) -> np.ndarray:
        if cache_dir is not None:
            cache_path = _index_cache_path(
                cache_dir,
                hf_id=self.hf_id,
                max_clips=self.max_clips,
                config=self.config,
            )
            if cache_path.is_file():
                data = np.load(cache_path, allow_pickle=False)
                self.windows = [
                    DadsWindowRef(clip_index=int(c), start_sample=int(s))
                    for c, s in zip(data["clip_index"], data["start_sample"])
                ]
                clip_labels = data["clip_labels"]
                for window_index, ref in enumerate(self.windows):
                    if int(clip_labels[ref.clip_index]) == DADS_LABEL_BKG:
                        self._background_indices.append(window_index)
                print(f"DADS index loaded from cache: {len(self.windows)} windows ({cache_path})")
                self._window_clip_index = data["clip_index"].astype(np.int32, copy=False)
                return clip_labels

        win = self.config.window_samples_source
        stride = self.config.stride_samples_source
        clip_labels = np.empty(len(self.hf), dtype=np.int8)

        from tqdm import tqdm

        clip_index_arr: list[int] = []
        start_sample_arr: list[int] = []

        for clip_index in tqdm(range(len(self.hf)), desc="DADS index", unit="clip"):
            row = self.hf[clip_index]
            label = int(row["label"])
            if label not in (DADS_LABEL_BKG, DADS_LABEL_UAV):
                raise ValueError(f"unexpected DADS label {label} at clip {clip_index}")
            clip_labels[clip_index] = label

            audio_bytes = row["audio"]["bytes"]
            num_samples = _audio_num_samples(audio_bytes)
            for start in iter_dads_window_starts(num_samples, win, stride):
                window_index = len(self.windows)
                self.windows.append(DadsWindowRef(clip_index=clip_index, start_sample=start))
                clip_index_arr.append(clip_index)
                start_sample_arr.append(start)
                if label == DADS_LABEL_BKG:
                    self._background_indices.append(window_index)

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = _index_cache_path(
                cache_dir,
                hf_id=self.hf_id,
                max_clips=self.max_clips,
                config=self.config,
            )
            np.savez(
                cache_path,
                clip_index=np.asarray(clip_index_arr, dtype=np.int32),
                start_sample=np.asarray(start_sample_arr, dtype=np.int32),
                clip_labels=clip_labels,
            )
            meta = {
                "hf_id": self.hf_id,
                "num_clips": len(self.hf),
                "num_windows": len(self.windows),
            }
            cache_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"DADS index cached -> {cache_path}")

        self._window_clip_index = np.fromiter(
            (w.clip_index for w in self.windows), dtype=np.int32, count=len(self.windows)
        )
        return clip_labels

    @property
    def num_clips(self) -> int:
        return len(self.hf)

    @property
    def clip_labels(self) -> np.ndarray:
        return self._clip_labels

    def set_augment_indices(self, indices: Sequence[int]) -> None:
        train = set(indices)
        self._augment_indices = train
        self._train_background_indices = [i for i in self._background_indices if i in train]

    def _load_clip_audio(self, clip_index: int) -> np.ndarray:
        if clip_index in self._audio_cache:
            return self._audio_cache[clip_index]

        row = self.hf[clip_index]
        pcm, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=False)
        if pcm.ndim > 1:
            pcm = pcm[:, 0]
        if int(sample_rate) != self.config.source_rate:
            raise ValueError(
                f"DADS clip {clip_index}: expected {self.config.source_rate} Hz, got {sample_rate}"
            )
        pcm = pcm.astype(np.float32, copy=False)
        self._audio_cache[clip_index] = pcm
        return pcm

    def _load_window_chunk(self, index: int) -> tuple[np.ndarray, int]:
        window = self.windows[index]
        audio = self._load_clip_audio(window.clip_index)
        win = self.config.window_samples_source
        chunk = audio[window.start_sample : window.start_sample + win]
        label = int(self._clip_labels[window.clip_index])
        return chunk, label

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        chunk, label = self._load_window_chunk(index)

        if index in self._augment_indices and self._train_background_indices:
            cfg = self.augment_config
            if label == DADS_LABEL_UAV and self._rng.random() < cfg.noise_mix_p:
                bg_index = int(self._rng.choice(self._train_background_indices))
                bg_chunk, _ = self._load_window_chunk(bg_index)
                snr_db = self._rng.uniform(cfg.snr_min_db, cfg.snr_max_db)
                chunk = mix_at_snr(chunk, bg_chunk, snr_db)

        fbank = self.preprocessor.process_window(chunk, self.config.source_rate)

        if index in self._augment_indices:
            fbank = apply_spec_augment(fbank, label, self.augment_config, self._rng)

        return {
            "fbank": torch.from_numpy(fbank).unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def split_dads_by_clip(
    dataset: DadsDataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Train/val split by source clip (no window leakage), stratified by label."""
    from sklearn.model_selection import train_test_split

    clip_indices = np.arange(dataset.num_clips)
    train_clips, val_clips = train_test_split(
        clip_indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=dataset.clip_labels,
    )
    train_clip_set = set(int(i) for i in train_clips)
    val_clip_set = set(int(i) for i in val_clips)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for index, window in enumerate(dataset.windows):
        if window.clip_index in val_clip_set:
            val_indices.append(index)
        elif window.clip_index in train_clip_set:
            train_indices.append(index)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def prepare_dads_train_split(
    dataset: DadsDataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    train_ds, val_ds = split_dads_by_clip(dataset, val_ratio=val_ratio, seed=seed)
    dataset.set_augment_indices(train_ds.indices)
    return train_ds, val_ds
