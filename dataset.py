"""PyTorch dataset for x502 listen-channel sessions (ground_truth.json + Channel_*.wav)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Subset

from config import PreprocessConfig
from preprocess import ListenChannelPreprocessor

SESSION_DIR_RE = re.compile(
    r"^(?P<date>\d{2}_\d{2}_\d{4})_(?P<time>\d{2}-\d{2}-\d{2}-\d+)_(?P<num>\d+)$"
)


class Label(IntEnum):
    BKG = 0
    DVS = 1
    ED = 2


CLASS_NAMES = ("bkg", "dvs", "ed")
LABEL_BY_NAME = {
    "Фон": Label.BKG,
    "ДВС": Label.DVS,
    "ЭД": Label.ED,
}

# Per-channel tabular features from ground_truth.json (same for all windows of a clip).
TABULAR_FEATURE_NAMES = (
    "mean_vip_vp",
    "mean_vip_gp",
    "mean_azimuth",
    "mean_elevation",
)


@dataclass(frozen=True)
class SessionInfo:
    """Parsed session folder name: ``DD_MM_YYYY_HH-MM-SS-mmm_N``."""

    path: Path
    session_id: str
    date: str
    time: str
    num: int

    @property
    def split_key(self) -> str:
        """``{date}_{num}`` — session number restarts at 1 for each date."""
        return f"{self.date}_{self.num}"


@dataclass(frozen=True)
class ChannelTabular:
    """Trace / manual metadata from ground_truth.json (one row per channel)."""

    mean_vip_vp: float
    mean_vip_gp: float
    mean_azimuth: float
    mean_elevation: float

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.mean_vip_vp, self.mean_vip_gp, self.mean_azimuth, self.mean_elevation],
            dtype=torch.float32,
        )


@dataclass(frozen=True)
class ChannelClip:
    session: SessionInfo
    channel_key: str
    label: Label
    segment_paths: tuple[Path, ...]
    num_samples: int
    tabular: ChannelTabular


@dataclass(frozen=True)
class ClipWindowRef:
    clip_index: int
    start_sample: int


def parse_session_dir(path: Path) -> Optional[SessionInfo]:
    path = Path(path)
    match = SESSION_DIR_RE.match(path.name)
    if not match:
        return None
    return SessionInfo(
        path=path,
        session_id=path.name,
        date=match.group("date"),
        time=match.group("time"),
        num=int(match.group("num")),
    )


def discover_sessions(root: Path, only: Optional[Sequence[Path]] = None) -> list[SessionInfo]:
    root = Path(root)
    if only is not None:
        sessions = []
        for path in only:
            path = Path(path)
            info = parse_session_dir(path)
            if info is None:
                raise ValueError(f"not a session folder: {path}")
            if not (path / "ground_truth.json").is_file():
                raise ValueError(f"missing ground_truth.json: {path}")
            sessions.append(info)
        return sorted(sessions, key=lambda s: (s.date, s.time, s.num))

    sessions: list[SessionInfo] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        info = parse_session_dir(child)
        if info is None:
            continue
        if not (child / "ground_truth.json").is_file():
            continue
        sessions.append(info)
    return sessions


def _channel_key_from_entry(entry: dict) -> Optional[str]:
    channel = entry.get("Channel")
    if channel == "Manual":
        manual_id = entry.get("№")
        return f"manual:{manual_id}" if manual_id is not None else None
    if isinstance(channel, int):
        return str(channel)
    return None


def _optional_float(entry: dict, *keys: str) -> float:
    for key in keys:
        value = entry.get(key)
        if value is None:
            continue
        return float(value)
    return float("nan")


def parse_channel_tabular(entry: dict) -> ChannelTabular:
    """
    Trace rows: ``Средний ВИП ВП/ГП``, ``Средний азимут``, ``Средний угол места``.
    Manual rows: ``Азимут``, ``Угол места`` (VIP fields absent → NaN).
    """
    return ChannelTabular(
        mean_vip_vp=_optional_float(entry, "Средний ВИП ВП"),
        mean_vip_gp=_optional_float(entry, "Средний ВИП ГП"),
        mean_azimuth=_optional_float(entry, "Средний азимут", "Азимут"),
        mean_elevation=_optional_float(entry, "Средний угол места", "Угол места"),
    )


def _channel_path_prefix(channel_key: str) -> str:
    if channel_key.startswith("manual:"):
        return f"Channel_Manual_{channel_key.split(':', 1)[1]}"
    return f"Channel_{channel_key}"


def channel_segment_paths(session_dir: Path, channel_key: str) -> list[Path]:
    """
    Ordered channel timeline: backups (earliest timestamp first), then current buffer.

    ``Channel_N_hh-mm-ss-zzz.wav`` … ``Channel_N.wav``
    """
    session_dir = Path(session_dir)
    prefix = _channel_path_prefix(channel_key)
    backup_re = re.compile(
        rf"^{re.escape(prefix)}_(?P<ts>\d{{2}}-\d{{2}}-\d{{2}}-\d+)\.wav$"
    )

    backups: list[tuple[str, Path]] = []
    for path in session_dir.glob(f"{prefix}_*.wav"):
        match = backup_re.match(path.name)
        if match:
            backups.append((match.group("ts"), path))
    backups.sort(key=lambda item: item[0])

    paths = [path for _, path in backups]
    current = session_dir / f"{prefix}.wav"
    if current.exists():
        paths.append(current)
    return paths


def load_concat_mono(paths: Sequence[Path]) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    sample_rate: Optional[int] = None
    for path in paths:
        pcm, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if pcm.ndim > 1:
            pcm = pcm[:, 0]
        pcm = pcm.astype(np.float32, copy=False)
        if sample_rate is None:
            sample_rate = int(sr)
        elif int(sr) != sample_rate:
            raise ValueError(f"sample rate mismatch in {path}: {sr} != {sample_rate}")
        chunks.append(pcm)

    return np.concatenate(chunks), int(sample_rate)


def iter_window_starts(num_samples: int, window_samples: int, stride_samples: int) -> Iterator[int]:
    if num_samples < window_samples:
        return iter(())
    return iter(range(0, num_samples - window_samples + 1, stride_samples))


def iter_labeled_clips(session: SessionInfo) -> Iterator[ChannelClip]:
    entries = json.loads((session.path / "ground_truth.json").read_text(encoding="utf-8"))

    for entry in entries:
        label_name = entry.get("Тип цели", "")
        if label_name not in LABEL_BY_NAME:
            continue

        channel_key = _channel_key_from_entry(entry)
        if channel_key is None:
            continue

        segments = channel_segment_paths(session.path, channel_key)
        if not segments:
            continue

        num_samples = sum(int(sf.info(str(path)).frames) for path in segments)

        yield ChannelClip(
            session=session,
            channel_key=channel_key,
            label=LABEL_BY_NAME[label_name],
            segment_paths=tuple(segments),
            num_samples=num_samples,
            tabular=parse_channel_tabular(entry),
        )


class ListenChannelDataset(Dataset):
    """
    One labeled channel per clip (``Тип цели``), concatenated segments, 1 s windows.

    Session folders: ``DD_MM_YYYY_HH-MM-SS-mmm_N`` (``N`` restarts at 1 for each date).
    """

    def __init__(
        self,
        root: Path | str,
        config: Optional[PreprocessConfig] = None,
        session_dirs: Optional[Sequence[Path | str]] = None,
        *,
        cache_clips: bool = True,
    ) -> None:
        self.root = Path(root)
        self.config = config or PreprocessConfig()
        self.preprocessor = ListenChannelPreprocessor(self.config)
        self.cache_clips = cache_clips

        self.clips: list[ChannelClip] = []
        self.windows: list[ClipWindowRef] = []
        self._audio_cache: dict[tuple[str, str], tuple[np.ndarray, int]] = {}

        only = [Path(p) for p in session_dirs] if session_dirs is not None else None
        self._build_index(discover_sessions(self.root, only))

    def _build_index(self, sessions: Sequence[SessionInfo]) -> None:
        win = self.config.window_samples_source
        stride = self.config.stride_samples_source

        for session in sessions:
            for clip in iter_labeled_clips(session):
                clip_index = len(self.clips)
                self.clips.append(clip)
                for start in iter_window_starts(clip.num_samples, win, stride):
                    self.windows.append(ClipWindowRef(clip_index=clip_index, start_sample=start))

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def num_clips(self) -> int:
        return len(self.clips)

    @property
    def num_sessions(self) -> int:
        return len({clip.session.session_id for clip in self.clips})

    def clip_at(self, clip_index: int) -> ChannelClip:
        return self.clips[clip_index]

    def _clip_cache_key(self, clip: ChannelClip) -> tuple[str, str]:
        return clip.session.session_id, clip.channel_key

    def _load_clip_audio(self, clip: ChannelClip) -> tuple[np.ndarray, int]:
        key = self._clip_cache_key(clip)
        if self.cache_clips and key in self._audio_cache:
            return self._audio_cache[key]

        audio, sample_rate = load_concat_mono(clip.segment_paths)
        if self.cache_clips:
            self._audio_cache[key] = (audio, sample_rate)
        return audio, sample_rate

    def __getitem__(self, index: int) -> dict[str, object]:
        window = self.windows[index]
        clip = self.clips[window.clip_index]
        audio, sample_rate = self._load_clip_audio(clip)

        win = self.config.window_samples_source
        start = window.start_sample
        chunk = audio[start : start + win]

        log_mel = self.preprocessor.process_window(chunk, sample_rate)

        return {
            "log_mel": torch.from_numpy(log_mel),
            "label": torch.tensor(int(clip.label), dtype=torch.long),
            "tabular": clip.tabular.as_tensor(),
            "session_id": clip.session.session_id,
            "session_date": clip.session.date,
            "session_num": clip.session.num,
            "split_key": clip.session.split_key,
            "channel": clip.channel_key,
            "start_sample": start,
            "label_name": CLASS_NAMES[int(clip.label)],
        }


def split_dataset_by_session(
    dataset: ListenChannelDataset,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Train/val split by session folder — no leakage (``date_num``, N per date)."""
    keys = list(dict.fromkeys(clip.session.split_key for clip in dataset.clips))
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n_val = max(1, int(round(len(keys) * val_ratio)))
    val_keys = set(keys[:n_val])

    train_indices: list[int] = []
    val_indices: list[int] = []
    for index, window in enumerate(dataset.windows):
        if dataset.clips[window.clip_index].session.split_key in val_keys:
            val_indices.append(index)
        else:
            train_indices.append(index)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
