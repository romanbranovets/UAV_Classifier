"""Training pipeline entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import ListenChannelDataset, prepare_train_split


def _cmd_inspect(args: argparse.Namespace) -> None:
    from collections import Counter

    dataset = ListenChannelDataset(args.dataset)
    print(f"sessions: {dataset.num_sessions}")
    print(f"clips:    {dataset.num_clips}")
    print(f"windows:  {len(dataset)}")

    clip_labels = Counter(int(clip.label) for clip in dataset.clips)
    windows_per_clip = Counter(window.clip_index for window in dataset.windows)
    window_labels: Counter[int] = Counter()
    for clip_idx, count in windows_per_clip.items():
        window_labels[int(dataset.clips[clip_idx].label)] += count
    print(f"clip labels:   {dict(sorted(clip_labels.items()))}")
    print(f"window labels: {dict(sorted(window_labels.items()))}")

    if args.val_ratio > 0:
        train_ds, val_ds, _sampler = prepare_train_split(dataset, val_ratio=args.val_ratio)
        print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}")
        print(f"augment: on for {len(train_ds.indices)} train windows")
        print(f"background pool: {len(dataset._background_indices)} windows")

    if len(dataset) > 0:
        sample = dataset[0]
        print(
            f"sample: log_mel={tuple(sample['log_mel'].shape)} "
            f"label={sample['label_name']} session={sample['session_id']} "
            f"channel={sample['channel']} tabular={sample['tabular'].tolist()}"
        )


def _cmd_train(_args: argparse.Namespace) -> None:
    raise NotImplementedError("training step not implemented yet")


def main() -> None:
    parser = argparse.ArgumentParser(description="x502 listen-channel training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="summarize ListenChannelDataset")
    p_inspect.add_argument("dataset", type=Path, help="Dataset root (session folders)")
    p_inspect.add_argument("--val-ratio", type=float, default=0.15)
    p_inspect.set_defaults(func=_cmd_inspect)

    p_train = sub.add_parser("train", help="train classifier (not implemented)")
    p_train.set_defaults(func=_cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
