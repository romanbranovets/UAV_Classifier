"""Listen-channel classifier on a pretrained BEATs encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from beats import BEATs, BEATsConfig
from config import BeatsClassifierConfig

class MlpClassifierHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_beats_encoder(
    checkpoint_path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[BEATs, BEATsConfig]:
    """
    Load a BEATs encoder from an official ``.pt`` checkpoint.

    Predictor weights from AudioSet fine-tuned checkpoints are dropped; attach
    ``ListenChannelBeatsClassifier`` head for bkg/dvs/ed.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    cfg.finetuned_model = False

    encoder = BEATs(cfg)
    state = {
        key: value
        for key, value in checkpoint["model"].items()
        if not key.startswith("predictor")
    }
    encoder.load_state_dict(state, strict=False)
    return encoder, cfg


class ListenChannelBeatsClassifier(nn.Module):
    """
    Precomputed Kaldi fbank ``[B, 1, 128, T]`` → logits ``[B, num_classes]``.

    Encoder pooled embedding ``[B, 768]`` is returned for FAISS / metric learning.
    """

    def __init__(
        self,
        encoder: BEATs,
        config: Optional[BeatsClassifierConfig] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config or BeatsClassifierConfig()

        embed_dim = encoder.cfg.encoder_embed_dim
        self.classifier = MlpClassifierHead(
            embed_dim,
            self.config.head_hidden_dim,
            self.config.num_classes,
            self.config.head_dropout,
        )

        self._configure_encoder_training()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        config: Optional[BeatsClassifierConfig] = None,
        map_location: str | torch.device = "cpu",
    ) -> ListenChannelBeatsClassifier:
        encoder, beats_cfg = load_beats_encoder(checkpoint_path, map_location=map_location)
        patch_size = beats_cfg.input_patch_size
        if patch_size <= 0:
            raise ValueError("checkpoint cfg missing input_patch_size")

        model_config = config or BeatsClassifierConfig(patch_size=patch_size)
        return cls(encoder, model_config)

    def _configure_encoder_training(self) -> None:
        self.freeze_encoder()

    def begin_encoder_finetune(self, last_n_layers: int = 2) -> None:
        """Phase 2: unfreeze the last ``last_n_layers`` transformer blocks."""
        if last_n_layers <= 0:
            raise ValueError("last_n_layers must be positive")
        self.unfreeze_encoder(last_n_layers=last_n_layers)

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def unfreeze_encoder(self, last_n_layers: Optional[int] = None) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()
        if last_n_layers is None:
            return
        frozen = self.encoder.encoder.layers[: -last_n_layers]
        for layer in frozen:
            for param in layer.parameters():
                param.requires_grad = False

    def encode_fbank(self, fbank: torch.Tensor) -> torch.Tensor:
        """fbank ``[B, 1, F, T]`` → pooled embedding ``[B, 768]``."""
        patch_size = self.config.patch_size
        time_bins = fbank.shape[-1]
        if time_bins % patch_size != 0:
            raise ValueError(
                f"fbank time {time_bins} must be divisible by patch_size {patch_size}"
            )

        fbank = fbank.transpose(2, 3)  # [B, 1, time, freq]

        x = self.encoder.patch_embedding(fbank)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        x = self.encoder.layer_norm(x)

        if self.encoder.post_extract_proj is not None:
            x = self.encoder.post_extract_proj(x)

        x = self.encoder.dropout_input(x)
        x, _ = self.encoder.encoder(x, padding_mask=None)
        return x.mean(dim=1)

    def forward(self, fbank: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encode_fbank(fbank)
        return {
            "embedding": embedding,
            "logits": self.classifier(embedding),
        }

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.classifier.parameters())

    def parameter_groups(
        self,
        *,
        encoder_lr: float,
        head_lr: float,
    ) -> list[dict]:
        """AdamW param groups: lower LR for unfrozen encoder, higher for MLP head."""
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = self.head_parameters()
        groups: list[dict] = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": encoder_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        return groups

    def head_parameter_group(self, *, lr: float) -> dict:
        return {"params": self.head_parameters(), "lr": lr}
