"""Listen-channel classifier on a pretrained BEATs encoder (SupCon head)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from beats import BEATs, BEATsConfig
from config import BeatsClassifierConfig


class ProjectionHead(nn.Module):
    """BEATs embedding → L2-normalized projection for contrastive learning."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class ClassPrototypes(nn.Module):
    """
    Per-class mean embeddings (EMA) for cosine classification at eval time.

    Updated on train batches only; saved in ``state_dict`` as buffers.
    """

    def __init__(self, num_classes: int, dim: int, *, momentum: float = 0.9) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.register_buffer("vectors", torch.zeros(num_classes, dim))
        self.register_buffer("ready", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, projections: torch.Tensor, labels: torch.Tensor) -> None:
        """EMA update from a batch of L2-normalized projections."""
        for class_id in labels.unique():
            idx = int(class_id.item())
            mask = labels == idx
            if not mask.any():
                continue
            batch_mean = F.normalize(projections[mask].mean(dim=0), dim=0)
            if not bool(self.ready[idx]):
                self.vectors[idx] = batch_mean
                self.ready[idx] = True
                continue
            m = self.momentum
            self.vectors[idx] = F.normalize(m * self.vectors[idx] + (1.0 - m) * batch_mean, dim=0)

    def logits(self, projections: torch.Tensor) -> torch.Tensor:
        """Cosine similarity to class prototypes → ``[B, num_classes]``."""
        if not bool(self.ready.all()):
            ready = self.ready.to(projections.dtype)
            proto = F.normalize(self.vectors, dim=1) * ready.unsqueeze(1)
            logits = projections @ proto.T
            return logits.masked_fill(~self.ready.unsqueeze(0), float("-inf"))
        proto = F.normalize(self.vectors, dim=1)
        return projections @ proto.T


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss (Khosla et al., 2020).

    ``features`` must be L2-normalized. Positives share the same label in the batch.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1)
        batch_size = features.shape[0]
        if batch_size < 2:
            return features.new_zeros(())

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        logits = features @ features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        diag = torch.eye(batch_size, device=features.device, dtype=torch.bool)
        self_mask = (~diag).float()
        mask = mask * self_mask

        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_per_anchor = mask.sum(dim=1)
        has_positive = pos_per_anchor > 0
        if not bool(has_positive.any()):
            return features.new_zeros(())

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_per_anchor.clamp(min=1.0)
        return -mean_log_prob_pos[has_positive].mean()


def load_beats_encoder(
    checkpoint_path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[BEATs, BEATsConfig]:
    """
    Load a BEATs encoder from an official ``.pt`` checkpoint.

    Predictor weights from AudioSet fine-tuned checkpoints are dropped; attach
    ``ListenChannelBeatsClassifier`` SupCon head for bkg/uav.
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
    Precomputed Kaldi fbank ``[B, 1, 128, T]`` → SupCon projection and class logits.

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
        self.projector = ProjectionHead(
            embed_dim,
            self.config.proj_hidden_dim,
            self.config.proj_dim,
        )
        self.prototypes = ClassPrototypes(
            self.config.num_classes,
            self.config.proj_dim,
            momentum=self.config.prototype_momentum,
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

        if config is None:
            model_config = BeatsClassifierConfig(patch_size=patch_size)
        else:
            model_config = replace(config, patch_size=patch_size)
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
        projection = self.projector(embedding)
        return {
            "embedding": embedding,
            "projection": projection,
            "logits": self.prototypes.logits(projection),
        }

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.projector.parameters())

    def parameter_groups(
        self,
        *,
        encoder_lr: float,
        head_lr: float,
    ) -> list[dict]:
        """AdamW param groups: lower LR for unfrozen encoder, higher for projection head."""
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
