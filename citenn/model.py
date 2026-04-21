"""
Bi-encoder citation retriever.

Two towers:
    f_ctx(text)    : encodes a passage in the user's draft
    f_paper(text)  : encodes a candidate paper (title + abstract)

Both share a single backbone by default (weight-tied) which is standard
for citation-retrieval bi-encoders and halves memory. They are distinguished
only by a prefix token ("ctx: " vs "doc: ") so the model can learn to
project the two domains into the same space.

Loss: InfoNCE over the in-batch negatives (symmetric).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class BiEncoderConfig:
    backbone: str = "sentence-transformers/all-MiniLM-L6-v2"
    proj_dim: int = 256
    max_len: int = 256
    temperature: float = 0.05


class BiEncoder(nn.Module):
    def __init__(self, cfg: BiEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
        self.backbone = AutoModel.from_pretrained(cfg.backbone)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, cfg.proj_dim, bias=False)

    # ------ encoding ------

    def _mean_pool(self, last_hidden, attn_mask):
        mask = attn_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def _encode(self, texts: list[str], device) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_len,
            return_tensors="pt",
        ).to(device)
        out = self.backbone(**enc)
        pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
        z = self.proj(pooled)
        return F.normalize(z, p=2, dim=-1)

    def encode_contexts(self, texts: list[str], device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return self._encode([f"ctx: {t}" for t in texts], device)

    def encode_papers(self, texts: list[str], device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return self._encode([f"doc: {t}" for t in texts], device)

    # ------ loss ------

    def info_nce(self, z_ctx: torch.Tensor, z_doc: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Symmetric InfoNCE with in-batch negatives. Diagonal = positives."""
        logits = (z_ctx @ z_doc.t()) / self.cfg.temperature
        b = logits.size(0)
        target = torch.arange(b, device=logits.device)
        loss_ctx = F.cross_entropy(logits, target)
        loss_doc = F.cross_entropy(logits.t(), target)
        loss = 0.5 * (loss_ctx + loss_doc)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            r1 = (pred == target).float().mean().item()
            # recall@5 (guarded if batch < 5)
            topk = min(5, b)
            top = logits.topk(topk, dim=1).indices
            r5 = (top == target.unsqueeze(1)).any(dim=1).float().mean().item()
        return loss, {"r@1": r1, "r@5": r5}
