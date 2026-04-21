"""
Contrastive training for the citation retriever.

Input: a JSONL file of {"context": str, "dois": [str, ...]} records produced
by `citenn.mine_arxiv`. At start-up we resolve every DOI to its paper text
(title + abstract) via Crossref metadata, dropping pairs we cannot resolve.
Then standard InfoNCE with in-batch negatives.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    pairs_file: str
    output_dir: str = "./checkpoints"
    backbone: str = "sentence-transformers/all-MiniLM-L6-v2"
    proj_dim: int = 256
    max_len: int = 256
    temperature: float = 0.05
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = 2000
    save_every: int = 250
    log_every: int = 10
    seed: int = 42
    device: str = "auto"
    dtype: str = "auto"
    crossref_email: Optional[str] = None


def _pick_device(pref: str):
    import torch
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _paper_text_from_metadata(meta: dict) -> Optional[str]:
    if not meta:
        return None
    title = ""
    if isinstance(meta.get("title"), list) and meta["title"]:
        title = meta["title"][0]
    elif isinstance(meta.get("title"), str):
        title = meta["title"]
    abstract = meta.get("abstract", "") or ""
    # Crossref abstracts often arrive wrapped in JATS tags <jats:p>...</jats:p>
    import re
    abstract = re.sub(r"<[^>]+>", " ", abstract)
    text = (title + ". " + abstract).strip()
    if len(text) < 20:
        return None
    return re.sub(r"\s+", " ", text)


def _load_pairs(path: Path, crossref_email: Optional[str]) -> list[tuple[str, str, str]]:
    """Return list of (context, doi, paper_text) tuples."""
    from citenn.crossref import CrossrefClient
    from tqdm.auto import tqdm

    cr = CrossrefClient(email=crossref_email)
    records: list[tuple[str, str, str]] = []
    seen_doi_text: dict[str, str] = {}

    raw = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"[data] {len(raw)} raw (context, dois) records from {path}")
    for r in tqdm(raw, desc="resolve", unit="rec"):
        ctx = r.get("context", "").strip()
        if not ctx:
            continue
        for doi in r.get("dois", []):
            if doi in seen_doi_text:
                text = seen_doi_text[doi]
            else:
                meta = cr.metadata(doi)
                text = _paper_text_from_metadata(meta) if meta else None
                if text:
                    seen_doi_text[doi] = text
            if not text:
                continue
            records.append((ctx, doi, text))
    print(f"[data] {len(records)} resolved pairs, {len(seen_doi_text)} unique DOIs")
    return records


def train(cfg: TrainConfig) -> None:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup
    from tqdm.auto import tqdm

    from citenn.model import BiEncoder, BiEncoderConfig

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = _pick_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} backbone={cfg.backbone}")

    pairs = _load_pairs(Path(cfg.pairs_file), cfg.crossref_email)
    if len(pairs) < cfg.batch_size:
        raise RuntimeError(
            f"only {len(pairs)} resolved pairs, need at least batch_size={cfg.batch_size}"
        )

    model = BiEncoder(BiEncoderConfig(
        backbone=cfg.backbone,
        proj_dim=cfg.proj_dim,
        max_len=cfg.max_len,
        temperature=cfg.temperature,
    )).to(device)

    optim = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    sched = get_cosine_schedule_with_warmup(optim, cfg.warmup_steps, cfg.max_steps)

    def sample_batch():
        batch = random.sample(pairs, cfg.batch_size) if len(pairs) >= cfg.batch_size else pairs
        # In-batch negatives degrade if a DOI appears twice in the same batch:
        # force unique DOIs per batch.
        seen: set[str] = set()
        uniq: list[tuple[str, str, str]] = []
        attempts = 0
        while len(uniq) < cfg.batch_size and attempts < cfg.batch_size * 5:
            c, d, t = random.choice(pairs)
            if d in seen:
                attempts += 1
                continue
            seen.add(d)
            uniq.append((c, d, t))
            attempts += 1
        if len(uniq) < cfg.batch_size:
            uniq = batch  # fall back
        return uniq

    log_csv = out_dir / "loss.csv"
    if not log_csv.exists():
        log_csv.write_text("step,loss,r@1,r@5,lr,elapsed_sec\n")

    pbar = tqdm(total=cfg.max_steps, desc="train", unit="step", dynamic_ncols=True)
    model.train()
    run_t0 = time.time()
    loss_running = 0.0
    r1_running = 0.0
    r5_running = 0.0

    for step in range(1, cfg.max_steps + 1):
        batch = sample_batch()
        ctxs = [b[0] for b in batch]
        docs = [b[2] for b in batch]
        z_ctx = model.encode_contexts(ctxs, device=device)
        z_doc = model.encode_papers(docs, device=device)
        loss, metrics = model.info_nce(z_ctx, z_doc)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()

        loss_running += loss.item()
        r1_running += metrics["r@1"]
        r5_running += metrics["r@5"]
        pbar.update(1)

        if step % cfg.log_every == 0:
            avg_loss = loss_running / cfg.log_every
            avg_r1 = r1_running / cfg.log_every
            avg_r5 = r5_running / cfg.log_every
            lr_now = sched.get_last_lr()[0]
            elapsed = time.time() - run_t0
            with log_csv.open("a") as f:
                f.write(
                    f"{step},{avg_loss:.6f},{avg_r1:.4f},{avg_r5:.4f},"
                    f"{lr_now:.3e},{elapsed:.1f}\n"
                )
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                r1=f"{avg_r1:.3f}",
                r5=f"{avg_r5:.3f}",
                lr=f"{lr_now:.1e}",
            )
            loss_running = r1_running = r5_running = 0.0

        if step % cfg.save_every == 0 or step == cfg.max_steps:
            ckpt = out_dir / f"step-{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "cfg": vars(cfg),
            }, ckpt / "model.pt")
            model.tokenizer.save_pretrained(ckpt)
            pbar.write(f"[save] {ckpt}")

    pbar.close()
    print("[done] training complete")
