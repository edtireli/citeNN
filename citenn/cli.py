"""citeNN command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cmd_mine(args: argparse.Namespace) -> int:
    from citenn.mine_arxiv import mine_arxiv_tarball, write_jsonl
    tarballs = [Path(p) for p in args.tarballs]
    all_records: list[dict] = []
    for p in tarballs:
        recs = mine_arxiv_tarball(p)
        print(f"[mine] {p}: {len(recs)} records")
        for r in recs:
            r["source"] = str(p)
        all_records.extend(recs)
    n = write_jsonl(all_records, args.out)
    print(f"[mine] wrote {n} records to {args.out}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from citenn.train import TrainConfig, train
    cfg = TrainConfig(
        pairs_file=args.pairs,
        output_dir=args.output_dir,
        backbone=args.backbone,
        proj_dim=args.proj_dim,
        max_len=args.max_len,
        temperature=args.temperature,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        device=args.device,
        crossref_email=args.email,
    )
    train(cfg)
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    from citenn.cite import build_index_from_pairs
    n = build_index_from_pairs(
        pairs_file=args.pairs,
        checkpoint_dir=args.checkpoint,
        index_dir=args.out,
        crossref_email=args.email,
    )
    print(f"[index] indexed {n} papers into {args.out}")
    return 0


def _cmd_cite(args: argparse.Namespace) -> int:
    from citenn.cite import suggest_citations
    text = Path(args.inp).read_text() if args.inp else args.text
    if not text:
        raise SystemExit("provide --in FILE or --text STRING")
    annotated, bib = suggest_citations(
        text=text,
        checkpoint_dir=args.checkpoint,
        index_dir=args.index,
        k=args.k,
        min_score=args.min_score,
        crossref_email=args.email,
    )
    out_tex = Path(args.out_tex) if args.out_tex else None
    out_bib = Path(args.out_bib) if args.out_bib else None
    if out_tex:
        out_tex.write_text(annotated)
        print(f"[cite] wrote annotated text -> {out_tex}")
    else:
        print(annotated)
    if out_bib:
        out_bib.write_text(bib)
        print(f"[cite] wrote bibtex -> {out_bib}")
    elif bib:
        print("\n% --- BIBTEX ---")
        print(bib)
    return 0


def _cmd_bibtex(args: argparse.Namespace) -> int:
    from citenn.crossref import doi_to_bibtex
    bib = doi_to_bibtex(args.doi, email=args.email)
    if not bib:
        raise SystemExit(f"could not resolve DOI: {args.doi}")
    if args.out:
        Path(args.out).write_text(bib)
    else:
        print(bib)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="citenn")
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("mine", help="mine (context, DOI) pairs from arXiv source tarballs")
    pm.add_argument("tarballs", nargs="+", help="paths to arXiv .tar.gz source archives")
    pm.add_argument("--out", required=True, help="output JSONL path")
    pm.set_defaults(func=_cmd_mine)

    pt = sub.add_parser("train", help="contrastive training of the bi-encoder")
    pt.add_argument("--pairs", required=True)
    pt.add_argument("--output-dir", default="./checkpoints")
    pt.add_argument("--backbone", default="sentence-transformers/all-MiniLM-L6-v2")
    pt.add_argument("--proj-dim", type=int, default=256)
    pt.add_argument("--max-len", type=int, default=256)
    pt.add_argument("--temperature", type=float, default=0.05)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--lr", type=float, default=2e-5)
    pt.add_argument("--max-steps", type=int, default=2000)
    pt.add_argument("--save-every", type=int, default=250)
    pt.add_argument("--log-every", type=int, default=10)
    pt.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    pt.add_argument("--email", default=None, help="email for polite Crossref User-Agent")
    pt.set_defaults(func=_cmd_train)

    pi = sub.add_parser("index", help="build paper embedding index")
    pi.add_argument("--pairs", required=True)
    pi.add_argument("--checkpoint", required=True)
    pi.add_argument("--out", required=True)
    pi.add_argument("--email", default=None)
    pi.set_defaults(func=_cmd_index)

    pc = sub.add_parser("cite", help="annotate text with \\cite{...} + bibtex")
    pc.add_argument("--in", dest="inp", default=None)
    pc.add_argument("--text", default=None)
    pc.add_argument("--checkpoint", required=True)
    pc.add_argument("--index", required=True)
    pc.add_argument("--k", type=int, default=3)
    pc.add_argument("--min-score", type=float, default=0.45)
    pc.add_argument("--out-tex", default=None)
    pc.add_argument("--out-bib", default=None)
    pc.add_argument("--email", default=None)
    pc.set_defaults(func=_cmd_cite)

    pb = sub.add_parser("bibtex", help="fetch a BibTeX entry for one DOI")
    pb.add_argument("doi")
    pb.add_argument("--out", default=None)
    pb.add_argument("--email", default=None)
    pb.set_defaults(func=_cmd_bibtex)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
