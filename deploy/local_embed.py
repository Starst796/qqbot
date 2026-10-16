"""在本机（CPU/CUDA）对导出的语料做一次性向量化，产出可回传的 npz。

**为什么复用仓库里的 `qqbot_core.embeddings.Embedder`**：
向量能不能和服务器上已有的混用，取决于「模型 + 池化 + 预处理」三者一致。
直接 import 服务器在用的那一份代码，就天然保证了截断长度、
`max_seq_length`、长度排序、L2 归一化全都一模一样，
不会出现「本地一套口径、线上一套口径」的分叉。

用法：
    python deploy/local_embed.py corpus.jsonl vectors.npz D:\\gits\\bge-m3
    python deploy/local_embed.py corpus.jsonl vectors.npz D:\\gits\\bge-m3 --device cpu --batch 64

`--model` 既可以是本地目录，也可以是 HF 模型名（本地目录不联网、更快更稳）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bot"))

import numpy as np  # noqa: E402

from qqbot_core.embeddings import MAX_CHARS, MAX_SEQ_LENGTH, Embedder  # noqa: E402


def load_corpus(path: str) -> tuple[list[int], list[str], list[str]]:
    ids: list[int] = []
    groups: list[str] = []
    texts: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            ids.append(int(o["i"]))
            groups.append(str(o["g"]))
            texts.append(str(o["c"]))
    return ids, groups, texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="export_corpus.py 产出的 jsonl")
    ap.add_argument("out", help="输出的 .npz")
    ap.add_argument(
        "--model",
        default=r"D:\gits\bge-m3",
        help="本地模型目录或 HF 模型名（默认 D:\\gits\\bge-m3）",
    )
    ap.add_argument("--device", default="cuda", help="cuda / cpu（默认 cuda）")
    ap.add_argument("--batch", type=int, default=256, help="每批编码条数（GPU 用 256）")
    ap.add_argument(
        "--chunk", type=int, default=4096, help="每次交给 encode 的条数（控制内存）"
    )
    args = ap.parse_args()

    ids, groups, texts = load_corpus(args.corpus)
    if not texts:
        print("语料是空的", file=sys.stderr)
        return 1
    print(f"读入 {len(texts)} 条")

    emb = Embedder(args.model, device=args.device, batch_size=args.batch)
    if not emb.available:
        print(f"模型加载失败：{emb.error}", file=sys.stderr)
        return 1
    print(
        f"模型 {args.model}\n"
        f"  dim={emb.dim}  device={args.device}  batch={args.batch}\n"
        f"  预处理：截断 {MAX_CHARS} 字符 / max_seq_length={MAX_SEQ_LENGTH}"
        "（与服务器同源，务必保持一致）"
    )

    mat = np.zeros((len(texts), emb.dim), dtype=np.float32)
    t0 = time.time()
    step = max(1, min(args.chunk, len(texts)))
    for i in range(0, len(texts), step):
        chunk = texts[i : i + step]
        mat[i : i + len(chunk)] = emb.encode(chunk)
        done = i + len(chunk)
        elapsed = max(1e-6, time.time() - t0)
        print(
            f"  {done}/{len(texts)}  {done / elapsed:.0f} 条/秒"
            f"  已用 {elapsed:.0f}s",
            flush=True,
        )

    elapsed = time.time() - t0
    # 自检：Embedder 内部已做 L2 归一化，模长应全部约等于 1
    norms = np.linalg.norm(mat, axis=1)
    print(
        f"完成 {mat.shape}  耗时 {elapsed:.1f}s  "
        f"（{len(texts) / max(1e-6, elapsed):.0f} 条/秒）"
    )
    print(f"归一化自检：min={norms.min():.5f} max={norms.max():.5f}（应≈1.0）")

    np.savez(
        args.out,
        ids=np.asarray(ids, dtype=np.int64),
        groups=np.asarray(groups),
        vecs=mat,
    )
    print(f"写出 {args.out}（{Path(args.out).stat().st_size / 1048576:.1f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
