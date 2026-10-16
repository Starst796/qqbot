"""把本机算好的向量导回服务器数据库。

语义是 **upsert**（`INSERT OR REPLACE` + 标记 embedded=1），**不做全表删除**：
导出到导入之间线上还会有新消息进来，删表会让它们丢掉已算好的向量却
仍被标记成 `embedded=1`，从此再也补不上。

用法（服务器，项目 venv）：
    /home/ubuntu/qqbot/venv/bin/python deploy/import_vectors.py vectors.npz

导入完成后需要重启服务，让 rag 的内存向量缓存失效重建：
    sudo systemctl restart qqbot      # 或面板上的「重启服务」
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bot"))

from qqbot_core import db  # noqa: E402
from qqbot_core.embeddings import vec_to_blob  # noqa: E402

# 单次 executemany 的条数：24k 条一次性提交会占不少内存，分块更稳
CHUNK = 2000


def sample_check(conn: sqlite3.Connection, ids: np.ndarray, groups: np.ndarray) -> int:
    """抽查若干条，确认 npz 里的 group_id 与库中消息对得上。"""
    n = len(ids)
    if n == 0:
        return 0
    idx = np.linspace(0, n - 1, min(200, n)).astype(int)
    probe = [(int(ids[i]), str(groups[i])) for i in idx]
    marks = ",".join("?" for _ in probe)
    actual = {
        int(r["id"]): str(r["group_id"])
        for r in conn.execute(
            f"SELECT id, group_id FROM messages WHERE id IN ({marks})",
            [p[0] for p in probe],
        )
    }
    bad = 0
    missing = 0
    for msg_id, group_id in probe:
        if msg_id not in actual:
            missing += 1
        elif actual[msg_id] != group_id:
            bad += 1
    print(f"抽查 {len(probe)} 条：群号不符 {bad} 条，库中不存在 {missing} 条")
    return bad + missing


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"找不到文件：{path}", file=sys.stderr)
        return 1

    data = np.load(path, allow_pickle=False)
    ids = data["ids"]
    groups = data["groups"]
    vecs = data["vecs"]
    if not (len(ids) == len(groups) == len(vecs)):
        print("npz 三个数组长度不一致，文件损坏？", file=sys.stderr)
        return 1
    dim = int(vecs.shape[1])
    print(f"读入 {len(ids)} 条向量，dim={dim}，dtype={vecs.dtype}")

    conn = db.connect()

    existing = {
        int(r["dim"]) for r in conn.execute("SELECT DISTINCT dim FROM embeddings")
    }
    if existing and existing != {dim}:
        print(
            f"⚠️ 库里已有维度 {sorted(existing)} 的向量，与本次 {dim} 不一致！\n"
            f"   混用不同模型的向量会让余弦相似度变成噪声，已中止。",
            file=sys.stderr,
        )
        return 1

    if sample_check(conn, ids, groups):
        print("抽查不通过，已中止（先确认 npz 与当前数据库是否配套）", file=sys.stderr)
        return 1

    before = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    pending_before = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE embedded = 0 AND is_bot = 0"
    ).fetchone()[0]

    inserted = 0
    for start in range(0, len(ids), CHUNK):
        stop = min(start + CHUNK, len(ids))
        rows = [
            (int(ids[i]), str(groups[i]), dim, vec_to_blob(vecs[i]))
            for i in range(start, stop)
        ]
        db.add_embeddings(rows)
        db.mark_embedded(int(ids[i]) for i in range(start, stop))
        inserted += len(rows)
        print(f"  写入 {inserted}/{len(ids)}", flush=True)

    after = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    pending_after = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE embedded = 0 AND is_bot = 0"
    ).fetchone()[0]

    print(
        f"\n向量表：{before} -> {after}（本次写入 {inserted} 条）\n"
        f"待补消息：{pending_before} -> {pending_after}\n"
        "\n别忘了重启服务让内存向量缓存失效：sudo systemctl restart qqbot"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
