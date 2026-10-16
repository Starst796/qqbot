"""把一个群的接话对复制一份给另一个群（测试用）。

向量（vec）**原样复制**：两个群用的是同一个 embedding 模型，向量空间相同，
所以不需要重新编码 —— 800 条在服务器 CPU 上要编码约 50 秒，复制是毫秒级。

用法（服务器，项目 venv）：
    python deploy/copy_style_pairs.py <源群号> <目标群号> [--dry-run]

复制后服务的进程内缓存需要失效，最省事的办法（不重启、不中断服务）：
    curl -s -X POST http://127.0.0.1:8081/api/style/delete \\
         -H 'Content-Type: application/json' -d '{"ids":[]}'
（空 ids 是 no-op 删除，但会把 style_distill 的缓存全部丢弃）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bot"))

from qqbot_core import db  # noqa: E402


def counts(conn, group_id: str) -> tuple[int, int]:
    total = conn.execute(
        "SELECT COUNT(*) FROM style_pairs WHERE group_id = ?", (group_id,)
    ).fetchone()[0]
    with_vec = conn.execute(
        "SELECT COUNT(*) FROM style_pairs "
        "WHERE group_id = ? AND vec IS NOT NULL AND dim > 0",
        (group_id,),
    ).fetchone()[0]
    return int(total), int(with_vec)


def main() -> int:
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in sys.argv
    if len(positional) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    src, dst = positional[0], positional[1]
    if src == dst:
        print("源群与目标群相同，无需复制", file=sys.stderr)
        return 2

    conn = db.connect()
    src_total, src_vec = counts(conn, src)
    dst_total, _ = counts(conn, dst)
    print(f"源群   {src}: {src_total} 对（带向量 {src_vec} 对）")
    print(f"目标群 {dst}: {dst_total} 对（复制前）")

    rows = list(
        conn.execute(
            "SELECT context, reply, score, source, dim, vec FROM style_pairs "
            "WHERE group_id = ? ORDER BY score DESC, id DESC",
            (src,),
        )
    )
    if not rows:
        print(f"源群 {src} 没有接话对：先去面板对该群点一次「只重建该群」", file=sys.stderr)
        return 1

    payload = [
        (r["context"], r["reply"], r["score"], r["source"], r["dim"], r["vec"])
        for r in rows
    ]
    missing = sum(1 for item in payload if not item[5])
    print(f"待复制 {len(payload)} 对，其中缺向量 {missing} 对")
    if missing:
        # 目标群只要有任意一行缺向量，_load_matrix 就会整组返回空矩阵，
        # 语义召回退化成字符 2-gram，所以这里要提醒
        print("⚠️ 源群里存在缺向量的对，建议先重建源群再接续测试", file=sys.stderr)

    if dry_run:
        print("\n--dry-run：未写入任何数据")
        return 0

    written = db.replace_style_pairs(dst, payload)
    after_total, after_vec = counts(conn, dst)
    print(f"\n已复制 {written} 对到群 {dst}")
    print(f"目标群 {dst}: {after_total} 对（带向量 {after_vec} 对）")
    print("\n别忘了让服务内的缓存失效（见脚本头部说明）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
