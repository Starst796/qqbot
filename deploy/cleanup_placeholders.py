"""清理「纯占位消息」造成的垃圾向量，并把它们标记为已处理。

背景：`[图片:0033BA87….jpg]` / `[表情x2]` 这类内容没有任何语义，
不该进入向量库。历史聊天记录导入器把它们以 `embedded = 0` 写进库，
后台 rag 循环照单全收 —— 既白烧 CPU，又在检索矩阵里塞噪声。

rag 插件的循环现在会跳过占位内容（防御性，已覆盖后续所有来源），
本脚本负责**把已经产生的垃圾清掉**。可重复执行（幂等）。

用法（服务器，项目 venv）：
    /home/ubuntu/qqbot/venv/bin/python deploy/cleanup_placeholders.py [--dry-run]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bot"))

from qqbot_core import db  # noqa: E402
from qqbot_core.media import is_placeholder_only  # noqa: E402

CHUNK = 2000


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    conn = db.connect()

    placeholder_ids = [
        int(r["id"])
        for r in conn.execute(
            "SELECT id, content FROM messages "
            "WHERE is_bot = 0 AND group_id IS NOT NULL"
        )
        if is_placeholder_only(str(r["content"]))
    ]
    print(f"纯占位消息：{len(placeholder_ids)} 条")
    if not placeholder_ids:
        return 0

    marks = ",".join("?" for _ in placeholder_ids)
    junk = conn.execute(
        f"SELECT COUNT(*) FROM embeddings WHERE msg_id IN ({marks})", placeholder_ids
    ).fetchone()[0]
    still_pending = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE embedded = 0 AND is_bot = 0 "
        f"AND id IN ({marks})",
        placeholder_ids,
    ).fetchone()[0]
    print(f"  其中有垃圾向量：{junk} 条")
    print(f"  其中还挂在待补队列：{still_pending} 条")

    if dry_run:
        print("\n--dry-run：未做任何修改")
        return 0

    deleted = 0
    marked = 0
    for start in range(0, len(placeholder_ids), CHUNK):
        part = placeholder_ids[start : start + CHUNK]
        sub = ",".join("?" for _ in part)
        cur = conn.execute(f"DELETE FROM embeddings WHERE msg_id IN ({sub})", part)
        deleted += cur.rowcount or 0
        cur = conn.execute(
            f"UPDATE messages SET embedded = 1 WHERE embedded = 0 AND id IN ({sub})",
            part,
        )
        marked += cur.rowcount or 0
        conn.commit()

    print(f"\n已删除垃圾向量 {deleted} 条，已标记为已处理 {marked} 条")

    left = conn.execute(
        f"SELECT COUNT(*) FROM embeddings WHERE msg_id IN ({marks})", placeholder_ids
    ).fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE embedded = 0 AND is_bot = 0"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    print(f"复查：占位向量剩余 {left}，向量总数 {total}，待补 {pending}")
    print("\n重建内存向量缓存：sudo systemctl restart qqbot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
