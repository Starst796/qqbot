"""导出需要向量化的语料，供本机（GPU）做一次性全量重算。

设计要点：
* 导出**全量**可向量化消息（而非只导 `embedded = 0`），这样本机重算后
  整个向量库口径统一，顺带覆盖掉早期用旧 `max_seq_length` 生成的那批向量。
* 只导出 `(id, group_id, content)` 三列：本机不需要任何其它上下文，
  而且**绝不打印正文**。

用法（服务器，项目 venv）：
    /home/ubuntu/qqbot/venv/bin/python deploy/export_corpus.py corpus.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bot"))

from qqbot_core import db  # noqa: E402
from qqbot_core.media import is_placeholder_only  # noqa: E402

QUERY = (
    "SELECT id, group_id, content FROM messages "
    "WHERE is_bot = 0 AND group_id IS NOT NULL AND content IS NOT NULL "
    "ORDER BY id ASC"
)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus.jsonl")
    conn = db.connect()

    total = 0
    written = 0
    skipped_placeholder = 0
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for row in conn.execute(QUERY):
            content = str(row["content"])
            total += 1
            if not content.strip():
                continue
            # 与 msg_store 的 embeddable=bool(text) 对齐：
            # 纯 `[图片]` / `[表情x2]` 这类占位消息没有语义，不参与向量化
            if is_placeholder_only(content):
                skipped_placeholder += 1
                continue
            fh.write(
                json.dumps(
                    {"i": int(row["id"]), "g": str(row["group_id"]), "c": content},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    size_mb = out.stat().st_size / 1048576
    print(f"导出 {written} 条 -> {out}（{size_mb:.2f} MB）")
    print(
        f"  候选 {total} 条，\n"
        f"  跳过空内容 {total - written - skipped_placeholder} 条，\n"
        f"  跳过纯占位 {skipped_placeholder} 条"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
