#!/usr/bin/env python3
"""把 QQChatExporter 导出的聊天记录 JSON 导入机器人语料库。

用法（服务器上，用项目 venv）：
    /home/ubuntu/qqbot/venv/bin/python deploy/import_chat_export.py <export.json> [--dry-run]

设计：
  * 复用 qqbot_core.db —— 表结构与 FTS 索引和机器人完全一致；
  * 以 (group_id, user_id, 秒级 ts, content) 去重，脚本可重复执行；
  * 只导入真实聊天：跳过系统消息 / 撤回 / 纯占位符（[图片] 等）/ 空文本；
  * **不打印任何消息正文**，只报统计（保护隐私）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent.parent / "bot"
sys.path.insert(0, str(BOT_DIR))

from qqbot_core import db  # noqa: E402  (需要先设置 sys.path)

PLACEHOLDER_RE = re.compile(r"^\[[^\[\]]{1,12}\]$")
KEEP_TYPES = {"text", "reply", "forward"}
BATCH = 2000


def clean_text(msg: dict) -> str:
    content = msg.get("content") or {}
    return " ".join(str(content.get("text") or "").split())


def sender_of(msg: dict) -> tuple[str, str]:
    sender = msg.get("sender") or {}
    uid = str(sender.get("uin") or sender.get("uid") or "")
    name = sender.get("groupCard") or sender.get("nickname") or sender.get("name") or ""
    return uid, str(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--group-id", default="")
    ap.add_argument(
        "--bot-uin",
        default="",
        help="机器人自身 QQ；用于把它的发言标成 is_bot（导出文件的 selfUin 是导出者本人）",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    chat = data.get("chatInfo") or {}
    group_id = args.group_id or str(chat.get("peerUid") or "")
    self_uin = str(chat.get("selfUin") or "")
    messages = data.get("messages") or []
    if not group_id:
        print("无法确定 group_id（chatInfo.peerUid 为空）；可用 --group-id 指定")
        return 2

    print(f"群号: {group_id}   导出者QQ: {self_uin or '(未知)'}   导出条数: {len(messages)}")
    if args.bot_uin:
        print(f"机器人QQ(标 is_bot): {args.bot_uin}")
    existing = db.message_keys(group_id)
    print(f"库中已有该群消息: {len(existing)}")

    reasons: Counter[str] = Counter()
    users: Counter[str] = Counter()
    batch: list[dict] = []
    added = 0
    t_min: float | None = None
    t_max: float | None = None

    def flush() -> None:
        nonlocal batch, added
        if batch:
            if args.dry_run:
                added += len(batch)
            else:
                added += db.add_messages_bulk(batch)
            batch = []

    for msg in messages:
        if msg.get("system"):
            reasons["system"] += 1
            continue
        if msg.get("recalled"):
            reasons["recalled"] += 1
            continue
        mtype = str(msg.get("type") or "")
        if mtype not in KEEP_TYPES:
            reasons[f"type:{mtype}"] += 1
            continue
        text = clean_text(msg)
        if not text:
            reasons["empty"] += 1
            continue
        if PLACEHOLDER_RE.match(text):
            reasons["placeholder"] += 1
            continue
        uid, name = sender_of(msg)
        if not uid:
            reasons["no_sender"] += 1
            continue
        ts = float(msg.get("timestamp") or 0) / 1000.0
        if ts <= 0:
            reasons["no_ts"] += 1
            continue

        key = (uid, text, round(ts))
        if key in existing:
            reasons["duplicate"] += 1
            continue
        existing.add(key)

        users[uid] += 1
        t_min = ts if t_min is None else min(t_min, ts)
        t_max = ts if t_max is None else max(t_max, ts)
        batch.append(
            {
                "group_id": group_id,
                "user_id": uid,
                "nickname": name,
                "content": text,
                "ts": ts,
                "is_bot": bool(args.bot_uin) and uid == args.bot_uin,
            }
        )
        if len(batch) >= BATCH:
            flush()

    flush()

    print("\n跳过原因统计:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")
    print(f"\n参与发言人数: {len(users)}")
    if t_min and t_max:
        print(
            "时间范围:",
            dt.datetime.fromtimestamp(t_min).strftime("%Y-%m-%d %H:%M"),
            "→",
            dt.datetime.fromtimestamp(t_max).strftime("%Y-%m-%d %H:%M"),
        )
    print(f"{'将导入' if args.dry_run else '已导入'}: {added} 条消息")
    if args.dry_run:
        print("（dry-run：未写库）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
