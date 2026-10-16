"""SQLite 存储层：消息落库、FTS5 索引、向量表、会话上下文。

并发模型：本项目为单进程 asyncio，DB 操作都很快；这里用一个带锁的共享连接
（`check_same_thread=False`）串行化写入，避免 WAL 下的写冲突。真正的重活
（embedding 计算）在插件层用线程执行，不占用本锁。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from . import config

DB_PATH = config.DATA_DIR / "qqbot.db"

_conn: sqlite3.Connection | None = None
_fts_enabled = False
_lock = threading.RLock()

# 不依赖 FTS：在任何 SQLite 版本上都能建
_SCHEMA_BASE = """
CREATE TABLE IF NOT EXISTS messages (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT,
    user_id  TEXT NOT NULL,
    nickname TEXT,
    content  TEXT NOT NULL,
    ts       REAL NOT NULL,
    is_bot   INTEGER NOT NULL DEFAULT 0,
    embedded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_group_ts ON messages(group_id, ts);
CREATE INDEX IF NOT EXISTS idx_messages_embedded ON messages(embedded) WHERE embedded = 0;

-- 向量表：vec 为 float32 小端字节序列
CREATE TABLE IF NOT EXISTS embeddings (
    msg_id   INTEGER PRIMARY KEY,
    group_id TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_group ON embeddings(group_id);

-- 每群最近对话（给模型当上下文）
CREATE TABLE IF NOT EXISTS sessions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_key TEXT NOT NULL,
    role     TEXT NOT NULL,
    content  TEXT NOT NULL,
    ts       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_key ON sessions(conv_key, id);
"""

# FTS5 是 embedding 不可用时的降级检索方案。trigram 需 SQLite >= 3.34，
# 旧版本回退到 unicode61；两者都不可用则静默关闭该能力。
_FTS_VARIANTS = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts "
    "USING fts5(content, content='messages', content_rowid='id', tokenize='trigram')",
    "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts "
    "USING fts5(content, content='messages', content_rowid='id', tokenize='unicode61')",
)


def connect() -> sqlite3.Connection:
    global _conn, _fts_enabled
    with _lock:
        if _conn is None:
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA synchronous=NORMAL")
            _conn.executescript(_SCHEMA_BASE)
            _conn.commit()
            for stmt in _FTS_VARIANTS:
                try:
                    _conn.execute(stmt)
                    _conn.commit()
                    _fts_enabled = True
                    break
                except sqlite3.Error:
                    continue
        return _conn


def add_message(
    *,
    user_id: str,
    content: str,
    group_id: str | None = None,
    nickname: str | None = None,
    is_bot: bool = False,
    ts: float | None = None,
    embeddable: bool = True,
) -> int:
    """写入一条消息，返回其 id。

    `embeddable=False` 用于纯图片/表情的占位消息：它们没有语义，
    直接标记为已处理，免得往向量库里灌一堆无意义的 `[图片]`。
    """
    content = (content or "").strip()
    if not content:
        return -1
    conn = connect()
    with _lock:
        cur = conn.execute(
            "INSERT INTO messages(group_id, user_id, nickname, content, ts, is_bot, embedded) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                group_id,
                user_id,
                nickname,
                content,
                ts or time.time(),
                1 if is_bot else 0,
                0 if embeddable else 1,
            ),
        )
        conn.commit()
        rowid = int(cur.lastrowid or -1)
        if _fts_enabled:
            try:
                conn.execute(
                    "INSERT INTO messages_fts(rowid, content) VALUES (?, ?)",
                    (rowid, content),
                )
                conn.commit()
            except sqlite3.Error:
                # FTS 写入失败不影响主流程
                pass
        return rowid


def fetch_pending_embeddings(limit: int = 64) -> list[sqlite3.Row]:
    """取出尚未做 embedding 的消息（跳过机器人自己的）。"""
    conn = connect()
    with _lock:
        return list(
            conn.execute(
                "SELECT id, group_id, content FROM messages "
                "WHERE embedded = 0 AND is_bot = 0 AND group_id IS NOT NULL "
                "ORDER BY id ASC LIMIT ?",
                (limit,),
            )
        )


def mark_embedded(msg_ids: Iterable[int]) -> None:
    ids = list(msg_ids)
    if not ids:
        return
    conn = connect()
    with _lock:
        conn.executemany(
            "UPDATE messages SET embedded = 1 WHERE id = ?", [(i,) for i in ids]
        )
        conn.commit()


def add_embeddings(rows: list[tuple[int, str, int, bytes]]) -> None:
    if not rows:
        return
    conn = connect()
    with _lock:
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings(msg_id, group_id, dim, vec) VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()


def load_group_vectors(group_id: str) -> list[sqlite3.Row]:
    conn = connect()
    with _lock:
        return list(
            conn.execute(
                "SELECT e.msg_id AS msg_id, e.dim AS dim, e.vec AS vec, m.content AS content "
                "FROM embeddings e JOIN messages m ON m.id = e.msg_id "
                "WHERE e.group_id = ?",
                (group_id,),
            )
        )


def prune_group_vectors(group_id: str, keep: int) -> None:
    """每组只保留最新的 keep 条向量，避免内存与磁盘无限增长。"""
    conn = connect()
    with _lock:
        conn.execute(
            "DELETE FROM embeddings WHERE group_id = ? AND msg_id NOT IN ("
            "  SELECT id FROM messages WHERE group_id = ? ORDER BY id DESC LIMIT ?"
            ")",
            (group_id, group_id, keep),
        )
        conn.commit()


def fts_search(group_id: str, query: str, limit: int = 5) -> list[str]:
    """FTS5 关键词检索（降级方案）。query 会做简单转义。"""
    normalized = "".join(ch for ch in (query or "") if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    if len(normalized) < 2:
        return []
    if not _fts_enabled:
        return []
    conn = connect()
    with _lock:
        try:
            rows = conn.execute(
                "SELECT m.content FROM messages_fts f JOIN messages m ON m.id = f.rowid "
                "WHERE messages_fts MATCH ? AND m.group_id = ? "
                "ORDER BY rank LIMIT ?",
                (f'"{normalized}"', group_id, limit),
            )
            return [r["content"] for r in rows]
        except sqlite3.Error:
            return []


def add_session(conv_key: str, role: str, content: str) -> None:
    conn = connect()
    with _lock:
        conn.execute(
            "INSERT INTO sessions(conv_key, role, content, ts) VALUES (?,?,?,?)",
            (conv_key, role, content, time.time()),
        )
        conn.commit()


def get_session(conv_key: str, turns: int) -> list[dict[str, str]]:
    conn = connect()
    with _lock:
        rows = list(
            conn.execute(
                "SELECT role, content FROM sessions WHERE conv_key = ? "
                "ORDER BY id DESC LIMIT ?",
                (conv_key, max(1, turns) * 2),
            )
        )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def stats() -> dict[str, Any]:
    conn = connect()
    with _lock:
        total = conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"]
        embedded = conn.execute(
            "SELECT COUNT(*) AS c FROM embeddings"
        ).fetchone()["c"]
        groups = conn.execute(
            "SELECT COUNT(DISTINCT group_id) AS c FROM messages WHERE group_id IS NOT NULL"
        ).fetchone()["c"]
        last = conn.execute("SELECT MAX(ts) AS t FROM messages").fetchone()["t"]
    return {
        "messages": int(total),
        "vectors": int(embedded),
        "groups": int(groups),
        "last_message_ts": float(last) if last else None,
        "db_path": str(DB_PATH),
    }


# --------------------------------------------------------------------------- #
# 上下文 / 数据管理（供管理面板使用）
# --------------------------------------------------------------------------- #
def _rebuild_fts() -> None:
    """messages_fts 是 external content 表，主表删改后必须 rebuild 才保持一致。"""
    if not _fts_enabled:
        return
    try:
        conn = connect()
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        conn.commit()
    except sqlite3.Error:
        pass


def _drop_orphan_embeddings() -> None:
    conn = connect()
    conn.execute("DELETE FROM embeddings WHERE msg_id NOT IN (SELECT id FROM messages)")
    conn.commit()


def list_groups() -> list[dict[str, Any]]:
    """列出有消息记录的群及其统计（消息数 / 机器人发言数 / 上下文条数 / 最后时间）。"""
    conn = connect()
    with _lock:
        rows = list(
            conn.execute(
                "SELECT group_id, COUNT(*) AS messages, SUM(is_bot) AS bot_messages, "
                "MAX(ts) AS last_ts FROM messages WHERE group_id IS NOT NULL "
                "GROUP BY group_id ORDER BY last_ts DESC"
            )
        )
        sess = {
            r["conv_key"]: int(r["c"])
            for r in conn.execute(
                "SELECT conv_key, COUNT(*) AS c FROM sessions GROUP BY conv_key"
            )
        }
    return [
        {
            "group_id": r["group_id"],
            "messages": int(r["messages"]),
            "bot_messages": int(r["bot_messages"] or 0),
            "last_ts": float(r["last_ts"]) if r["last_ts"] else None,
            "sessions": sess.get(f"group:{r['group_id']}", 0),
        }
        for r in rows
    ]


def recent_messages(
    group_id: str, limit: int = 50, offset: int = 0
) -> list[dict[str, Any]]:
    """按时间倒序取某群最近的消息，供面板浏览/单条删除。"""
    conn = connect()
    with _lock:
        rows = list(
            conn.execute(
                "SELECT id, user_id, nickname, content, ts, is_bot FROM messages "
                "WHERE group_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (group_id, max(1, limit), max(0, offset)),
            )
        )
    return [dict(r) for r in rows]


def delete_messages(
    *,
    group_id: str | None = None,
    user_id: str | None = None,
    ids: list[int] | None = None,
) -> int:
    """删除消息（含其向量）。三者都不给 = 清空全部。返回删除条数。"""
    conn = connect()
    with _lock:
        if ids:
            marks = ",".join("?" * len(ids))
            cur = conn.execute(
                f"DELETE FROM messages WHERE id IN ({marks})", list(ids)
            )
        elif group_id is not None:
            cur = conn.execute("DELETE FROM messages WHERE group_id = ?", (group_id,))
        elif user_id is not None:
            cur = conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        else:
            cur = conn.execute("DELETE FROM messages")
        removed = int(cur.rowcount or 0)
        conn.commit()
        _drop_orphan_embeddings()
        _rebuild_fts()
    return removed


def clear_sessions(
    *, group_id: str | None = None, user_id: str | None = None
) -> int:
    """清空对话上下文（不影响聊天记录语料）。都不给 = 清空全部。"""
    conn = connect()
    with _lock:
        if group_id is not None:
            cur = conn.execute(
                "DELETE FROM sessions WHERE conv_key = ?", (f"group:{group_id}",)
            )
        elif user_id is not None:
            cur = conn.execute(
                "DELETE FROM sessions WHERE conv_key = ?", (f"user:{user_id}",)
            )
        else:
            cur = conn.execute("DELETE FROM sessions")
        removed = int(cur.rowcount or 0)
        conn.commit()
    return removed
