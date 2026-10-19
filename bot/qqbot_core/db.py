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

-- 学习型复读：关键词桶 -> 学到的接话（count 越大越常用）
CREATE TABLE IF NOT EXISTS repeat_learn (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL,
    keywords TEXT NOT NULL,
    answer   TEXT NOT NULL,
    count    INTEGER NOT NULL DEFAULT 1,
    ts       REAL NOT NULL,
    UNIQUE(group_id, keywords, answer)
);
CREATE INDEX IF NOT EXISTS idx_repeat_learn_lookup ON repeat_learn(group_id, keywords);

-- 风格蒸馏：从真实相邻对话里抽出的「前句 -> 接话」对
CREATE TABLE IF NOT EXISTS style_pairs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id   TEXT NOT NULL,
    context    TEXT NOT NULL,
    reply      TEXT NOT NULL,
    score      REAL NOT NULL DEFAULT 1.0,
    source     TEXT NOT NULL DEFAULT 'heuristic',
    dim        INTEGER NOT NULL DEFAULT 0,
    vec        BLOB,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_style_pairs_group ON style_pairs(group_id);

-- 表情包库：persona=默认人设图（永远排最前）；approved=0 为待审核的学习图
CREATE TABLE IF NOT EXISTS stickers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256       TEXT NOT NULL UNIQUE,
    file_path    TEXT NOT NULL,
    mime         TEXT NOT NULL DEFAULT 'image/png',
    bytes        INTEGER NOT NULL DEFAULT 0,
    source       TEXT NOT NULL DEFAULT 'learned',
    group_id     TEXT,
    user_id      TEXT,
    mface_id     TEXT,
    summary      TEXT,
    tags         TEXT NOT NULL DEFAULT '',
    mood         TEXT,
    enabled      INTEGER NOT NULL DEFAULT 1,
    approved     INTEGER NOT NULL DEFAULT 0,
    persona      INTEGER NOT NULL DEFAULT 0,
    sort_order   INTEGER NOT NULL DEFAULT 1000,
    weight       REAL NOT NULL DEFAULT 1.0,
    seen_count   INTEGER NOT NULL DEFAULT 1,
    use_count    INTEGER NOT NULL DEFAULT 0,
    last_used_at REAL,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stickers_pick ON stickers(enabled, approved, persona, sort_order);

-- 发送记录：用于同群冷却、近期去重与使用统计
CREATE TABLE IF NOT EXISTS sticker_usage (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    sticker_id INTEGER NOT NULL,
    group_id   TEXT,
    ts         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sticker_usage_group ON sticker_usage(group_id, ts);
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


# --------------------------------------------------------------------------- #
# 学习型复读（repeat_learn）
# --------------------------------------------------------------------------- #
def learn_repeat_answer(
    group_id: str, keywords: str, answer: str, ts: float | None = None
) -> None:
    """记一次「关键词桶 -> 接话」；同一对重复出现则 count+1。"""
    if not keywords or not answer:
        return
    conn = connect()
    with _lock:
        conn.execute(
            "INSERT INTO repeat_learn(group_id, keywords, answer, count, ts) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(group_id, keywords, answer) "
            "DO UPDATE SET count = count + 1, ts = excluded.ts",
            (group_id, keywords, answer, 1, ts or time.time()),
        )
        conn.commit()


def repeat_answers(group_id: str, keywords: str, limit: int = 32) -> list[sqlite3.Row]:
    """取某个关键词桶下的候选取话（按常用度、新鲜度排序）。"""
    if not keywords:
        return []
    conn = connect()
    with _lock:
        return list(
            conn.execute(
                "SELECT answer, count, ts FROM repeat_learn "
                "WHERE group_id = ? AND keywords = ? "
                "ORDER BY count DESC, ts DESC LIMIT ?",
                (group_id, keywords, max(1, limit)),
            )
        )


def prune_repeat_learn(group_id: str, keep_per_bucket: int) -> int:
    """每个关键词桶只保留最常用的 keep_per_bucket 条，防止桶无限膨胀。"""
    conn = connect()
    with _lock:
        cur = conn.execute(
            "DELETE FROM repeat_learn WHERE group_id = ? AND id NOT IN ("
            "  SELECT id FROM ("
            "    SELECT id, ROW_NUMBER() OVER ("
            "      PARTITION BY keywords ORDER BY count DESC, ts DESC"
            "    ) AS rn FROM repeat_learn WHERE group_id = ?"
            "  ) WHERE rn <= ?"
            ")",
            (group_id, group_id, max(1, keep_per_bucket)),
        )
        conn.commit()
        return int(cur.rowcount or 0)


def clear_repeat_learn(group_id: str | None = None) -> int:
    conn = connect()
    with _lock:
        cur = (
            conn.execute("DELETE FROM repeat_learn WHERE group_id = ?", (group_id,))
            if group_id is not None
            else conn.execute("DELETE FROM repeat_learn")
        )
        removed = int(cur.rowcount or 0)
        conn.commit()
    return removed


def repeat_learn_stats() -> dict[str, Any]:
    conn = connect()
    with _lock:
        rows = list(
            conn.execute(
                "SELECT group_id, COUNT(*) AS entries, COUNT(DISTINCT keywords) AS buckets, "
                "SUM(count) AS learned FROM repeat_learn GROUP BY group_id"
            )
        )
    return {
        r["group_id"]: {
            "entries": int(r["entries"]),
            "buckets": int(r["buckets"]),
            "learned": int(r["learned"] or 0),
        }
        for r in rows
    }


# --------------------------------------------------------------------------- #
# 风格蒸馏（style_pairs）
# --------------------------------------------------------------------------- #
def replace_style_pairs(group_id: str, rows: list[tuple]) -> int:
    """整体替换某群的接话对。rows: (context, reply, score, source, dim, vec)。"""
    conn = connect()
    now = time.time()
    with _lock:
        conn.execute("DELETE FROM style_pairs WHERE group_id = ?", (group_id,))
        if rows:
            conn.executemany(
                "INSERT INTO style_pairs"
                "(group_id, context, reply, score, source, dim, vec, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [
                    (group_id, ctx, rep, float(score), str(source), int(dim), vec, now)
                    for ctx, rep, score, source, dim, vec in rows
                ],
            )
        conn.commit()
    return len(rows)


def load_style_pairs(group_id: str) -> list[sqlite3.Row]:
    """取出某群全部接话对（含向量，供检索用）。"""
    conn = connect()
    with _lock:
        return list(
            conn.execute(
                "SELECT id, context, reply, dim, vec FROM style_pairs "
                "WHERE group_id = ? ORDER BY id",
                (group_id,),
            )
        )


def list_style_pairs(
    group_id: str, limit: int = 50, offset: int = 0
) -> list[dict[str, Any]]:
    """供面板浏览（不返回向量）。`offset` 用于分页。

    排序固定为 `score DESC, id DESC`：只要不重建，翻页结果就是稳定的，
    不会出现某条记录在两页里重复或整条漏掉。
    """
    conn = connect()
    with _lock:
        rows = list(
            conn.execute(
                "SELECT id, context, reply, score, source, created_at FROM style_pairs "
                "WHERE group_id = ? ORDER BY score DESC, id DESC LIMIT ? OFFSET ?",
                (group_id, max(1, limit), max(0, offset)),
            )
        )
    return [dict(r) for r in rows]


def delete_style_pairs(ids: list[int]) -> int:
    if not ids:
        return 0
    marks = ",".join("?" * len(ids))
    conn = connect()
    with _lock:
        cur = conn.execute(f"DELETE FROM style_pairs WHERE id IN ({marks})", list(ids))
        removed = int(cur.rowcount or 0)
        conn.commit()
    return removed


def clear_style_pairs(group_id: str | None = None) -> int:
    conn = connect()
    with _lock:
        cur = (
            conn.execute("DELETE FROM style_pairs WHERE group_id = ?", (group_id,))
            if group_id is not None
            else conn.execute("DELETE FROM style_pairs")
        )
        removed = int(cur.rowcount or 0)
        conn.commit()
    return removed


def style_pair_stats() -> dict[str, int]:
    conn = connect()
    with _lock:
        rows = list(
            conn.execute(
                "SELECT group_id, COUNT(*) AS pairs FROM style_pairs GROUP BY group_id"
            )
        )
    return {r["group_id"]: int(r["pairs"]) for r in rows}


# --------------------------------------------------------------------------- #
# 表情包库（供 qqbot_core.stickers 与管理面板使用）
# --------------------------------------------------------------------------- #
def add_sticker(
    *,
    sha256: str,
    file_path: str,
    mime: str,
    size: int,
    source: str,
    group_id: str | None = None,
    user_id: str | None = None,
    mface_id: str | None = None,
    summary: str | None = None,
    tags: str = "",
    mood: str | None = None,
    approved: bool = False,
    enabled: bool = True,
    persona: bool = False,
    sort_order: int = 1000,
    weight: float = 1.0,
) -> int | None:
    """新增一张表情；sha256 已存在时只增加 seen_count 并返回 None。"""
    conn = connect()
    with _lock:
        row = conn.execute(
            "SELECT id FROM stickers WHERE sha256 = ?", (sha256,)
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE stickers SET seen_count = seen_count + 1 WHERE id = ?",
                (row["id"],),
            )
            conn.commit()
            return None
        cur = conn.execute(
            "INSERT INTO stickers("
            "sha256, file_path, mime, bytes, source, group_id, user_id, mface_id, "
            "summary, tags, mood, enabled, approved, persona, sort_order, weight, created_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sha256,
                file_path,
                mime,
                int(size),
                source,
                group_id,
                user_id,
                mface_id,
                summary,
                tags,
                mood,
                1 if enabled else 0,
                1 if approved else 0,
                1 if persona else 0,
                int(sort_order),
                float(weight),
                time.time(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid or -1)


def get_sticker(sticker_id: int) -> dict[str, Any] | None:
    conn = connect()
    with _lock:
        row = conn.execute(
            "SELECT * FROM stickers WHERE id = ?", (int(sticker_id),)
        ).fetchone()
    return dict(row) if row is not None else None


def get_sticker_by_sha(sha256: str) -> dict[str, Any] | None:
    conn = connect()
    with _lock:
        row = conn.execute(
            "SELECT * FROM stickers WHERE sha256 = ?", (sha256,)
        ).fetchone()
    return dict(row) if row is not None else None


def list_stickers(
    *,
    page: int = 1,
    page_size: int = 24,
    status: str = "all",
    source: str = "",
    q: str = "",
) -> dict[str, Any]:
    """分页列出表情包。persona 永远排最前，其次按 sort_order/时间。"""
    page = max(1, int(page or 1))
    page_size = min(100, max(1, int(page_size or 24)))
    where: list[str] = []
    params: list[Any] = []

    status = (status or "all").strip()
    if status == "pending":
        where.append("approved = 0")
    elif status == "enabled":
        where.append("enabled = 1")
    elif status == "disabled":
        where.append("enabled = 0")
    elif status == "persona":
        where.append("persona = 1")
    elif status == "learned":
        where.append("source = 'learned'")
    elif status == "upload":
        where.append("source = 'upload'")
    elif status == "archive":
        where.append("source = 'archive'")

    if source.strip():
        where.append("source = ?")
        params.append(source.strip())

    keyword = (q or "").strip()
    if keyword:
        like = f"%{keyword}%"
        where.append(
            "(tags LIKE ? OR mood LIKE ? OR summary LIKE ? OR source LIKE ?)"
        )
        params.extend([like, like, like, like])

    clause = f"WHERE {' AND '.join(where)}" if where else ""
    conn = connect()
    with _lock:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM stickers {clause}", params
        ).fetchone()["c"]
        rows = conn.execute(
            "SELECT * FROM stickers "
            f"{clause} "
            "ORDER BY persona DESC, sort_order ASC, created_at DESC, id DESC "
            "LIMIT ? OFFSET ?",
            (*params, page_size, (page - 1) * page_size),
        ).fetchall()
    return {
        "items": [dict(r) for r in rows],
        "total": int(total),
        "page": page,
        "page_size": page_size,
    }


def update_sticker(sticker_id: int, fields: dict[str, Any]) -> dict[str, Any] | None:
    """更新面板可编辑字段；未知字段忽略。"""
    allowed = {
        "tags",
        "mood",
        "enabled",
        "approved",
        "persona",
        "sort_order",
        "weight",
        "summary",
        "file_path",
        "source",
    }
    sets: list[str] = []
    params: list[Any] = []
    for key, value in fields.items():
        if key not in allowed:
            continue
        if key in {"enabled", "approved", "persona"}:
            value = 1 if value else 0
        elif key == "sort_order":
            value = int(value)
        elif key == "weight":
            value = max(0.0, float(value))
        else:
            value = str(value or "")
        sets.append(f"{key} = ?")
        params.append(value)
    if sets:
        conn = connect()
        with _lock:
            conn.execute(
                f"UPDATE stickers SET {', '.join(sets)} WHERE id = ?",
                (*params, int(sticker_id)),
            )
            conn.commit()
    return get_sticker(sticker_id)


def delete_sticker(sticker_id: int) -> dict[str, Any] | None:
    """删除记录及发送历史；文件由调用方负责删除。返回被删行。"""
    row = get_sticker(sticker_id)
    if row is None:
        return None
    conn = connect()
    with _lock:
        conn.execute("DELETE FROM sticker_usage WHERE sticker_id = ?", (int(sticker_id),))
        conn.execute("DELETE FROM stickers WHERE id = ?", (int(sticker_id),))
        conn.commit()
    return row


def sticker_candidates(mood: str = "", limit: int = 200) -> list[dict[str, Any]]:
    """候选表情：已启用且已审核；mood 命中主情绪或标签。"""
    conn = connect()
    where = ["enabled = 1", "approved = 1"]
    params: list[Any] = []
    mood = (mood or "").strip()
    if mood:
        where.append("(mood = ? OR tags LIKE ?)")
        params.extend([mood, f"%{mood}%"])
    with _lock:
        rows = conn.execute(
            "SELECT * FROM stickers WHERE "
            + " AND ".join(where)
            + " ORDER BY persona DESC, sort_order ASC, id DESC LIMIT ?",
            (*params, max(1, int(limit))),
        ).fetchall()
    return [dict(r) for r in rows]


def record_sticker_use(sticker_id: int, group_id: str | None) -> None:
    conn = connect()
    now = time.time()
    with _lock:
        conn.execute(
            "UPDATE stickers SET use_count = use_count + 1, last_used_at = ? "
            "WHERE id = ?",
            (now, int(sticker_id)),
        )
        conn.execute(
            "INSERT INTO sticker_usage(sticker_id, group_id, ts) VALUES (?,?,?)",
            (int(sticker_id), group_id, now),
        )
        conn.commit()


def sticker_usage_count(group_id: str | None, since: float) -> int:
    conn = connect()
    with _lock:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM sticker_usage "
            "WHERE group_id IS ? AND ts >= ?",
            (group_id, float(since)),
        ).fetchone()
    return int(row["c"])


def recent_sticker_ids(group_id: str | None, since: float) -> set[int]:
    conn = connect()
    with _lock:
        rows = conn.execute(
            "SELECT DISTINCT sticker_id FROM sticker_usage "
            "WHERE group_id IS ? AND ts >= ?",
            (group_id, float(since)),
        ).fetchall()
    return {int(r["sticker_id"]) for r in rows}


def sticker_stats() -> dict[str, int]:
    conn = connect()
    with _lock:
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(persona = 1), 0) AS persona, "
            "COALESCE(SUM(source = 'learned'), 0) AS learned, "
            "COALESCE(SUM(source = 'archive'), 0) AS archive, "
            "COALESCE(SUM(approved = 0), 0) AS pending, "
            "COALESCE(SUM(enabled = 0), 0) AS disabled, "
            "COALESCE(SUM(use_count), 0) AS uses "
            "FROM stickers"
        ).fetchone()
    return {k: int(row[k] or 0) for k in row.keys()}


def persona_sticker_count() -> int:
    conn = connect()
    with _lock:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM stickers WHERE persona = 1"
        ).fetchone()
    return int(row["c"])


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
        learned = conn.execute("SELECT COUNT(*) AS c FROM repeat_learn").fetchone()["c"]
        pairs = conn.execute("SELECT COUNT(*) AS c FROM style_pairs").fetchone()["c"]
        stickers = conn.execute("SELECT COUNT(*) AS c FROM stickers").fetchone()["c"]
    return {
        "messages": int(total),
        "vectors": int(embedded),
        "groups": int(groups),
        "last_message_ts": float(last) if last else None,
        "repeat_entries": int(learned),
        "style_pairs": int(pairs),
        "stickers": int(stickers),
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


# --------------------------------------------------------------------------- #
# 批量导入（历史聊天记录导入用）
# --------------------------------------------------------------------------- #
def groups_of_user(user_id: str) -> list[str]:
    """某用户发过言的群号列表（用于清理其派生数据）。"""
    conn = connect()
    with _lock:
        rows = conn.execute(
            "SELECT DISTINCT group_id FROM messages "
            "WHERE user_id = ? AND group_id IS NOT NULL",
            (user_id,),
        )
        return [str(r["group_id"]) for r in rows]


def message_keys(group_id: str) -> set[tuple[str, str, int]]:
    """某群已有消息的 (user_id, content, 秒级 ts) 集合，供导入去重。"""
    conn = connect()
    with _lock:
        rows = conn.execute(
            "SELECT user_id, content, ts FROM messages WHERE group_id = ?",
            (group_id,),
        )
        return {(r["user_id"], r["content"], round(float(r["ts"]))) for r in rows}


def add_messages_bulk(rows: list[dict[str, Any]]) -> int:
    """批量插入消息；FTS 索引在末尾统一 rebuild（比逐条写快得多）。"""
    if not rows:
        return 0
    payload = [
        (
            r.get("group_id"),
            str(r.get("user_id") or ""),
            r.get("nickname"),
            str(r.get("content") or ""),
            float(r.get("ts") or time.time()),
            1 if r.get("is_bot") else 0,
        )
        for r in rows
    ]
    conn = connect()
    with _lock:
        conn.executemany(
            "INSERT INTO messages(group_id, user_id, nickname, content, ts, is_bot) "
            "VALUES (?,?,?,?,?,?)",
            payload,
        )
        conn.commit()
        _rebuild_fts()
    return len(payload)
