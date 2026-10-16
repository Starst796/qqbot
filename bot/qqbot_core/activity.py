"""群里「刚刚已经有机器人发言」的轻量标记。

ai_chat 与 repeater 是两套独立的回复路径，同时命中就会出现"一句话两个回复"的
刷屏感。这里用一个内存时间戳做最小协调：谁发过言就登记一次，别的插件在
`quiet_seconds` 内让路。单进程 asyncio，不需要锁。
"""

from __future__ import annotations

import time

_last_spoken: dict[str, float] = {}


def mark_spoken(group_id: str | None) -> None:
    if group_id:
        _last_spoken[str(group_id)] = time.monotonic()


def spoken_recently(group_id: str | None, seconds: float) -> bool:
    if not group_id or seconds <= 0:
        return False
    last = _last_spoken.get(str(group_id))
    return last is not None and time.monotonic() - last < seconds


def forget(group_id: str | None = None) -> None:
    if group_id is None:
        _last_spoken.clear()
    else:
        _last_spoken.pop(str(group_id), None)
