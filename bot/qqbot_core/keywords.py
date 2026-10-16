"""轻量关键词抽取：给「学习型复读」的关键词桶当分桶键。

Pallas-Bot 用 `jieba.analyse.extract_tags(topK=2)` 取词。本项目不引入 jieba
（带词典、吃内存，而本机同时装着 bge-m3），这里用「CJK 二元组 + 拉丁词」近似：
先按频次排序，再贪心挑出**在原文里互不重叠**的若干块（按位置判定，而不是按字符
去重），这样「今天天气不错」得到「今天/天气」而不是「今天/气不」。只当分桶键用，
不追求语言学正确——Pallas 那边同样是「topK 个词拼成精确匹配的键」。
"""

from __future__ import annotations

import re
from collections import Counter

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[a-zA-Z0-9_]+")


def _spans(text: str) -> list[tuple[str, int, int]]:
    """返回 (词, 起点, 终点)；CJK 按二元组切，拉丁按整词切。"""
    spans: list[tuple[str, int, int]] = []
    for match in _CJK.finditer(text):
        run = match.group()
        start = match.start()
        if len(run) == 1:
            spans.append((run, start, start + 1))
        else:
            spans.extend(
                (run[i : i + 2], start + i, start + i + 2) for i in range(len(run) - 1)
            )
    spans.extend(
        (match.group(), match.start(), match.end()) for match in _LATIN.finditer(text)
    )
    return spans


def _overlaps(start: int, end: int, taken: list[tuple[int, int]]) -> bool:
    return any(start < other_end and other_start < end for other_start, other_end in taken)


def extract_keywords(text: str, top_k: int = 2) -> str:
    """返回空格连接的关键词（分桶键）；没有可用词时返回空串。"""
    normalized = (text or "").strip().lower()
    if not normalized:
        return ""
    spans = _spans(normalized)
    if not spans:
        return ""

    counts = Counter(token for token, _start, _end in spans)
    ranked = sorted(spans, key=lambda item: (-counts[item[0]], item[1]))

    picked: list[str] = []
    taken: list[tuple[int, int]] = []
    for token, start, end in ranked:
        if _overlaps(start, end, taken):
            continue
        picked.append(token)
        taken.append((start, end))
        if len(picked) >= max(1, top_k):
            break
    return " ".join(picked or [ranked[0][0]])
