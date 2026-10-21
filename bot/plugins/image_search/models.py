"""识图结果的数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Candidate:
    """一个可能的图片来源/账号。"""

    platform: str
    name: str = ""
    handle: str = ""
    url: str = ""
    match: str = "视觉匹配"
    confidence: float = 0.0
    source: str = ""
    is_exact: bool = False
    position: int = 0

    def identity(self) -> str:
        handle = (self.handle or "").lower()
        if handle:
            return f"{(self.platform or '').lower()}|{handle}"
        return f"{(self.platform or '').lower()}|{(self.url or '').split('?', 1)[0].lower()}"

    def score(self) -> float:
        value = max(0.0, min(1.0, float(self.confidence or 0.0))) * 100.0
        if "x (twitter)" in (self.platform or "").lower():
            value += 25.0
        if self.handle:
            value += 15.0
        if self.is_exact:
            value += 12.0
        if self.url:
            value += 5.0
        value -= max(0, int(self.position or 0)) * 0.5
        return value

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    """一次识图的汇总结果。"""

    candidates: list[Candidate] = field(default_factory=list)
    provider: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def as_dict(self, limit: int = 3) -> dict:
        return {
            "provider": self.provider,
            "candidates": [item.as_dict() for item in self.candidates[:limit]],
            "errors": self.errors,
        }
