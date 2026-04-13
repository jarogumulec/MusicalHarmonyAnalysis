from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Iterable, Iterator
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path


_BRACKETED_SUFFIX_RE = re.compile(r"\s*[\[(].*?[\])]\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def strip_accents(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(value: str | None) -> str:
    cleaned = strip_accents(value).lower().strip()
    cleaned = _NON_ALNUM_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def normalize_title(value: str | None) -> str:
    if not value:
        return ""
    stripped = _BRACKETED_SUFFIX_RE.sub("", value).strip()
    return normalize_text(stripped)


def similarity(left: str | None, right: str | None) -> float:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    if not normalized_left and not normalized_right:
        return 1.0
    if not normalized_left or not normalized_right:
        return 0.0
    return SequenceMatcher(a=normalized_left, b=normalized_right).ratio()


def chunked(items: Iterable[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def isoformat_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def load_dotenv(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded