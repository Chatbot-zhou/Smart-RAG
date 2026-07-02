import hashlib
import re


def normalize_query(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[，。！？；：,.!?;:、\"'“”‘’（）()\[\]【】]", "", normalized)
    return normalized


def query_hash(normalized_query: str) -> str:
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:32]
