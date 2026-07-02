import json
from datetime import datetime, timezone
from typing import Any

import redis

from legal_brain.config import settings


class LegalRedisCache:
    def __init__(self):
        self.client = redis.StrictRedis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            decode_responses=True,
        )

    def get_json(self, key: str) -> Any | None:
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set_json(self, key: str, value: Any, ttl_seconds: int = 86400) -> None:
        self.client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_seconds)

    def hot_question_key(self, normalized_hash: str) -> str:
        return f"hotq:{normalized_hash}"

    def get_hot_question(self, normalized_hash: str) -> dict[str, Any] | None:
        return self.get_json(self.hot_question_key(normalized_hash))

    def set_hot_question(self, normalized_hash: str, payload: dict[str, Any]) -> None:
        value = {
            **payload,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        self.set_json(self.hot_question_key(normalized_hash), value, ttl_seconds=settings.hot_question_ttl_seconds)

    def refresh_hot_question(self, normalized_hash: str, payload: dict[str, Any]) -> None:
        self.set_hot_question(normalized_hash, payload)

    def delete_hot_question(self, normalized_hash: str) -> None:
        self.client.delete(self.hot_question_key(normalized_hash))
