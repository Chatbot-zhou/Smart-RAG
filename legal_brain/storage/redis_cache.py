import json
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

