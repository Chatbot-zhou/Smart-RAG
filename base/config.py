import configparser
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    def __init__(self, config_file: str | None = None):
        if load_dotenv:
            load_dotenv(PROJECT_ROOT / ".env")

        self.config = configparser.ConfigParser()
        config_path = Path(config_file) if config_file else PROJECT_ROOT / "config.ini"
        if config_path.exists():
            self.config.read(config_path, encoding="utf-8")

        self.PROJECT_ROOT = PROJECT_ROOT

        self.MYSQL_HOST = self._get("MYSQL_HOST", "mysql", "host", "localhost")
        self.MYSQL_PORT = self._get_int("MYSQL_PORT", "mysql", "port", 3306)
        self.MYSQL_USER = self._get("MYSQL_USER", "mysql", "user", "root")
        self.MYSQL_PASSWORD = self._get("MYSQL_PASSWORD", "mysql", "password", "")
        self.MYSQL_DATABASE = self._get("MYSQL_DATABASE", "mysql", "database", "subjects_kg")

        self.REDIS_HOST = self._get("REDIS_HOST", "redis", "host", "localhost")
        self.REDIS_PORT = self._get_int("REDIS_PORT", "redis", "port", 6379)
        self.REDIS_PASSWORD = self._get("REDIS_PASSWORD", "redis", "password", "")
        self.REDIS_DB = self._get_int("REDIS_DB", "redis", "db", 0)

        self.MILVUS_HOST = self._get("MILVUS_HOST", "milvus", "host", "localhost")
        self.MILVUS_PORT = self._get("MILVUS_PORT", "milvus", "port", "19530")
        self.MILVUS_DATABASE_NAME = self._get("MILVUS_DATABASE_NAME", "milvus", "database_name", "itcast")
        self.MILVUS_COLLECTION_NAME = self._get(
            "MILVUS_COLLECTION_NAME", "milvus", "collection_name", "edurag_final"
        )

        self.LLM_MODEL = self._get("LLM_MODEL", "llm", "model", "qwen-plus")
        self.DASHSCOPE_API_KEY = self._get("DASHSCOPE_API_KEY", "llm", "dashscope_api_key", "")
        self.DASHSCOPE_BASE_URL = self._get(
            "DASHSCOPE_BASE_URL",
            "llm",
            "dashscope_base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        self.PARENT_CHUNK_SIZE = self._get_int("PARENT_CHUNK_SIZE", "retrieval", "parent_chunk_size", 1200)
        self.CHILD_CHUNK_SIZE = self._get_int("CHILD_CHUNK_SIZE", "retrieval", "child_chunk_size", 300)
        self.CHUNK_OVERLAP = self._get_int("CHUNK_OVERLAP", "retrieval", "chunk_overlap", 50)
        self.RETRIEVAL_K = self._get_int("RETRIEVAL_K", "retrieval", "retrieval_k", 10)
        self.CANDIDATE_M = self._get_int("CANDIDATE_M", "retrieval", "candidate_m", 3)

        self.VALID_SOURCES = self._get_list("VALID_SOURCES", "app", "valid_sources", ["ai", "java", "test", "ops", "bigdata"])
        self.CUSTOMER_SERVICE_PHONE = self._get("CUSTOMER_SERVICE_PHONE", "app", "customer_service_phone", "12345678")
        self.LOG_FILE = self._get_path("LOG_FILE", "logger", "log_file", PROJECT_ROOT / "logs" / "app.log")

        self.MODEL_DEVICE = self._get("MODEL_DEVICE", "models", "device", "auto")
        self.EMBEDDING_MODEL = self._get("EMBEDDING_MODEL", "models", "embedding_model", "BAAI/bge-m3")
        self.RERANKER_MODEL = self._get("RERANKER_MODEL", "models", "reranker_model", "BAAI/bge-reranker-large")
        self.BERT_PRETRAINED_MODEL = self._get("BERT_PRETRAINED_MODEL", "models", "bert_pretrained_model", "bert-base-chinese")
        self.QUERY_CLASSIFIER_MODEL = self._get_path(
            "QUERY_CLASSIFIER_MODEL",
            "models",
            "query_classifier_model",
            PROJECT_ROOT / "models" / "bert_query_classifier",
        )
        self.DOCUMENT_SEGMENTATION_MODEL = self._get(
            "DOCUMENT_SEGMENTATION_MODEL",
            "models",
            "document_segmentation_model",
            "damo/nlp_bert_document-segmentation_chinese-base",
        )
        self.DOCUMENT_DATA_DIR = self._get_path("DOCUMENT_DATA_DIR", "data", "document_data_dir", PROJECT_ROOT / "data")
        self.FAQ_CSV_PATH = self._get_path("FAQ_CSV_PATH", "data", "faq_csv_path", PROJECT_ROOT / "data" / "faq.csv")
        self.CLASSIFIER_TRAIN_DATA = self._get_path(
            "CLASSIFIER_TRAIN_DATA",
            "data",
            "classifier_train_data",
            PROJECT_ROOT / "data" / "classifier_train.jsonl",
        )

    def _get(self, env_name: str, section: str, option: str, fallback: str) -> str:
        value = os.getenv(env_name)
        if value is not None:
            return value
        return self.config.get(section, option, fallback=fallback)

    def _get_int(self, env_name: str, section: str, option: str, fallback: int) -> int:
        value = os.getenv(env_name)
        if value is not None:
            return int(value)
        return self.config.getint(section, option, fallback=fallback)

    def _get_path(self, env_name: str, section: str, option: str, fallback: Path) -> str:
        value = self._get(env_name, section, option, str(fallback))
        return str(Path(value).expanduser())

    def _get_list(self, env_name: str, section: str, option: str, fallback: list[str]) -> list[str]:
        raw_value = self._get(env_name, section, option, json.dumps(fallback, ensure_ascii=False))
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    def require_llm_config(self) -> None:
        if not self.DASHSCOPE_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured. Copy .env.example to .env and set it.")


single_config = Config()
