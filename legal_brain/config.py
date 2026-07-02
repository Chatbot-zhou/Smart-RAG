import configparser
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings:
    def __init__(self, config_file: str | None = None):
        if load_dotenv:
            load_dotenv(PROJECT_ROOT / ".env")

        parser = configparser.ConfigParser()
        config_path = Path(config_file) if config_file else PROJECT_ROOT / "config.ini"
        if config_path.exists():
            parser.read(config_path, encoding="utf-8")

        self.project_root = PROJECT_ROOT
        self.app_name = self._get(parser, "APP_NAME", "app", "name", "智慧法务大脑")
        self.service_disclaimer = self._get(
            parser,
            "LEGAL_DISCLAIMER",
            "app",
            "legal_disclaimer",
            "仅供业务参考，不构成正式法律意见。",
        )

        self.mysql_host = self._get(parser, "MYSQL_HOST", "mysql", "host", "localhost")
        self.mysql_port = self._get_int(parser, "MYSQL_PORT", "mysql", "port", 3307)
        self.mysql_user = self._get(parser, "MYSQL_USER", "mysql", "user", "legal_user")
        self.mysql_password = self._get(parser, "MYSQL_PASSWORD", "mysql", "password", "change_me_user")
        self.mysql_database = self._get(parser, "MYSQL_DATABASE", "mysql", "database", "legal_brain")

        self.redis_host = self._get(parser, "REDIS_HOST", "redis", "host", "localhost")
        self.redis_port = self._get_int(parser, "REDIS_PORT", "redis", "port", 6380)
        self.redis_password = self._get(parser, "REDIS_PASSWORD", "redis", "password", "")
        self.redis_db = self._get_int(parser, "REDIS_DB", "redis", "db", 0)

        self.milvus_host = self._get(parser, "MILVUS_HOST", "milvus", "host", "localhost")
        self.milvus_port = self._get(parser, "MILVUS_PORT", "milvus", "port", "19531")
        self.milvus_database_name = self._get(parser, "MILVUS_DATABASE_NAME", "milvus", "database_name", "legal_brain")
        self.milvus_collection_name = self._get(
            parser, "MILVUS_COLLECTION_NAME", "milvus", "collection_name", "legal_brain_chunks"
        )

        self.dashscope_api_key = self._get(parser, "DASHSCOPE_API_KEY", "llm", "dashscope_api_key", "")
        self.dashscope_base_url = self._get(
            parser,
            "DASHSCOPE_BASE_URL",
            "llm",
            "dashscope_base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.llm_model = self._get(parser, "LLM_MODEL", "llm", "model", "qwen-plus")

        self.embedding_model = self._get(parser, "EMBEDDING_MODEL", "models", "embedding_model", "BAAI/bge-m3")
        self.reranker_model = self._get(
            parser, "RERANKER_MODEL", "models", "reranker_model", "BAAI/bge-reranker-large"
        )
        self.bert_pretrained_model = self._get(
            parser, "BERT_PRETRAINED_MODEL", "models", "bert_pretrained_model", "bert-base-chinese"
        )
        self.query_classifier_model = self._path(
            self._get(
                parser,
                "QUERY_CLASSIFIER_MODEL",
                "models",
                "query_classifier_model",
                str(PROJECT_ROOT / "models" / "legal_intent_classifier"),
            )
        )
        self.intent_labels = self._get_list(
            parser,
            "INTENT_LABELS",
            "models",
            "intent_labels",
            ["legal_qa", "regulation_query", "risk_identification", "contract_review", "contract_drafting"],
        )
        self.model_device = self._get(parser, "MODEL_DEVICE", "models", "device", "auto")

        self.legal_seed_file = self._path(
            self._get(
                parser,
                "LEGAL_SEED_FILE",
                "data",
                "legal_seed_file",
                str(PROJECT_ROOT / "data_sources" / "legal_seed.yml"),
            )
        )
        self.legal_source_file = self._path(
            self._get(
                parser,
                "LEGAL_SOURCE_FILE",
                "data",
                "legal_source_file",
                str(PROJECT_ROOT / "data_sources" / "official_legal_sources.yml"),
            )
        )
        self.legal_corpus_dir = self._path(
            self._get(parser, "LEGAL_CORPUS_DIR", "data", "legal_corpus_dir", str(PROJECT_ROOT / "data" / "legal_corpus"))
        )
        self.chunk_output_file = self._path(
            self._get(parser, "LEGAL_CHUNKS_FILE", "data", "legal_chunks_file", str(PROJECT_ROOT / "data" / "legal_chunks.jsonl"))
        )
        self.parent_chunk_size = self._get_int(parser, "PARENT_CHUNK_SIZE", "retrieval", "parent_chunk_size", 1200)
        self.child_chunk_size = self._get_int(parser, "CHILD_CHUNK_SIZE", "retrieval", "child_chunk_size", 320)
        self.chunk_overlap = self._get_int(parser, "CHUNK_OVERLAP", "retrieval", "chunk_overlap", 80)
        self.retrieval_k = self._get_int(parser, "RETRIEVAL_K", "retrieval", "retrieval_k", 10)
        self.candidate_m = self._get_int(parser, "CANDIDATE_M", "retrieval", "candidate_m", 2)
        self.rerank_top_n = self._get_int(parser, "RERANK_TOP_N", "retrieval", "rerank_top_n", 2)
        self.faq_match_threshold = self._get_float(parser, "FAQ_MATCH_THRESHOLD", "faq", "match_threshold", 0.85)
        self.faq_high_confidence_threshold = self._get_float(
            parser, "FAQ_HIGH_CONFIDENCE_THRESHOLD", "faq", "high_confidence_threshold", 0.9
        )
        self.faq_bm25_weight = self._get_float(parser, "FAQ_BM25_WEIGHT", "faq", "bm25_weight", 0.3)
        self.faq_embedding_weight = self._get_float(parser, "FAQ_EMBEDDING_WEIGHT", "faq", "embedding_weight", 0.7)
        self.hot_question_ttl_seconds = self._get_int(parser, "HOT_QUESTION_TTL_SECONDS", "faq", "hot_question_ttl_seconds", 604800)
        self.hot_question_promote_hits = self._get_int(parser, "HOT_QUESTION_PROMOTE_HITS", "faq", "hot_question_promote_hits", 3)
        self.corpus_version = self._get(parser, "CORPUS_VERSION", "data", "corpus_version", "legal-corpus-dev")
        self.legal_domains = self._get_list(
            parser,
            "LEGAL_DOMAINS",
            "app",
            "legal_domains",
            ["civil", "commercial", "labor", "contract", "corporate", "dispute_resolution"],
        )
        self.log_file = self._path(
            self._get(parser, "LOG_FILE", "logger", "log_file", str(PROJECT_ROOT / "logs" / "legal_brain.log"))
        )

    def _get(self, parser: configparser.ConfigParser, env_name: str, section: str, option: str, fallback: str) -> str:
        env_value = os.getenv(env_name)
        if env_value is not None:
            return env_value
        return parser.get(section, option, fallback=fallback)

    def _get_int(self, parser: configparser.ConfigParser, env_name: str, section: str, option: str, fallback: int) -> int:
        env_value = os.getenv(env_name)
        if env_value is not None:
            return int(env_value)
        return parser.getint(section, option, fallback=fallback)

    def _get_float(
        self, parser: configparser.ConfigParser, env_name: str, section: str, option: str, fallback: float
    ) -> float:
        env_value = os.getenv(env_name)
        if env_value is not None:
            return float(env_value)
        return parser.getfloat(section, option, fallback=fallback)

    def _get_list(
        self, parser: configparser.ConfigParser, env_name: str, section: str, option: str, fallback: list[str]
    ) -> list[str]:
        raw_value = self._get(parser, env_name, section, option, json.dumps(fallback, ensure_ascii=False))
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    def _path(self, value: str) -> Path:
        return Path(value).expanduser()

    def require_llm(self) -> None:
        if not self.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured. Copy .env.example to .env and set it.")


settings = Settings()
