from datetime import datetime
from typing import Any

import pymysql
from pymysql import MySQLError

from legal_brain.config import settings
from legal_brain.logging import logger


class LegalMySQLStore:
    def __init__(self):
        self.connection = pymysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
            charset="utf8mb4",
            autocommit=False,
        )
        self.cursor = self.connection.cursor()
        self.init_schema()

    def init_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS legal_documents (
                id VARCHAR(80) PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                domain VARCHAR(80) NOT NULL,
                authority VARCHAR(255) NOT NULL,
                source_url VARCHAR(1000),
                effective_date VARCHAR(40),
                status VARCHAR(40),
                created_at DATETIME NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS legal_ingestion_batches (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                source_file VARCHAR(1000) NOT NULL,
                document_count INT NOT NULL,
                chunk_count INT NOT NULL,
                created_at DATETIME NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                question TEXT NOT NULL,
                answer MEDIUMTEXT NOT NULL,
                created_at DATETIME NOT NULL,
                INDEX idx_session_created (session_id, created_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS faq_items (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                original_question TEXT NOT NULL,
                normalized_question VARCHAR(1000) NOT NULL,
                answer MEDIUMTEXT NOT NULL,
                category VARCHAR(120),
                source VARCHAR(500),
                source_type VARCHAR(40) NOT NULL DEFAULT 'curated',
                review_status VARCHAR(40) NOT NULL DEFAULT 'approved',
                hit_count INT NOT NULL DEFAULT 0,
                is_hot BOOLEAN NOT NULL DEFAULT FALSE,
                faq_version VARCHAR(80) NOT NULL DEFAULT 'faq-dev',
                corpus_version VARCHAR(80) NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                INDEX idx_faq_normalized (normalized_question(255)),
                INDEX idx_faq_status (review_status, corpus_version)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS qa_records (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64),
                original_query TEXT NOT NULL,
                normalized_query VARCHAR(1000) NOT NULL,
                rewritten_query TEXT,
                answer MEDIUMTEXT NOT NULL,
                answer_type VARCHAR(40) NOT NULL,
                source_type VARCHAR(40) NOT NULL,
                faq_id BIGINT NULL,
                faq_version VARCHAR(80),
                corpus_version VARCHAR(80) NOT NULL,
                disclaimer VARCHAR(500) NOT NULL,
                hit_count INT NOT NULL DEFAULT 0,
                is_hot BOOLEAN NOT NULL DEFAULT FALSE,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                INDEX idx_qa_normalized (normalized_query(255)),
                INDEX idx_qa_hot (is_hot, corpus_version),
                CONSTRAINT fk_qa_faq FOREIGN KEY (faq_id) REFERENCES faq_items(id) ON DELETE SET NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rag_references (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                qa_record_id BIGINT NOT NULL,
                chunk_id VARCHAR(80),
                document_id VARCHAR(80),
                title VARCHAR(255),
                article_no VARCHAR(80),
                source_url VARCHAR(1000),
                retrieval_score DOUBLE,
                rerank_score DOUBLE,
                parent_id VARCHAR(120),
                corpus_version VARCHAR(80) NOT NULL,
                created_at DATETIME NOT NULL,
                INDEX idx_reference_qa (qa_record_id),
                CONSTRAINT fk_reference_qa FOREIGN KEY (qa_record_id) REFERENCES qa_records(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ragas_eval_runs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                run_name VARCHAR(160) NOT NULL,
                dataset_path VARCHAR(1000),
                metrics_json JSON,
                notes TEXT,
                corpus_version VARCHAR(80) NOT NULL,
                created_at DATETIME NOT NULL
            )
            """,
        ]
        for sql in statements:
            self.cursor.execute(sql)
        self.connection.commit()

    def upsert_document(self, metadata: dict[str, Any]) -> None:
        self.cursor.execute(
            """
            INSERT INTO legal_documents (id, title, domain, authority, source_url, effective_date, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                domain = VALUES(domain),
                authority = VALUES(authority),
                source_url = VALUES(source_url),
                effective_date = VALUES(effective_date),
                status = VALUES(status)
            """,
            (
                metadata["id"],
                metadata["title"],
                metadata["domain"],
                metadata.get("authority", ""),
                metadata.get("source_url", ""),
                metadata.get("effective_date", ""),
                metadata.get("status", "effective"),
                datetime.now(),
            ),
        )

    def record_ingestion_batch(self, source_file: str, document_count: int, chunk_count: int) -> None:
        self.cursor.execute(
            """
            INSERT INTO legal_ingestion_batches (source_file, document_count, chunk_count, created_at)
            VALUES (%s, %s, %s, %s)
            """,
            (source_file, document_count, chunk_count, datetime.now()),
        )
        self.connection.commit()

    def recent_history(self, session_id: str, limit: int = 5) -> list[dict[str, str]]:
        self.cursor.execute(
            """
            SELECT question, answer FROM conversations
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (session_id, limit),
        )
        rows = self.cursor.fetchall()
        return [{"question": row[0], "answer": row[1]} for row in rows][::-1]

    def append_history(self, session_id: str, question: str, answer: str) -> None:
        self.cursor.execute(
            "INSERT INTO conversations (session_id, question, answer, created_at) VALUES (%s, %s, %s, %s)",
            (session_id, question, answer, datetime.now()),
        )
        self.connection.commit()

    def upsert_faq_item(self, item: dict[str, Any]) -> int:
        now = datetime.now()
        self.cursor.execute(
            """
            INSERT INTO faq_items (
                original_question, normalized_question, answer, category, source, source_type,
                review_status, faq_version, corpus_version, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                item["original_question"],
                item["normalized_question"],
                item["answer"],
                item.get("category", ""),
                item.get("source", ""),
                item.get("source_type", "curated"),
                item.get("review_status", "approved"),
                item.get("faq_version", "faq-dev"),
                item["corpus_version"],
                now,
                now,
            ),
        )
        self.connection.commit()
        return int(self.cursor.lastrowid)

    def list_active_faq_items(self, corpus_version: str | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT id, original_question, normalized_question, answer, category, source,
                   source_type, review_status, hit_count, is_hot, faq_version, corpus_version
            FROM faq_items
            WHERE review_status = 'approved'
        """
        params: tuple[Any, ...] = ()
        if corpus_version:
            sql += " AND corpus_version = %s"
            params = (corpus_version,)
        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()
        keys = [
            "id",
            "original_question",
            "normalized_question",
            "answer",
            "category",
            "source",
            "source_type",
            "review_status",
            "hit_count",
            "is_hot",
            "faq_version",
            "corpus_version",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def list_faq_items(self, limit: int = 100) -> list[dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT id, original_question, normalized_question, answer, category, source,
                   source_type, review_status, hit_count, is_hot, faq_version, corpus_version, updated_at
            FROM faq_items
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = self.cursor.fetchall()
        keys = [
            "id",
            "original_question",
            "normalized_question",
            "answer",
            "category",
            "source",
            "source_type",
            "review_status",
            "hit_count",
            "is_hot",
            "faq_version",
            "corpus_version",
            "updated_at",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def get_faq_item(self, faq_id: int) -> dict[str, Any] | None:
        self.cursor.execute(
            """
            SELECT id, original_question, normalized_question, answer, category, source,
                   source_type, review_status, hit_count, is_hot, faq_version, corpus_version
            FROM faq_items
            WHERE id = %s
            """,
            (faq_id,),
        )
        row = self.cursor.fetchone()
        if not row:
            return None
        keys = [
            "id",
            "original_question",
            "normalized_question",
            "answer",
            "category",
            "source",
            "source_type",
            "review_status",
            "hit_count",
            "is_hot",
            "faq_version",
            "corpus_version",
        ]
        return dict(zip(keys, row))

    def increment_faq_hit(self, faq_id: int) -> int:
        self.cursor.execute(
            "UPDATE faq_items SET hit_count = hit_count + 1, updated_at = %s WHERE id = %s",
            (datetime.now(), faq_id),
        )
        self.connection.commit()
        self.cursor.execute("SELECT hit_count FROM faq_items WHERE id = %s", (faq_id,))
        row = self.cursor.fetchone()
        return int(row[0]) if row else 0

    def mark_faq_hot(self, faq_id: int) -> None:
        self.cursor.execute("UPDATE faq_items SET is_hot = TRUE, updated_at = %s WHERE id = %s", (datetime.now(), faq_id))
        self.connection.commit()

    def insert_qa_record(self, record: dict[str, Any]) -> int:
        now = datetime.now()
        self.cursor.execute(
            """
            INSERT INTO qa_records (
                session_id, original_query, normalized_query, rewritten_query, answer,
                answer_type, source_type, faq_id, faq_version, corpus_version,
                disclaimer, hit_count, is_hot, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.get("session_id"),
                record["original_query"],
                record["normalized_query"],
                record.get("rewritten_query", ""),
                record["answer"],
                record["answer_type"],
                record["source_type"],
                record.get("faq_id"),
                record.get("faq_version"),
                record["corpus_version"],
                record["disclaimer"],
                record.get("hit_count", 0),
                record.get("is_hot", False),
                now,
                now,
            ),
        )
        self.connection.commit()
        return int(self.cursor.lastrowid)

    def find_qa_by_normalized_query(self, normalized_query: str, corpus_version: str) -> dict[str, Any] | None:
        self.cursor.execute(
            """
            SELECT id, original_query, normalized_query, rewritten_query, answer, answer_type,
                   source_type, faq_id, faq_version, corpus_version, disclaimer, hit_count, is_hot
            FROM qa_records
            WHERE normalized_query = %s AND corpus_version = %s
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (normalized_query, corpus_version),
        )
        row = self.cursor.fetchone()
        if not row:
            return None
        keys = [
            "id",
            "original_query",
            "normalized_query",
            "rewritten_query",
            "answer",
            "answer_type",
            "source_type",
            "faq_id",
            "faq_version",
            "corpus_version",
            "disclaimer",
            "hit_count",
            "is_hot",
        ]
        return dict(zip(keys, row))

    def get_qa_record(self, qa_id: int) -> dict[str, Any] | None:
        self.cursor.execute(
            """
            SELECT id, original_query, normalized_query, rewritten_query, answer, answer_type,
                   source_type, faq_id, faq_version, corpus_version, disclaimer, hit_count, is_hot
            FROM qa_records
            WHERE id = %s
            """,
            (qa_id,),
        )
        row = self.cursor.fetchone()
        if not row:
            return None
        keys = [
            "id",
            "original_query",
            "normalized_query",
            "rewritten_query",
            "answer",
            "answer_type",
            "source_type",
            "faq_id",
            "faq_version",
            "corpus_version",
            "disclaimer",
            "hit_count",
            "is_hot",
        ]
        return dict(zip(keys, row))

    def increment_qa_hit(self, qa_id: int) -> int:
        self.cursor.execute(
            "UPDATE qa_records SET hit_count = hit_count + 1, updated_at = %s WHERE id = %s",
            (datetime.now(), qa_id),
        )
        self.connection.commit()
        self.cursor.execute("SELECT hit_count FROM qa_records WHERE id = %s", (qa_id,))
        row = self.cursor.fetchone()
        return int(row[0]) if row else 0

    def mark_qa_hot(self, qa_id: int) -> None:
        self.cursor.execute("UPDATE qa_records SET is_hot = TRUE, updated_at = %s WHERE id = %s", (datetime.now(), qa_id))
        self.connection.commit()

    def insert_rag_references(self, qa_record_id: int, references: list[dict[str, Any]], corpus_version: str) -> None:
        now = datetime.now()
        for reference in references:
            self.cursor.execute(
                """
                INSERT INTO rag_references (
                    qa_record_id, chunk_id, document_id, title, article_no, source_url,
                    retrieval_score, rerank_score, parent_id, corpus_version, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    qa_record_id,
                    reference.get("chunk_id"),
                    reference.get("document_id"),
                    reference.get("title"),
                    reference.get("article_no"),
                    reference.get("source_url"),
                    reference.get("retrieval_score"),
                    reference.get("rerank_score"),
                    reference.get("parent_id"),
                    corpus_version,
                    now,
                ),
            )
        self.connection.commit()

    def list_qa_records_by_session(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT id, original_query, normalized_query, rewritten_query, answer, answer_type,
                   source_type, faq_id, faq_version, corpus_version, disclaimer, hit_count, is_hot, created_at
            FROM qa_records
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (session_id, limit),
        )
        rows = self.cursor.fetchall()
        keys = [
            "id",
            "original_query",
            "normalized_query",
            "rewritten_query",
            "answer",
            "answer_type",
            "source_type",
            "faq_id",
            "faq_version",
            "corpus_version",
            "disclaimer",
            "hit_count",
            "is_hot",
            "created_at",
        ]
        records = [dict(zip(keys, row)) for row in rows]
        for record in records:
            record["references"] = self.list_rag_references(int(record["id"]))
        return records

    def list_rag_references(self, qa_record_id: int) -> list[dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT chunk_id, document_id, title, article_no, source_url,
                   retrieval_score, rerank_score, parent_id, corpus_version
            FROM rag_references
            WHERE qa_record_id = %s
            ORDER BY id ASC
            """,
            (qa_record_id,),
        )
        rows = self.cursor.fetchall()
        keys = [
            "chunk_id",
            "document_id",
            "title",
            "article_no",
            "source_url",
            "retrieval_score",
            "rerank_score",
            "parent_id",
            "corpus_version",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def record_ragas_eval_run(self, run: dict[str, Any]) -> int:
        import json

        self.cursor.execute(
            """
            INSERT INTO ragas_eval_runs (run_name, dataset_path, metrics_json, notes, corpus_version, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                run["run_name"],
                run.get("dataset_path", ""),
                json.dumps(run.get("metrics", {}), ensure_ascii=False),
                run.get("notes", ""),
                run["corpus_version"],
                datetime.now(),
            ),
        )
        self.connection.commit()
        return int(self.cursor.lastrowid)

    def list_ragas_eval_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT id, run_name, dataset_path, metrics_json, notes, corpus_version, created_at
            FROM ragas_eval_runs
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = self.cursor.fetchall()
        keys = ["id", "run_name", "dataset_path", "metrics_json", "notes", "corpus_version", "created_at"]
        return [dict(zip(keys, row)) for row in rows]

    def legal_document_summary(self) -> dict[str, Any]:
        self.cursor.execute("SELECT COUNT(*) FROM legal_documents")
        document_count = int(self.cursor.fetchone()[0])
        self.cursor.execute("SELECT COUNT(*) FROM legal_ingestion_batches")
        batch_count = int(self.cursor.fetchone()[0])
        self.cursor.execute(
            "SELECT id, title, domain, authority, source_url, effective_date, status FROM legal_documents ORDER BY created_at DESC LIMIT 50"
        )
        rows = self.cursor.fetchall()
        keys = ["id", "title", "domain", "authority", "source_url", "effective_date", "status"]
        return {
            "document_count": document_count,
            "batch_count": batch_count,
            "documents": [dict(zip(keys, row)) for row in rows],
        }

    def clear_history(self, session_id: str) -> bool:
        try:
            self.cursor.execute("DELETE FROM conversations WHERE session_id = %s", (session_id,))
            self.connection.commit()
            return True
        except MySQLError as exc:
            logger.error(f"清理会话历史失败: {exc}")
            self.connection.rollback()
            return False

    def close(self) -> None:
        self.cursor.close()
        self.connection.close()
