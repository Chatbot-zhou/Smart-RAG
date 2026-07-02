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

