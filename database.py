"""SQLite helpers for storing top-up orders."""

import sqlite3
from contextlib import closing
from typing import Optional

from config import DB_PATH


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_get_connection()) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    username TEXT,
                    game TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    receipt_file_id TEXT,
                    status TEXT NOT NULL DEFAULT 'Pending',
                    g2bulk_order_id TEXT,
                    provider_status TEXT,
                    provider_message TEXT,
                    admin_message_chat_id INTEGER,
                    admin_message_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            _ensure_column(conn, "orders", "g2bulk_order_id", "TEXT")
            _ensure_column(conn, "orders", "provider_status", "TEXT")
            _ensure_column(conn, "orders", "provider_message", "TEXT")


def _ensure_column(
    conn: sqlite3.Connection, table_name: str, column_name: str, column_type: str
) -> None:
    existing = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing_names = {row["name"] for row in existing}
    if column_name not in existing_names:
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )


def create_order(user_id: int, username: Optional[str], game: str, player_id: str) -> int:
    with closing(_get_connection()) as conn:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO orders (user_id, username, game, player_id, status)
                VALUES (?, ?, ?, ?, 'Pending')
                """,
                (user_id, username, game, player_id),
            )
            return int(cursor.lastrowid)


def get_order(order_id: int) -> Optional[sqlite3.Row]:
    with closing(_get_connection()) as conn:
        row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        return row


def set_receipt(order_id: int, receipt_file_id: str) -> None:
    with closing(_get_connection()) as conn:
        with conn:
            conn.execute(
                "UPDATE orders SET receipt_file_id = ? WHERE id = ?",
                (receipt_file_id, order_id),
            )


def set_admin_message(order_id: int, chat_id: int, message_id: int) -> None:
    with closing(_get_connection()) as conn:
        with conn:
            conn.execute(
                """
                UPDATE orders
                SET admin_message_chat_id = ?, admin_message_id = ?
                WHERE id = ?
                """,
                (chat_id, message_id, order_id),
            )


def update_status(order_id: int, status: str) -> None:
    with closing(_get_connection()) as conn:
        with conn:
            conn.execute("UPDATE orders SET status = ? WHERE id = ?", (status, order_id))


def update_provider_result(
    order_id: int,
    provider_status: str,
    provider_message: str = "",
    g2bulk_order_id: Optional[str] = None,
) -> None:
    with closing(_get_connection()) as conn:
        with conn:
            conn.execute(
                """
                UPDATE orders
                SET provider_status = ?, provider_message = ?, g2bulk_order_id = ?
                WHERE id = ?
                """,
                (provider_status, provider_message, g2bulk_order_id, order_id),
            )
