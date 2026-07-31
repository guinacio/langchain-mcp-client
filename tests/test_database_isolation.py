"""Tests for per-owner persistent conversation isolation."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from src.database import PersistentStorageManager


class PersistentStorageIsolationTests(unittest.TestCase):
    def test_same_thread_id_is_isolated_between_owners(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "conversations.db"
            alice = PersistentStorageManager("alice", str(db_path))
            bob = PersistentStorageManager("bob", str(db_path))

            alice.save_conversation_sync("default", [
                {"role": "user", "content": "alice secret", "timestamp": "now"}
            ])
            self.assertEqual([], bob.list_conversations())
            self.assertEqual([], bob.load_conversation_messages("default"))

            bob.save_conversation_sync("default", [
                {"role": "user", "content": "bob secret", "timestamp": "now"}
            ])
            self.assertEqual(
                "alice secret",
                alice.load_conversation_messages("default")[0]["content"],
            )
            self.assertEqual(
                "bob secret",
                bob.load_conversation_messages("default")[0]["content"],
            )
            self.assertNotEqual(
                alice.scoped_thread_id("default"),
                bob.scoped_thread_id("default"),
            )

    def test_legacy_global_rows_are_quarantined(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "legacy.db"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.executescript("""
                    CREATE TABLE conversation_metadata (
                        thread_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        title TEXT,
                        message_count INTEGER DEFAULT 0,
                        last_message TEXT
                    );
                    CREATE TABLE conversation_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        message_id TEXT,
                        metadata TEXT
                    );
                    INSERT INTO conversation_metadata
                        (thread_id, title, message_count) VALUES ('default', 'legacy', 1);
                    INSERT INTO conversation_messages
                        (thread_id, role, content, timestamp)
                        VALUES ('default', 'user', 'legacy secret', 'now');
                """)

            current = PersistentStorageManager("new-session", str(db_path))
            self.assertEqual([], current.list_conversations())
            self.assertEqual([], current.load_conversation_messages("default"))

            with closing(sqlite3.connect(db_path)) as conn:
                owner = conn.execute(
                    "SELECT owner_id FROM conversation_metadata WHERE thread_id = 'default'"
                ).fetchone()[0]
            self.assertEqual("legacy-unscoped", owner)


if __name__ == "__main__":
    unittest.main()
