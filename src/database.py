"""
Database and storage management for the LangChain MCP Client.

This module handles persistent conversation storage using SQLite
and provides interfaces for managing conversation metadata.
"""

import sqlite3
import threading
import datetime
import json
from contextlib import closing
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class PersistentStorageManager:
    """
    Manages SQLite database for persistent conversation storage.
    Uses a simple manual save/load approach to avoid async context issues.
    """
    
    def __init__(self, owner_id: str, db_path: str = "conversations.db"):
        if not owner_id or not isinstance(owner_id, str):
            raise ValueError("A server-issued storage owner ID is required")
        self.owner_id = owner_id
        self.db_path = Path(db_path)
        self.lock = threading.Lock()
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Create database and tables if they don't exist."""
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Migrate legacy globally-scoped tables before creating the
                # owner-scoped schema. Legacy rows are quarantined and are not
                # visible to any live browser session.
                existing_columns = {
                    row[1] for row in cursor.execute(
                        "PRAGMA table_info(conversation_metadata)"
                    ).fetchall()
                }
                if existing_columns and "owner_id" not in existing_columns:
                    cursor.execute("PRAGMA foreign_keys = OFF")
                    cursor.execute("""
                        CREATE TABLE conversation_metadata_scoped (
                            owner_id TEXT NOT NULL,
                            thread_id TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            title TEXT,
                            message_count INTEGER DEFAULT 0,
                            last_message TEXT,
                            PRIMARY KEY (owner_id, thread_id)
                        )
                    """)
                    cursor.execute("""
                        INSERT INTO conversation_metadata_scoped
                        (owner_id, thread_id, created_at, updated_at, title, message_count, last_message)
                        SELECT 'legacy-unscoped', thread_id, created_at, updated_at, title,
                               message_count, last_message
                        FROM conversation_metadata
                    """)
                    cursor.execute("""
                        CREATE TABLE conversation_messages_scoped (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            owner_id TEXT NOT NULL,
                            thread_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            content TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            message_id TEXT,
                            metadata TEXT,
                            FOREIGN KEY (owner_id, thread_id)
                                REFERENCES conversation_metadata_scoped (owner_id, thread_id)
                                ON DELETE CASCADE
                        )
                    """)
                    cursor.execute("""
                        INSERT INTO conversation_messages_scoped
                        (id, owner_id, thread_id, role, content, timestamp, message_id, metadata)
                        SELECT id, 'legacy-unscoped', thread_id, role, content, timestamp,
                               message_id, metadata
                        FROM conversation_messages
                    """)
                    cursor.execute("DROP TABLE conversation_messages")
                    cursor.execute("DROP TABLE conversation_metadata")
                    cursor.execute(
                        "ALTER TABLE conversation_metadata_scoped RENAME TO conversation_metadata"
                    )
                    cursor.execute(
                        "ALTER TABLE conversation_messages_scoped RENAME TO conversation_messages"
                    )
                    conn.commit()
                    cursor.execute("PRAGMA foreign_keys = ON")

                # Create owner-scoped conversations metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_metadata (
                        owner_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        title TEXT,
                        message_count INTEGER DEFAULT 0,
                        last_message TEXT,
                        PRIMARY KEY (owner_id, thread_id)
                    )
                """)
                
                # Create conversation messages table for manual persistence
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        owner_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        message_id TEXT,
                        metadata TEXT,
                        FOREIGN KEY (owner_id, thread_id)
                            REFERENCES conversation_metadata (owner_id, thread_id)
                            ON DELETE CASCADE
                    )
                """)
                
                # Create index for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversation_messages_owner_thread
                    ON conversation_messages (owner_id, thread_id)
                """)
                
                conn.commit()
        except Exception as e:
            st.error(f"Error initializing database: {str(e)}")
            raise

    def _upsert_conversation_metadata(
        self,
        cursor: sqlite3.Cursor,
        thread_id: str,
        title: str = None,
        message_count: int = None,
        last_message: str = None,
    ) -> None:
        """Create or update metadata without REPLACE/delete semantics."""
        cursor.execute("""
            INSERT INTO conversation_metadata
                (owner_id, thread_id, created_at, updated_at, title, message_count, last_message)
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, COALESCE(?, 0), ?)
            ON CONFLICT(owner_id, thread_id) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP,
                title = COALESCE(excluded.title, conversation_metadata.title),
                message_count = COALESCE(?, conversation_metadata.message_count),
                last_message = COALESCE(excluded.last_message, conversation_metadata.last_message)
        """, (
            self.owner_id,
            thread_id,
            title,
            message_count,
            last_message,
            message_count,
        ))
    
    def get_checkpointer_sync(self):
        """
        Return a SqliteSaver-based checkpointer backed by this manager's DB.
        Falls back to InMemorySaver if SQLite setup fails.
        """
        try:
            # Use a dedicated SQLite connection for LangGraph checkpointer
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            return SqliteSaver(conn)
        except Exception:
            return InMemorySaver()

    def scoped_thread_id(self, thread_id: str) -> str:
        """Return the opaque LangGraph thread ID for this storage owner."""
        return f"{self.owner_id}:{thread_id}"

    async def get_async_checkpointer(self):
        """
        Create and return an AsyncSqliteSaver for async graph operations.
        Returns None on failure; caller should handle fallback.
        """
        try:
            return await AsyncSqliteSaver.from_conn_string(str(self.db_path))
        except Exception:
            return None
    
    def save_conversation_messages(self, thread_id: str, chat_history: List[Dict]):
        """
        Save conversation messages to SQLite database.
        This replaces the complex LangGraph checkpointer approach.
        """
        try:
            with self.lock:
                with closing(sqlite3.connect(self.db_path)) as conn:
                    cursor = conn.cursor()

                    self._upsert_conversation_metadata(
                        cursor,
                        thread_id=thread_id,
                        title=self._generate_title(chat_history),
                        message_count=len(chat_history),
                        last_message=self._get_last_message(chat_history),
                    )
                    
                    # Clear existing messages for this thread
                    cursor.execute(
                        "DELETE FROM conversation_messages WHERE owner_id = ? AND thread_id = ?",
                        (self.owner_id, thread_id),
                    )
                    
                    # Insert all messages
                    for msg in chat_history:
                        metadata_json = json.dumps({
                            'model_provider': msg.get('model_provider'),
                            'model_name': msg.get('model_name'),
                            'response_time': msg.get('response_time'),
                            'thinking': msg.get('thinking'),
                            'tool_executions': msg.get('tool_executions')
                        })
                        
                        cursor.execute("""
                            INSERT INTO conversation_messages 
                            (owner_id, thread_id, role, content, timestamp, message_id, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            self.owner_id,
                            thread_id,
                            msg.get('role', ''),
                            msg.get('content', ''),
                            msg.get('timestamp', ''),
                            msg.get('message_id', ''),
                            metadata_json
                        ))
                    
                    conn.commit()
                    
        except Exception as e:
            # Don't show error to user for auto-save failures
            pass
    
    def load_conversation_messages(self, thread_id: str, limit: int = 100) -> List[Dict]:
        """
        Load conversation messages from SQLite database.
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT role, content, timestamp, message_id, metadata
                    FROM conversation_messages
                    WHERE owner_id = ? AND thread_id = ?
                    ORDER BY id ASC
                    LIMIT ?
                """, (self.owner_id, thread_id, limit))
                
                messages = []
                for row in cursor.fetchall():
                    # Parse metadata
                    metadata = {}
                    try:
                        if row['metadata']:
                            metadata = json.loads(row['metadata'])
                    except (json.JSONDecodeError, TypeError):
                        pass
                    
                    # Create message dict
                    msg = {
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': row['timestamp'],
                        'message_id': row['message_id']
                    }
                    
                    # Add metadata fields if they exist
                    for key in ['model_provider', 'model_name', 'response_time', 'thinking', 'tool_executions']:
                        if key in metadata and metadata[key]:
                            msg[key] = metadata[key]
                    
                    messages.append(msg)
                
                return messages
                
        except Exception as e:
            st.warning(f"Could not load conversation messages: {str(e)}")
            return []
    
    def _generate_title(self, chat_history: List[Dict]) -> str:
        """Generate a title from the first user message."""
        for msg in chat_history[:3]:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                return content[:50] + "..." if len(content) > 50 else content
        return "Untitled Conversation"
    
    def _get_last_message(self, chat_history: List[Dict]) -> str:
        """Get the last message content for metadata."""
        if chat_history:
            last_msg = chat_history[-1].get('content', '')
            return last_msg[:100] + "..." if len(last_msg) > 100 else last_msg
        return ""
    
    def save_conversation_sync(self, thread_id: str, chat_history: List[Dict]):
        """
        Synchronously save conversation after agent response.
        This is the main method called after each interaction.
        """
        if chat_history:
            self.save_conversation_messages(thread_id, chat_history)
    
    def list_conversations(self) -> List[Dict]:
        """List all stored conversations."""
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT thread_id, created_at, updated_at, title, message_count, last_message
                    FROM conversation_metadata
                    WHERE owner_id = ?
                    ORDER BY updated_at DESC
                """, (self.owner_id,))
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            st.error(f"Error listing conversations: {str(e)}")
            return []
    
    def update_conversation_metadata(self, thread_id: str, title: str = None, 
                                   message_count: int = None, last_message: str = None):
        """Update conversation metadata."""
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()
                
                self._upsert_conversation_metadata(
                    cursor,
                    thread_id=thread_id,
                    title=title,
                    message_count=message_count,
                    last_message=last_message,
                )
                
                conn.commit()
        except Exception as e:
            st.error(f"Error updating conversation metadata: {str(e)}")
    
    def delete_conversation(self, thread_id: str):
        """Delete a conversation and its metadata."""
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Delete messages first
                cursor.execute(
                    "DELETE FROM conversation_messages WHERE owner_id = ? AND thread_id = ?",
                    (self.owner_id, thread_id),
                )
                
                # Delete metadata
                cursor.execute(
                    "DELETE FROM conversation_metadata WHERE owner_id = ? AND thread_id = ?",
                    (self.owner_id, thread_id),
                )
                
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error deleting conversation: {str(e)}")
            return False
    
    def export_conversation(self, thread_id: str) -> Dict:
        """Export a conversation's data."""
        try:
            # Get metadata
            metadata = None
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT thread_id, created_at, updated_at, title, message_count, last_message
                       FROM conversation_metadata
                       WHERE owner_id = ? AND thread_id = ?""",
                    (self.owner_id, thread_id),
                )
                row = cursor.fetchone()
                if row:
                    metadata = dict(row)
            
            # Get messages
            messages = self.load_conversation_messages(thread_id, limit=1000)
            
            return {
                'thread_id': thread_id,
                'metadata': metadata,
                'messages': messages,
                'export_timestamp': datetime.datetime.now().isoformat(),
                'format_version': '1.1'
            }
        except Exception as e:
            st.error(f"Error exporting conversation: {str(e)}")
            return {}
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Get conversation count
                cursor.execute(
                    "SELECT COUNT(*) FROM conversation_metadata WHERE owner_id = ?",
                    (self.owner_id,),
                )
                conversation_count = cursor.fetchone()[0]
                
                # Get total messages
                cursor.execute(
                    "SELECT SUM(message_count) FROM conversation_metadata WHERE owner_id = ?",
                    (self.owner_id,),
                )
                total_messages = cursor.fetchone()[0] or 0
                
                # Get database file size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                return {
                    'conversation_count': conversation_count,
                    'total_messages': total_messages,
                    'database_size_bytes': db_size,
                    'database_size_mb': round(db_size / (1024 * 1024), 2),
                    'database_path': str(self.db_path)
                }
        except Exception as e:
            st.error(f"Error getting database stats: {str(e)}")
            return {}
