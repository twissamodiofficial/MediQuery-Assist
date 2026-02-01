import sqlite3
from pathlib import Path

DB_PATH = Path("data/user_data.db")

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP                 
        );
                         
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS document_classifications (
            file_hash TEXT PRIMARY KEY,
            doc_type TEXT NOT NULL,
            classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def add_user(user_id, name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""INSERT INTO users (id, name) VALUES (?, ?)
                   ON CONFLICT(id) DO UPDATE SET name = excluded.name""", (user_id, name))
    conn.commit()
    conn.close()

def create_session(user_id, session_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (id, user_id) VALUES (?, ?)", (session_id, user_id))
    conn.commit()
    conn.close()

def user_exists(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM users WHERE id = ?", (user_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def get_document_label(file_hash: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT doc_type FROM document_classifications WHERE file_hash = ?",
        (file_hash,)
    )
    row = cursor.fetchone()
    conn.close()
    return row["doc_type"] if row else None


def save_document_label(file_hash: str, doc_type: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO document_classifications (file_hash, doc_type)
        VALUES (?, ?)
        ON CONFLICT(file_hash) DO UPDATE SET
            doc_type = excluded.doc_type,
            classified_at = CURRENT_TIMESTAMP
        """,
        (file_hash, doc_type)
    )
    conn.commit()
    conn.close()