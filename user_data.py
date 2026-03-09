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
