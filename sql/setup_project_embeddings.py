#!/usr/bin/env python3
"""
Create a code embeddings table for a given project name.
Usage:
    python setup_project_embeddings.py --project my_project
"""
import argparse
import json
import re
import sys
from pathlib import Path
import psycopg2

# Path to your db_config.json (adjust as needed)
DB_CONFIG_PATH = Path(__file__).parent.parent / "config" / "db_config.json"


def load_db_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_db_connection(cfg):
    return psycopg2.connect(
        host=cfg['DB_HOST'],
        port=cfg['DB_PORT'],
        dbname=cfg['DB_NAME'],
        user=cfg['DB_USER'],
        password=cfg['DB_PASSWORD'],
    )


def sanitize_identifier(name: str) -> str:
    """
    Ensure the project name is a safe SQL identifier (letters, numbers, underscores).
    """
    if not re.match(r'^[A-Za-z0-9_]+$', name):
        raise ValueError("Project name must contain only letters, numbers, and underscores.")
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Create a dedicated code_embeddings table for a project."
    )
    parser.add_argument(
        '--project',
        required=True,
        help='Project name (used as table name prefix)'
    )
    args = parser.parse_args()

    project = sanitize_identifier(args.project)
    table_name = f"{project}_code_embeddings"

    # Load DB config and connect
    cfg = load_db_config(DB_CONFIG_PATH)
    conn = get_db_connection(cfg)
    cur = conn.cursor()

    # SQL template
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
      id SERIAL PRIMARY KEY,
      file_path TEXT NOT NULL,
      start_line INTEGER,
      end_line INTEGER,
      node_type TEXT,
      content TEXT NOT NULL,
      embedding VECTOR(384) NOT NULL,
      metadata JSONB,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_{project}_file_path
      ON {table_name}(file_path);

    CREATE INDEX IF NOT EXISTS idx_{project}_embedding
      ON {table_name} USING ivfflat (embedding) WITH (lists = 100);

    -- trigger function if not exists
    CREATE OR REPLACE FUNCTION update_{project}_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
      NEW.updated_at = NOW();
      RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    DROP TRIGGER IF EXISTS trg_update_{project}_updated_at ON {table_name};
    CREATE TRIGGER trg_update_{project}_updated_at
      BEFORE UPDATE ON {table_name}
      FOR EACH ROW EXECUTE FUNCTION update_{project}_updated_at_column();
    """

    try:
        cur.execute(ddl)
        conn.commit()
        print(f"Table '{table_name}' and indexes created or already exist.")
    except Exception as e:
        conn.rollback()
        print("Error creating table:", e)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == '__main__':
    main()
