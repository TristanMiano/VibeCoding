#!/usr/bin/env python3
"""
Embed semantic units of source files into a pgvector table for a given project.

Usage:
  python embed_code.py --project my_project [--root path/to/code] [--reembed]

Requirements:
  pip install psycopg2-binary sentence-transformers tree_sitter tree_sitter_languages
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import psycopg2
from sentence_transformers import SentenceTransformer
from tree_sitter import Parser
from tree_sitter_language_pack import get_language, get_parser

# adjust these to suit your chunk-size needs:
MAX_CHARS = 1000
OVERLAP = 200

# postgres config file, relative to this script:
DB_CONFIG_PATH = Path(__file__).parent.parent / "config" / "db_config.json"

# map file extensions to Tree-sitter language names
EXT_LANG_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.go': 'go',
    '.java': 'java',
    '.c': 'c',
    '.cpp': 'cpp',
    '.rb': 'ruby',
    '.php': 'php',
    '.rs': 'rust',
    '.cs': 'c_sharp',
}


def load_db_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_db_connection(cfg: dict):
    return psycopg2.connect(
        host=cfg['DB_HOST'],
        port=cfg['DB_PORT'],
        dbname=cfg['DB_NAME'],
        user=cfg['DB_USER'],
        password=cfg['DB_PASSWORD'],
    )


def sanitize_identifier(name: str) -> str:
    if not re.match(r'^[A-Za-z0-9_]+$', name):
        raise ValueError("Project name must contain only letters, numbers, and underscores.")
    return name


def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks


def extract_units(text: str, lang_name: str) -> list[tuple[int,int,str,str]]:
    parser = Parser()
    try:
        parser = get_parser(lang_name)
    except Exception as e:
        print(f"Failed to load Tree-sitter language '{lang_name}': {e}")
        return [(1, len(text.splitlines()), text, 'file')]

    tree = parser.parse(bytes(text, 'utf8'))
    root = tree.root_node
    units = []
    for node in root.children:
        # adjust these node types per language if needed
        if node.type in ('function_definition', 'class_definition', 'method_definition',
                         'module', 'comment', 'declaration'):
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            content = text[node.start_byte:node.end_byte]
            units.append((start_line, end_line, content, node.type))

    if not units:
        # fallback to whole file
        units = [(1, len(text.splitlines()), text, 'file')]
    return units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, help='Project name (table prefix)')
    parser.add_argument('--root', default='.', help='Root directory to traverse')
    parser.add_argument('--reembed', action='store_true', help='Delete existing rows for each file before inserting')
    args = parser.parse_args()

    project = sanitize_identifier(args.project)
    table = f"{project}_code_embeddings"
    root_dir = Path(args.root)

    print("Loading embedding model…")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    cfg = load_db_config(DB_CONFIG_PATH)
    conn = get_db_connection(cfg)
    cur = conn.cursor()

    for root, dirs, files in os.walk(root_dir):
        # skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', '__pycache__')]
        for fname in files:
            path = Path(root) / fname
            ext = path.suffix.lower()
            if ext not in EXT_LANG_MAP:
                continue

            rel_path = str(path.relative_to(root_dir))
            print(f"\n---\nProcessing '{rel_path}'")

            if args.reembed:
                cur.execute(f"DELETE FROM {table} WHERE file_path = %s", (rel_path,))
                conn.commit()

            text = path.read_text(encoding='utf-8', errors='ignore')
            units = extract_units(text, EXT_LANG_MAP[ext])
            print(f"  → {len(units)} semantic units")

            for idx, (start, end, content, node_type) in enumerate(units, start=1):
                chunks = chunk_text(content) if len(content) > MAX_CHARS else [content]
                for c_idx, chunk in enumerate(chunks, start=1):
                    print(f"    • unit {idx}/{len(units)}, chunk {c_idx}/{len(chunks)}", end="\r")
                    emb = model.encode(chunk, normalize_embeddings=True)
                    vec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
                    meta = json.dumps({
                        "file": rel_path,
                        "unit_index": idx-1,
                        "chunk_index": c_idx-1,
                        "start_line": start,
                        "end_line": end,
                        "node_type": node_type
                    })

                    cur.execute(
                        f"""
                        INSERT INTO {table}
                          (file_path, start_line, end_line, node_type, content, embedding, metadata, created_at, updated_at)
                        VALUES
                          (%s, %s, %s, %s, %s, %s::vector, %s, NOW(), NOW())
                        """,
                        (rel_path, start, end, node_type, chunk, vec_literal, meta)
                    )
            conn.commit()
            print(f"  ✓ done '{rel_path}'")

    cur.close()
    conn.close()
    print("\nAll done!")


if __name__ == '__main__':
    main()
