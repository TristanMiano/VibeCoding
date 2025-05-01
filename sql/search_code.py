#!/usr/bin/env python3
"""
Search embedded code chunks or files by semantic similarity.

Usage:
  python search_code.py \
    --project my_project \
    --query "function to connect to DB" \
    --mode chunks|files \
    --max_size 5000 \
    --root path/to/code \
    --output results.txt

Modes:
  chunks : return top code chunks in similarity order up to max_size characters total.
  files  : collect unique files by chunk similarity, then return full file contents up to max_size total.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import psycopg2
from sentence_transformers import SentenceTransformer

# postgres config file, relative to this script:
DB_CONFIG_PATH = Path(__file__).parent.parent / "config" / "db_config.json"
VECTOR_DIM = 384
MAX_CANDIDATES = 1000  # internal limit for fetched chunks


def load_db_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_db_connection(cfg):
    return psycopg2.connect(
        host=cfg['DB_HOST'], port=cfg['DB_PORT'], dbname=cfg['DB_NAME'],
        user=cfg['DB_USER'], password=cfg['DB_PASSWORD']
    )


def cast_vec_literal(vec):
    # represent as '[0.123,0.456,...]'
    return '[' + ','.join(f"{v:.6f}" for v in vec) + ']'  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, help='Project name (table prefix)')
    parser.add_argument('--query', required=True, help='Search query text')
    parser.add_argument('--mode', choices=('chunks','files'), required=True,
                        help='chunks: return code chunks; files: return full files')
    parser.add_argument('--max_size', type=int, required=True,
                        help='Maximum total characters to return')
    parser.add_argument('--root', default='.', help='Root dir for reading full files in files mode')
    parser.add_argument('--output', required=True, help='Output file path')
    args = parser.parse_args()

    table = f"{args.project}_code_embeddings"
    root_dir = Path(args.root)

    # load model & embed query
    print("Loading embedding model…")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    query_emb = model.encode(args.query, normalize_embeddings=True)
    vec_lit = cast_vec_literal(query_emb)

    # connect DB
    cfg = load_db_config(DB_CONFIG_PATH)
    conn = get_db_connection(cfg)
    cur = conn.cursor()

    # fetch top-N similar chunks
    sql = f"""
    SELECT file_path, content
      FROM {table}
      ORDER BY embedding <=> %s::vector
      LIMIT {MAX_CANDIDATES}
    """
    cur.execute(sql, (vec_lit,))
    rows = cur.fetchall()

    out_parts = []
    total = 0

    if args.mode == 'chunks':
        for file_path, content in rows:
            if total >= args.max_size:
                break
            part = f"// {file_path}\n{content}\n"
            out_parts.append(part)
            total += len(part)

    else:  # files mode
        seen = set()
        for file_path, _ in rows:
            if total >= args.max_size:
                break
            if file_path in seen:
                continue
            seen.add(file_path)
            full_path = root_dir / file_path
            try:
                text = full_path.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                print(f"Warning: could not read {full_path}: {e}")
                continue
            part = f"// {file_path}\n{text}\n"
            if total + len(part) > args.max_size:
                break
            out_parts.append(part)
            total += len(part)

    cur.close()
    conn.close()

    # write to output
    output = '\n'.join(out_parts)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"Wrote {len(out_parts)} sections to {args.output} ({total} chars)")

if __name__ == '__main__':
    main()
