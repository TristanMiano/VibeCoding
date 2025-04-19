#!/usr/bin/env python3
"""
semantic_code_selector.py

A utility to select the most contextually relevant code snippets from a large codebase
based on a natural-language query. It:
  1. Scans .py files under the project (skips data/, large dirs).
  2. Chunks each file into functions/classes with their surrounding imports.
  3. Embeds each chunk using OpenAI embeddings.
  4. Builds a FAISS index for fast similarity search.
  5. At query time, embeds the user query, retrieves top-k chunks, and prints them.

Dependencies:
  - openai
  - faiss-cpu
  - tiktoken

Configuration:
  Create `config/semantic_selector_config.json`:
  {
    "OPENAI_API_KEY": "<your_key>",
    "EMBED_MODEL": "text-embedding-ada-002"
  }

Usage:
  python semantic_code_selector.py \
    --query "describe next step: add semantic search utility" \
    --top-k 8 \
    --min-tok 200 --max-tok 1200
"""
import os
import sys
import json
import ast
import argparse
from pathlib import Path

import tiktoken
import openai
import faiss


def load_config(config_path="config/semantic_selector_config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    openai.api_key = cfg.get("OPENAI_API_KEY")
    return cfg


def chunk_python_file(file_path, encoder, min_tokens, max_tokens):
    """
    Parse a .py file and yield (chunk_text, metadata) tuples where chunk_text
    is between min_tokens and max_tokens tokens long. Metadata holds file/loc.
    """
    text = Path(file_path).read_text(encoding='utf-8')
    tree = ast.parse(text)
    lines = text.splitlines()

    # Gather function & class definitions spans
    spans = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno      # inclusive
            spans.append((start, end))

    # Fallback: if no spans, take entire file
    if not spans:
        spans = [(0, len(lines))]

    for start, end in spans:
        snippet = "\n".join(lines[start:end])
        tokens = encoder.encode(snippet)
        if len(tokens) < min_tokens:
            # pad with context: include imports at top
            imports = []
            for line in lines[:start]:
                if line.startswith("import") or line.startswith("from"):
                    imports.append(line)
            snippet = "\n".join(imports + [snippet])
            tokens = encoder.encode(snippet)
        if min_tokens <= len(tokens) <= max_tokens:
            yield snippet, {"file": str(file_path), "start": start+1, "end": end}


def build_embeddings(chunks, model):
    texts = [c for c, _ in chunks]
    resp = openai.Embedding.create(input=texts, model=model)
    return [e['embedding'] for e in resp['data']]


def create_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype='float32'))
    return index


def search(index, query_embedding, top_k):
    D, I = index.search(np.array([query_embedding], dtype='float32'), top_k)
    return I[0]


def main():
    parser = argparse.ArgumentParser(description="Select relevant code snippets by semantic query.")
    parser.add_argument("--query", required=True, help="Natural-language query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of snippets to return.")
    parser.add_argument("--min-tok", type=int, default=100, help="Minimum tokens per snippet.")
    parser.add_argument("--max-tok", type=int, default=800, help="Maximum tokens per snippet.")
    args = parser.parse_args()

    cfg = load_config()
    model = cfg.get("EMBED_MODEL", "text-embedding-ada-002")

    encoder = tiktoken.get_encoding("cl100k_base")
    project_root = Path(__file__).parent.resolve()

    # 1) chunk all .py files
    chunks = []
    for py in project_root.rglob('*.py'):
        if 'data' in py.parts or 'venv' in py.parts:
            continue
        for snippet, meta in chunk_python_file(py, encoder, args.min_tok, args.max_tok):
            chunks.append((snippet, meta))

    # 2) embed chunks
    embeddings = build_embeddings(chunks, model)

    # 3) build FAISS index
    idx = create_index(embeddings)

    # 4) embed query and search
    q_emb = openai.Embedding.create(input=[args.query], model=model)['data'][0]['embedding']
    hits = search(idx, q_emb, args.top_k)

    # 5) print selected snippets
    for i in hits:
        snippet, meta = chunks[i]
        print(f"--- {meta['file']}:{meta['start']}-{meta['end']} ---\n")
        print(snippet)
        print("\n")

if __name__ == '__main__':
    main()
