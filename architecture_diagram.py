#!/usr/bin/env python3
import os
import ast
import argparse
from collections import defaultdict

def find_py_files(root):
    """Recursively find all .py files under root, returning paths relative to root."""
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith('.py'):
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, root)
                files.append(rel)
    return files

def build_module_map(py_files):
    """
    Build a map from module name (e.g. "pkg.sub") to its relative path ("pkg/sub.py" or "pkg/sub/__init__.py").
    Treat __init__.py specially so that "pkg" points to "pkg/__init__.py".
    """
    mod_map = {}
    for rel in py_files:
        if rel.endswith('__init__.py'):
            mod = os.path.dirname(rel).replace(os.sep, '.')
        else:
            mod = rel[:-3].replace(os.sep, '.')
        mod_map[mod] = rel
    return mod_map

def parse_imports(full_path):
    """Return a list of raw import strings (including levels for from … import)."""
    with open(full_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=full_path)
    imps = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imps.append((0, alias.name))   # level=0 means absolute
        elif isinstance(node, ast.ImportFrom):
            level = node.level
            module = node.module or ''
            imps.append((level, module))
    return imps

def resolve_import(module_name, level, target_module):
    """
    Resolve a possibly-relative import (level, module) into an absolute module path.
    - module_name: the module of the file doing the import (e.g. "pkg.sub")
    - level: 0 for absolute imports, >0 for relative (number of leading dots)
    - target_module: the module path after the dots (e.g. "utils")
    """
    if level == 0:
        return target_module
    parts = module_name.split('.')
    if level > len(parts):
        # e.g. `from ... import foo` beyond top level
        base = []
    else:
        base = parts[:-level]
    if target_module:
        base += target_module.split('.')
    return '.'.join(base)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a module‐dependency diagram for a directory of .py files."
    )
    parser.add_argument(
        "root", help="Path to the root of your Python project"
    )
    args = parser.parse_args()
    root = args.root

    # 1. Find all .py files
    py_files = find_py_files(root)
    mod_map = build_module_map(py_files)

    # 2. Build dependency graph
    deps = defaultdict(set)
    rev_deps = defaultdict(set)
    for mod, rel_path in mod_map.items():
        full_path = os.path.join(root, rel_path)
        for level, imported in parse_imports(full_path):
            abs_mod = resolve_import(mod, level, imported)
            # Only record if it's one of our modules
            if abs_mod in mod_map:
                deps[rel_path].add(mod_map[abs_mod])
                rev_deps[mod_map[abs_mod]].add(rel_path)

    # 3. Print adjacency list
    print("MODULE DEPENDENCIES:\n")
    for path in sorted(py_files):
        children = sorted(deps.get(path, []))
        print(f"{path}: {', '.join(children) if children else '–'}")

    # 4. Identify truly independent modules
    independent = [
        path for path in py_files
        if not deps.get(path) and not rev_deps.get(path)
    ]

    if independent:
        print("\nINDEPENDENT MODULES (no imports, no-one imports them):\n")
        for path in sorted(independent):
            print(f"  {path}")

if __name__ == "__main__":
    main()
