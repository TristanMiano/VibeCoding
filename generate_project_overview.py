#!/usr/bin/env python3
"""
generate_project_overview.py

Usage examples:

  # include everything under src and drafts, but prune one subtree
  python generate_project_overview.py \
      -i src -i drafts \
      -e drafts/other_rulesets

  # exclude build/ and data/, default include whole tree
  python generate_project_overview.py -e build -e data

  # include only docs/ and examples/
  python generate_project_overview.py -i docs -i examples
"""

import os
import sys
import argparse
from pathlib import Path

from tree_sitter import Language, Parser
from tree_sitter_language_pack import get_parser
import ast  # still used for Python import parsing

# Map file extensions to tree-sitter language names (must be compiled into the shared library)
EXT_LANG_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.java': 'java',
    '.c': 'c',
    '.cpp': 'cpp',
    '.go': 'go',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
    '.rs': 'rust',
}

# A naive list of known standard library modules we skip from "requirements".
STANDARD_LIBS = {
    "abc", "argparse", "ast", "asyncio", "base64", "binascii", "bisect", "builtins", "calendar",
    "collections", "concurrent", "contextlib", "copy", "csv", "ctypes", "datetime", "decimal",
    "difflib", "dis", "distutils", "email", "enum", "errno", "faulthandler", "filecmp", "fileinput",
    "fnmatch", "fractions", "functools", "gc", "getopt", "getpass", "gettext", "glob", "gzip", "hashlib",
    "heapq", "hmac", "http", "imaplib", "imp", "importlib", "inspect", "io", "ipaddress", "itertools",
    "json", "logging", "lzma", "math", "multiprocessing", "numbers", "operator", "os", "pathlib",
    "pickle", "platform", "plistlib", "pprint", "queue", "random", "re", "runpy", "sched", "secrets",
    "select", "shlex", "shell", "shutil", "signal", "site", "smtp", "smtplib", "socket", "socketserver",
    "sqlite3", "ssl", "stat", "statistics", "string", "struct", "subprocess", "sys", "tempfile", "termios",
    "textwrap", "threading", "time", "timeit", "tkinter", "traceback", "types", "typing", "unittest",
    "urllib", "uuid", "venv", "warnings", "wave", "weakref", "webbrowser", "xml", "xmlrpc", "zipfile", "zipimport"
}

PROMPT_TEXT = """PROMPT FOR AI MODEL:

You are about to read a detailed overview of a software project. Please read everything in the following text and act as a helpful software engineering assistant. 

At the end of the overview, there will be a list of next steps for implementation. Please tailor your response for these steps. Generally, if more than one step is listed, focus on the first one only in your first response. The user will probably request the subsequent steps later.

Do not add unnecessary complexity. Do not assume the user will infer the proper steps to take if you leave some out. Be very explicit. If you generate code, generate the entire file fully working. You may generate code snippets if the user asks for those. If you do, please explain exactly where to put them.

If you can't find any next steps for the project listed at the bottom of the file, please do your best to look for mistakes, errors, discrepancies, or ways to clean up and refine the project, and decide yourself what should be considered high priority, and include that in your first response.

--------------------------------------------------------------------------------
"""


def is_text_file(file_path: Path) -> bool:
    text_extensions = set(EXT_LANG_MAP.keys()) | {'.txt', '.md', '.rst', '.html', '.css', '.xml', '.yaml', '.yml', '.sh', '.bat', '.sql'}
    return file_path.suffix.lower() in text_extensions


def get_file_type(file_path: Path) -> str:
    return file_path.suffix.lower() if file_path.suffix else 'No Extension'


def parse_imports_from_python(file_content: str) -> set[str]:
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split('.')[0])
    return mods


def is_standard_library(module_name: str) -> bool:
    return module_name in STANDARD_LIBS


def extract_tree_functions(file_bytes: bytes, ext: str, parsers: dict) -> str:
    lang_name = EXT_LANG_MAP.get(ext)
    parser = parsers.get(lang_name)
    if not parser:
        return ''
    tree = parser.parse(file_bytes)
    root = tree.root_node
    lines: list[str] = []

    # Module-level docstring or leading comments
    if lang_name == 'python':
        for child in root.children:
            if child.type == 'expression_statement' and child.children and child.children[0].type == 'string':
                module_doc = child.children[0].text.decode('utf-8')
                lines.append(f'Module Docstring:\n{module_doc}\n')
                break
    else:
        leading_comments = []
        for child in root.children:
            if child.type == 'comment':
                leading_comments.append(child.text.decode('utf-8'))
            else:
                break
        if leading_comments:
            lines.append('Module Comments:')
            lines += leading_comments
            lines.append('')

    def traverse(node):
        yield node
        for c in node.children:
            yield from traverse(c)

    func_nodes = {'function_definition', 'function_declaration', 'method_definition', 'method_declaration', 'arrow_function', 'function'}
    class_nodes = {'class_definition', 'class_declaration', 'struct_specifier'}

    for node in traverse(root):
        if node.type in func_nodes or node.type in class_nodes:
            name = '<anonymous>'
            for c in node.children:
                if c.type in ('identifier', 'name') or c.type.endswith('name'):
                    name = c.text.decode('utf-8')
                    break
            kind = 'def' if node.type in func_nodes else 'class'
            lines.append(f'{kind} {name}()')
            doc = None
            if lang_name == 'python':
                for c in node.children:
                    if c.type == 'suite':
                        for stmt in c.children:
                            if stmt.type == 'expression_statement' and stmt.children and stmt.children[0].type == 'string':
                                doc = stmt.children[0].text.decode('utf-8')
                                break
                        break
            else:
                comments = [n for n in traverse(root) if n.type == 'comment' and n.end_byte < node.start_byte]
                if comments:
                    doc = comments[-1].text.decode('utf-8')
            if doc:
                for l in doc.splitlines():
                    lines.append('    ' + l)
            lines.append('')
    return '\n'.join(lines)


def normalize_prefix(p: str) -> str:
    # Turn a user-specified path (possibly with glob suffix) into a simple prefix
    p = p.replace('\\', '/').rstrip('/')
    for ch in ('*', '?'):
        idx = p.find(ch)
        if idx != -1:
            p = p[:idx]
            break
    p = p.rstrip('/')
    return p or '.'


def traverse_directory(
    root_path: Path,
    short_version: bool = False,
    exclude_prefixes: list[str] = None,
    include_prefixes: list[str] = None,
    parsers: dict = None
):
    exclude_prefixes = exclude_prefixes or []
    include_prefixes = include_prefixes or []

    directory_structure: list[str] = []
    relevant_contents: list[str] = []
    third_party_libraries: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root_path, topdown=True):
        rel_dir = Path(dirpath).relative_to(root_path).as_posix()
        rel_dir = '.' if rel_dir == '' else rel_dir

        # 1) Prune excluded subtrees
        if rel_dir != '.' and any(rel_dir == ex or rel_dir.startswith(ex + '/') for ex in exclude_prefixes):
            dirnames[:] = []
            continue

        # 2) Skip any directory not under an include prefix (if any includes given)
        if include_prefixes and rel_dir != '.' and not any(rel_dir == inc or rel_dir.startswith(inc + '/') for inc in include_prefixes):
            dirnames[:] = []
            continue

        directory_structure.append(f'Directory: {rel_dir}')

        # 3) Prune children of this directory
        new_dirnames: list[str] = []
        for d in dirnames:
            child_rel = d if rel_dir == '.' else f"{rel_dir}/{d}"
            # a) exclude wins
            if any(child_rel == ex or child_rel.startswith(ex + '/') for ex in exclude_prefixes):
                continue
            # b) must be on path to or under an include (if includes given)
            if include_prefixes:
                keep = False
                for inc in include_prefixes:
                    if (child_rel == inc
                            or child_rel.startswith(inc + '/')
                            or inc.startswith(child_rel + '/')):
                        keep = True
                        break
                if not keep:
                    continue
            new_dirnames.append(d)
        dirnames[:] = new_dirnames

        # 4) Process files in this directory
        for fn in filenames:
            if fn.startswith('.'):
                continue
            file_rel = fn if rel_dir == '.' else f"{rel_dir}/{fn}"
            # a) exclude wins
            if any(file_rel == ex or file_rel.startswith(ex + '/') for ex in exclude_prefixes):
                continue
            # b) must be under an include directory (if includes given)
            if include_prefixes and not any(file_rel == inc or file_rel.startswith(inc + '/') for inc in include_prefixes):
                continue

            p = Path(dirpath) / fn
            directory_structure.append(f'  File: {fn} | Type: {get_file_type(p)}')

            if is_text_file(p):
                try:
                    ext = p.suffix.lower()
                    if ext == '.py':
                        content_py = p.read_text(encoding='utf-8')
                        imports = parse_imports_from_python(content_py)
                        for lib in imports:
                            if not is_standard_library(lib):
                                third_party_libraries.add(lib)
                    if short_version and ext in EXT_LANG_MAP:
                        content_bytes = p.read_bytes()
                        snippet = extract_tree_functions(content_bytes, ext, parsers)
                        relevant_contents.append(f"\n=== Functions & Docstrings in {file_rel} ===\n{snippet}")
                    else:
                        content = p.read_text(encoding='utf-8')
                        relevant_contents.append(f"\n=== Content of {file_rel} ===\n{content}")
                except Exception as e:
                    print(f"Warning: could not read {p}: {e}", file=sys.stderr)

    return directory_structure, relevant_contents, third_party_libraries


def main():
    parser = argparse.ArgumentParser(description="Generate a project overview.")
    parser.add_argument(
        "--short", action="store_true", default=False,
        help="Only include function signatures & docstrings for supported languages."
    )
    parser.add_argument(
        "-e", "--exclude", nargs="*", default=[],
        help="Path prefixes to skip (with cascading)."
    )
    parser.add_argument(
        "-i", "--include", nargs="*", default=[],
        help="Path prefixes to include (with cascading)."
    )

    args = parser.parse_args()

    exclude_prefixes = [normalize_prefix(p) for p in args.exclude]
    include_prefixes = [normalize_prefix(p) for p in args.include]

    # Initialize parsers for each language
    parsers: dict[str, any] = {}
    for ext, lang_name in EXT_LANG_MAP.items():
        try:
            parsers[lang_name] = get_parser(lang_name)
        except KeyError:
            pass

    root = Path.cwd().resolve()
    print(f"Traversing directory: {root}")

    dirs, contents, libs = traverse_directory(
        root_path=root,
        short_version=args.short,
        exclude_prefixes=exclude_prefixes,
        include_prefixes=include_prefixes,
        parsers=parsers
    )

    overview_path = root / "project_overview.txt"
    with overview_path.open('w', encoding='utf-8') as f:
        f.write(PROMPT_TEXT)
        f.write("\n=== Directory Structure ===\n")
        f.write("\n".join(dirs))
        f.write("\n\n=== Consolidated Documentation ===\n")
        f.write("\n".join(contents))
        f.write("\n\n" + "-"*80)
    print(f"Written overview to {overview_path}")

    req_path = root / "requirements_autogenerated.txt"
    with req_path.open('w', encoding='utf-8') as f:
        for lib in sorted(libs):
            f.write(lib.lower() + "\n")
    print(f"Written requirements to {req_path}")


if __name__ == "__main__":
    main()
