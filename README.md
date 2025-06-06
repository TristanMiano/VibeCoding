# VibeCoding

VibeCoding is a toolkit of “vibe‑coding” scripts and utilities—AI‑assisted helpers to streamline everyday development tasks. Each utility is standalone and configurable via JSON files under `config/`.

---

## 📦 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/TristanMiano/vibecoding.git
   cd vibecoding
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   # macOS/Linux:
   source venv/bin/activate
   # Windows (PowerShell):
   venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

Populate the following JSON files under `config/` before running any scripts:

* **`config/db_config.json`**
  PostgreSQL connection parameters: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`.

* **`config/llm_config.json`**
  Gemini/OpenAI settings: `GEMINI_API_KEY`, `DEFAULT_MODEL` (e.g., `gemini-2.0-flash`).

---

## 🛠 Utilities

### 1. `architecture_diagram.py`

Generate a module‑dependency diagram for Python projects.

```bash
python architecture_diagram.py path/to/project_root
```

### 2. `generate_project_overview.py`

Traverse the repo, extract directory structure, functions/classes, docstrings, and third‑party imports.

**Flags**:

* `--short`   Only extract function/class signatures and docstrings (where supported).
* `-e, --exclude`    Glob patterns for files/dirs to skip.
* `-i, --include`    Glob patterns to force‑include (overrides exclude).

```bash
# Full overview
python generate_project_overview.py

# Short mode, excluding generated folders
python generate_project_overview.py --short -e data/** build/**

# Force include a subdirectory
python generate_project_overview.py --short -e data/** -i data/important/**
```

Outputs:

* `project_overview.txt` (consolidated documentation)
* `requirements_autogenerated.txt` (sorted third‑party libs)

### 3. `project_query.py`

Build a dynamic prompt using `README.md` as background, then traverse and select relevant files via Gemini or fallback.

**Flags**:

* `--prompt`     (required)  User question to guide relevance filtering.
* `--directory`  (default: `.`)  Directory to traverse.
* `--token_limit` (default: `100000`)  Maximum tokens in the output.

```bash
python project_query.py \
  --prompt "Find all files relevant to large‑scale land battle planning." \
  --directory . \
  --token_limit 50000
```

Outputs to `output.txt` (or `output_1.txt`, etc.).

### 4. SQL‑Based Embeddings (`sql/` directory)

#### a. Create embeddings table

```bash
python sql/setup_project_embeddings.py --project my_project
```

#### b. Embed semantic units into Postgres

```bash
python sql/embed_code.py --project my_project --root path/to/code [--reembed]
```

#### c. Search embeddings

```bash
python sql/search_code.py \
  --project my_project \
  --query "function to connect to DB" \
  --mode chunks|files \
  --max_size 5000 \
  --root . \
  --output results.txt
```

---

## 📄 License

This project is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html).

---

## 🚀 Contributing

Contributions welcome! Add new “vibe‑coding” utilities under the repo root and update this README accordingly.
