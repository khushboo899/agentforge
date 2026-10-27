"""Dependency-free structural checks for an AgentForge checkout.

Run from any working directory with: python scripts/preflight.py
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "requirements.txt",
    ".env.example",
    "app/main.py",
    "app/graph/state.py",
    "app/graph/nodes.py",
    "app/graph/workflow.py",
    "app/rag/ingestion.py",
    "app/rag/retriever.py",
    "app/memory/conversation.py",
    "app/tools/calculator.py",
    "app/observability/tracker.py",
]

def main() -> int:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        print("FAIL: required project files are missing:")
        for name in missing:
            print(f"  - {name}")
        return 1

    python_files = sorted((ROOT / "app").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    python_files += sorted((ROOT / "scripts").glob("*.py"))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            print(f"FAIL: {path.relative_to(ROOT)}: {exc}")
            return 1

    if (ROOT / ".env").exists():
        print("WARNING: .env exists locally; never include it in a shared archive or Git commit.")
    print(f"PASS: {len(REQUIRED)} required files present; {len(python_files)} Python files parsed successfully.")
    print("NOTE: this preflight does not test installed dependencies, API credentials, or live provider access.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
