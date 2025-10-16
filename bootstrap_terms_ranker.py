#!/usr/bin/env python3
"""
Creates a minimal 'terms-ranker' style project with first-version files.
Usage:
  python bootstrap_terms_ranker.py --name terms-ranker
If --name is omitted, defaults to 'terms-ranker'.
"""

from __future__ import annotations
import argparse, os, sys, textwrap
from pathlib import Path

PYPROJECT = """\
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "termsranker"
version = "0.1.0"
description = "Interactive GUI to rank search terms by picking the best of 4 images."
readme = "README.md"
requires-python = ">=3.10"
authors = [{ name = "Your Name" }]
license = { text = "MIT" }

dependencies = [
  "ddgs>=2.0.0",
  "requests>=2.31",
  "pillow>=10.0",
]

[project.optional-dependencies]
vision = [
  "opencv-python-headless>=4.8",
  "numpy>=1.26",
  "imagehash>=4.3",
]
deepface = [
  "deepface>=0.0.92",
  "numpy>=1.26",
]

[project.scripts]
terms-ranker = "termsranker.app:main"
"""

GITIGNORE = """\
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv/
venv/
.envrc

# Build
dist/
build/
*.egg-info/

# OS/editor
.DS_Store
Thumbs.db
.vscode/
.idea/

# App artifacts
ranked_images/
*_used_urls.txt
*_seen_hashes.txt
"""

README = """\
# Terms Ranker

A Tkinter GUI to rank internet search terms by picking the best of 4 images.

## Quick start

```sh
# create & activate venv (Windows PowerShell)
py -m venv .venv
.\.venv\\Scripts\\Activate.ps1

# macOS/Linux
# python3 -m venv .venv && source .venv/bin/activate

python -m pip install -U pip
pip install -e .
# optional extras:
# pip install -e .[vision]
# pip install -e .[deepface]

terms-ranker
# (or) python -m termsranker.app
```

Put a `terms.csv` (one term per line, or `term,rating`) anywhere and open it from the app.
"""

INIT_PY = """\
__all__ = ["__version__"]
__version__ = "0.1.0"
"""

CORE_PY = """\
from dataclasses import dataclass

DEFAULT_RATING = 1500.0
SIGMA0 = 350.0
SIGMA_FLOOR = 60.0
BASE_K = 36.0

@dataclass
class Term:
    name: str
    rating: float = DEFAULT_RATING
    games: int = 0
    sigma: float = SIGMA0

def elo_update(winner: Term, losers: list[Term]) -> None:
    for loser in losers:
        diff = loser.rating - winner.rating
        expected = 1.0 / (1.0 + 10.0 ** (diff / 400.0))
        u_scale = min(2.0, max(0.5, (winner.sigma + loser.sigma) / (2 * SIGMA0)))
        k = BASE_K * u_scale
        delta = k * (1.0 - expected)
        winner.rating += delta
        loser.rating -= delta
        winner.games += 1
        loser.games += 1
        winner.sigma = max(SIGMA_FLOOR, winner.sigma * 0.98)
        loser.sigma  = max(SIGMA_FLOOR, loser.sigma * 0.98)
"""

PERSISTENCE_PY = """\
from __future__ import annotations
import csv, os
from .core import Term, DEFAULT_RATING

def load_terms(path: str) -> list[Term]:
    terms: list[Term] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row: continue
            name = (row[0] or "").strip()
            if not name: continue
            key = " ".join(name.split()).casefold()
            if key in seen: continue
            seen.add(key)
            rating = DEFAULT_RATING
            if len(row) >= 2:
                try: rating = float(row[1])
                except: pass
            terms.append(Term(name=name, rating=rating))
    return terms

def save_terms(path: str, terms: list[Term]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for t in terms:
            w.writerow([t.name, f"{t.rating:.1f}"])
    os.replace(tmp, path)
"""

IMAGES_PY = """\
from __future__ import annotations
from typing import Optional
import io
import requests
from PIL import Image
# from ddgs import DDGS  # hook this up when wiring image search

def fetch_image_bytes(url: str, timeout: int = 12) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")
"""

APP_PY = """\
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from .persistence import load_terms

def main():
    root = tk.Tk()
    root.title("Terms Ranker")
    root.geometry("900x600")

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill=tk.BOTH, expand=True)

    title = ttk.Label(frm, text="Terms Ranker (skeleton)", font=("Segoe UI", 14, "bold"))
    title.pack(anchor="w")

    info = ttk.Label(frm, text="Open a terms file (TXT/CSV). Next step: plug in 4-image chooser UI.")
    info.pack(anchor="w", pady=(6, 12))

    status = tk.StringVar(value="No file loaded.")
    ttk.Label(frm, textvariable=status).pack(anchor="w", pady=(0,12))

    def open_terms():
        path = filedialog.askopenfilename(
            title="Select terms file",
            filetypes=[("Text/CSV", "*.txt *.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            terms = load_terms(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")
            return
        if not terms:
            messagebox.showwarning("Empty", "No terms found in that file.")
            return
        status.set(f"Loaded {len(terms)} terms from {path}")
        messagebox.showinfo("Loaded", f"{len(terms)} terms loaded. (UI to come)")

    ttk.Button(frm, text="Open Terms File…", command=open_terms).pack(anchor="w")
    root.mainloop()
"""

TERMS_CSV = """\
puppies
sunsets
mountain landscapes
"""

LEGACY_README = """\
Put your old script file here (e.g., terms_ranker_manage_alpha.py) for reference.
This folder is not imported by the package.
"""

def safe_write(path: Path, content: str):
    if path.exists():
        print(f"SKIP (exists): {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"CREATE       : {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="terms-ranker", help="Project folder name to create")
    args = ap.parse_args()

    root = Path(args.name).resolve()
    print(f"Creating project at: {root}")
    root.mkdir(parents=True, exist_ok=True)

    # Top-level files
    safe_write(root / "pyproject.toml", PYPROJECT)
    safe_write(root / ".gitignore", GITIGNORE)
    safe_write(root / "README.md", README)

    # Package
    pkg = root / "src" / "termsranker"
    safe_write(pkg / "__init__.py", INIT_PY)
    safe_write(pkg / "core.py", CORE_PY)
    safe_write(pkg / "persistence.py", PERSISTENCE_PY)
    safe_write(pkg / "images.py", IMAGES_PY)
    safe_write(pkg / "app.py", APP_PY)
    (pkg / "gui").mkdir(parents=True, exist_ok=True)

    # Examples
    safe_write(root / "examples" / "terms.csv", TERMS_CSV)

    # Legacy placeholder
    safe_write(root / "legacy" / "README.txt", LEGACY_README)

    # tests dir
    (root / "tests").mkdir(parents=True, exist_ok=True)

    print("\nDone. Next steps:")
    print(textwrap.dedent(f"""
      1) cd "{root}"
      2) Create + activate a venv, then install:
         - Windows PowerShell:
             py -m venv .venv
             .\\.venv\\Scripts\\Activate.ps1
             python -m pip install -U pip
             pip install -e .
         - macOS/Linux:
             python3 -m venv .venv
             source .venv/bin/activate
             python -m pip install -U pip
             pip install -e .
      3) Run the app:
             terms-ranker
         or: python -m termsranker.app
      4) Move your old file into: {root / "legacy"}
    """).strip())

if __name__ == "__main__":
    sys.exit(main())
