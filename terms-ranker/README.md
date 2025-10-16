# Terms Ranker

A Tkinter GUI to rank internet search terms by picking the best of 4 images.

## Quick start

```sh
# create & activate venv (Windows PowerShell)
py -m venv .venv
.\.venv\Scripts\Activate.ps1

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

Drop-in replacement modules for your project (v0.2.0).

HOW TO APPLY (Windows PowerShell recommended):
  1) Extract termsranker_port_v1.zip
  2) Copy the folder 'src/termsranker/' over your project's src/termsranker/
     (replace existing files `__init__.py`, `core.py`, `persistence.py`, `images.py`, `app.py`).
  3) Activate your venv and reinstall (editable) if needed:
        .\.venv\Scripts\Activate.ps1
        pip install -e .
  4) Run:
        terms-ranker
     or: python -m termsranker.app

Notes:
  - Optional deps (DeepFace, OpenCV, imagehash) are still optional. Install extras as needed:
        pip install -e .[vision]
        pip install -e .[deepface]
  - Settings dialog lets you toggle filters and weightings at runtime.
