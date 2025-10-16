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
