from __future__ import annotations
import os, io, threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageOps

from .settings import (
    PADDING, THUMB_SIZE, SELF_TEST_EACH_BATCH, HOST_DIVERSITY_IN_FALLBACK, FACE_MIN_RATIO,
)
from .model import Term, elo_update
from .storage import (
    load_terms, save_terms, used_cache_path, load_used, save_used,
    seen_hashes_path, load_seen_hashes, save_seen_hashes,
)
from .utils import slugify
from .image_search import ddg_candidates, download_image_bytes
from .config import get_last_terms_file, set_last_terms_file  # NEW

class PrefetchBatch:
    __slots__ = ("terms", "urls", "bytes_list")
    def __init__(self, terms, urls, bytes_list):
        self.terms = terms
        self.urls = urls
        self.bytes_list = bytes_list

class App:
    def __init__(self, root: tk.Tk, terms_path: str | None):
        self.root = root

        # Resolve terms file: use provided, else last-used, else prompt once
        resolved = terms_path
        if not resolved or not os.path.isfile(resolved):
            last = get_last_terms_file()
            if last and os.path.isfile(last):
                resolved = last
            else:
                pick = filedialog.askopenfilename(
                    title="Select terms file",
                    filetypes=[("Text/CSV", "*.txt *.csv"), ("All files", "*.*")]
                )
                if not pick:
                    messagebox.showinfo("No file selected", "No terms file was chosen. Exiting.")
                    raise SystemExit(0)
                resolved = pick
        set_last_terms_file(resolved)

        self.terms_path = resolved
        self.terms: list[Term] = load_terms(self.terms_path)
        if not self.terms:
            messagebox.showerror("No Terms", "The file is empty. Add terms (one per line) or 'term,rating'.")
            raise SystemExit(1)

        # caches
        self.used_path = used_cache_path(self.terms_path)
        self.used = load_used(self.used_path)
        self.seen_hashes_path = seen_hashes_path(self.terms_path)
        self.seen_hashes = load_seen_hashes(self.seen_hashes_path)

        # UI state
        self.extra_var = tk.StringVar(value="")
        self.status = tk.StringVar(value="Loading…")
        self.frames: list[ttk.Frame] = []
        self.labels: list[tk.Canvas] = []
        self.current_pils: list[Image.Image | None] = []
        self.captions: list[ttk.Label] = []
        self.other_btns: list[ttk.Button] = []

        self.current_four: list[Term] = []
        self.current_photos: list[ImageTk.PhotoImage | None] = [None]*4
        self.current_urls: list[str | None] = [None]*4
        self.current_raw_bytes: list[bytes | None] = [None]*4

        self.prefetch_lock = threading.Lock()
        self.prefetch_queue: list[PrefetchBatch] = []

        # Build UI
        self._build_ui()
        self._start_initial_prefetch()

    # ... (UNCHANGED UI + logic below, except one small tweak in _open_file to persist)
    # Keep everything from your current file, but in _open_file() add a call to set_last_terms_file(path)
    # Right after validating the selected file:

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select terms file",
            filetypes=[("Text/CSV", "*.txt *.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        terms = load_terms(path)
        if not terms:
            messagebox.showerror("No Terms", "That file is empty. Add terms (one per line) or 'term,rating'.")
            return

        # NEW: persist selection
        set_last_terms_file(path)

        # ... then your existing reload code (rebuild caches, clear grid, prefetch, status)
