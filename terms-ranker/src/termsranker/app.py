from __future__ import annotations
import io, os, sys, threading, json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageOps

import termsranker.images as images
import termsranker.core as core
from .core import Term, elo_update, weighted_sample_terms, COOLDOWN_WINDOW
from .persistence import (
    load_terms, save_terms, used_cache_path, seen_hashes_path, load_used, save_used,
    load_seen_hashes, save_seen_hashes, slugify
)
from .images import (
    ddg_candidates, download_image_bytes, looks_like_woman, phash_bytes, host_of
)

CONFIG_PATH = os.path.join(Path.home(), ".termsranker.json")
PADDING = 6
GRID_GAP = 2
DEBUG_OVERLAY = True

SETTINGS_DEFAULTS = {
    "cooldown_window": 10,
    "recency_cap_factor": 10,
    "enforce_woman": True,
    "face_required": True,
    "face_min_frac": 0.18,
    "face_min_neighbors": 4,
    "face_scale_factor": 1.10,
    "min_dim": 256,
    "ddg_max_results": 60,
    "top_sample_k": 8,
    "timeout": 7,
    "unique_hosts_per_round": True,
    "avoid_dup_hashes": True,
    "prefetch_max_queue": 2,
    "use_used_url_cache": True,
    "use_seen_hash_cache": True,
    "gallery_batch_size": 8,
    "gallery_vis_buffer": 800,
}

def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return {}

def save_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f: json.dump(cfg, f, indent=2)
    except Exception: pass

def cfg_get():
    cfg = load_config()
    for k, v in SETTINGS_DEFAULTS.items():
        cfg.setdefault(k, v)
    return cfg

def cfg_save(cfg: dict):
    for k, v in SETTINGS_DEFAULTS.items():
        cfg.setdefault(k, v)
    save_config(cfg)

def draw_title(canvas: "tk.Canvas", text: str):
    try:
        canvas.delete("title")
        canvas.create_text(8, 8, text=text, anchor="nw", fill="#ddd", tags="title")
    except Exception: pass

def draw_debug(canvas: "tk.Canvas", text: str):
    if not DEBUG_OVERLAY: return
    try: w = max(1, int(canvas.winfo_width()))
    except Exception: w = 400
    try:
        canvas.delete("dbg")
        canvas.create_text(w-8, 8, text=text, anchor="ne", fill="#aaa", font=("Segoe UI", 9), tags="dbg")
    except Exception: pass

def canvas_show_error(canvas: "tk.Canvas", msg: str):
    try:
        canvas.delete("img"); canvas.delete("err")
        w = max(1, int(canvas.winfo_width())); h = max(1, int(canvas.winfo_height()))
        canvas.create_text(w//2, h//2, text=msg, fill="#ddd", tags="err")
    except Exception: pass

def render_to_canvas(canvas: "tk.Canvas", pil_img: Image.Image):
    try:
        w = max(1, int(canvas.winfo_width())); h = max(1, int(canvas.winfo_height()))
    except Exception:
        w = h = 0
    if w < 10 or h < 10: w, h = 400, 400
    try: resample = Image.Resampling.LANCZOS
    except Exception: resample = Image.ANTIALIAS  # type: ignore
    fitted = ImageOps.contain(pil_img.convert("RGB"), (w, h), method=resample)
    tkimg = ImageTk.PhotoImage(fitted)
    canvas.delete("img")
    canvas.create_image(w//2, h//2, image=tkimg, anchor="center", tags="img")
    canvas.image = tkimg

def _has_face_bytes(data: bytes) -> bool:
    cfg = cfg_get()
    if not cfg.get("face_required", True):
        return True
    try:
        import cv2, numpy as np
    except Exception:
        return True
    try:
        nparr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        face_min_frac = float(cfg.get("face_min_frac", 0.18))
        min_size = max(16, int(max(h, w) * face_min_frac))
        cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=float(cfg.get("face_scale_factor", 1.10)),
            minNeighbors=int(cfg.get("face_min_neighbors", 4)),
            minSize=(min_size, min_size)
        )
        return bool(faces)
    except Exception:
        return True

@dataclass(frozen=True)
class PrefetchEntry:
    term: Term
    url: str | None
    data: bytes | None

@dataclass(frozen=True)
class PrefetchBatch:
    entries: list[PrefetchEntry]

class App:
    def __init__(self, root: tk.Tk, terms_path: str | None):
        self.root = root
        cfg = cfg_get()
        core.COOLDOWN_WINDOW = int(cfg.get("cooldown_window", 10))
        images.ENFORCE_WOMAN = bool(cfg.get("enforce_woman", True))
        images.MIN_DIM = int(cfg.get("min_dim", 256))
        images.DDG_MAX_RESULTS = int(cfg.get("ddg_max_results", 60))
        images.TOP_SAMPLE_K = int(cfg.get("top_sample_k", 8))
        images.TIMEOUT = int(cfg.get("timeout", 7))

        self.terms_path = terms_path or load_config().get("last_terms_path")
        if not self.terms_path:
            self.terms_path = filedialog.askopenfilename(
                title="Select terms file",
                filetypes=[("Text/CSV","*.txt *.csv"),("All files","*.*")]
            )
            if not self.terms_path: print("No file selected."); sys.exit(0)
        cfg2 = load_config(); cfg2["last_terms_path"] = self.terms_path; save_config(cfg2)

        self.terms: list[Term] = load_terms(self.terms_path)
        if not self.terms:
            messagebox.showerror("No Terms","The file is empty. Add terms (one per line) or 'term,rating[,games,sigma]'.")
            sys.exit(1)

        self.used_path = used_cache_path(self.terms_path); self.used = load_used(self.used_path)
        self.seen_hashes_path = seen_hashes_path(self.terms_path); self.seen_hashes = load_seen_hashes(self.seen_hashes_path)

        self.round_id = 0
        self.recency_map: dict[str,int] = {}

        self.extra_var = tk.StringVar(value="")
        self.frames: list[ttk.Frame] = []
        self.labels: list[tk.Canvas] = []
        self.overlay_btns: list[ttk.Button] = []
        self.current_pils: list[Image.Image | None] = [None]*4
        self.current_four: list[Term] = []
        self.current_urls: list[str | None] = [None]*4
        self.current_raw_bytes: list[bytes | None] = [None]*4

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label = None

        self.prefetch_lock = threading.Lock()
        self.prefetch_queue: list[PrefetchBatch] = []

        self.root.title("Terms Ranker")
        root.grid_rowconfigure(1, weight=1); root.grid_columnconfigure(0, weight=1)

        style = ttk.Style(self.root)
        style.configure("Other.TButton", font=("Segoe UI", 12, "bold"), padding=(14, 10))

        top = ttk.Frame(root, padding=PADDING); top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)
        leftbar = ttk.Frame(top); leftbar.grid(row=0, column=0, sticky="w")
        ttk.Label(leftbar, text="Click the best image").pack(side=tk.LEFT)
        ttk.Label(leftbar, text="  |  Extra terms:").pack(side=tk.LEFT, padx=(8,2))
        ttk.Entry(leftbar, textvariable=self.extra_var, width=30).pack(side=tk.LEFT, padx=(0,10))
        rightbar = ttk.Frame(top); rightbar.grid(row=0, column=1, sticky="e")
        for text, cmd in [("Gallery", self._open_gallery), ("Settings…", self._open_settings),
                          ("Open Terms File", self._open_file), ("Remove Terms…", self._open_remove_terms),
                          ("Add Terms…", self._open_add_terms),
                          ("Progress 3D…", self._open_progress3d),
                          ("DeDupe", self._run_dedupe)]:
            ttk.Button(rightbar, text=text, command=cmd).pack(side=tk.LEFT, padx=(0,6))

        ttk.Label(rightbar, text="  ").pack(side=tk.LEFT)
        ttk.Progressbar(rightbar, orient="horizontal", length=180, mode="determinate",
                        variable=self.progress_var).pack(side=tk.LEFT, padx=(0,6))
        self.progress_label = ttk.Label(rightbar, text="Certainty: 0% (avg σ=0.00)")
        self.progress_label.pack(side=tk.LEFT)

        grid = ttk.Frame(root, padding=0); grid.grid(row=1, column=0, sticky="nsew")
        for r in (0,1): grid.grid_rowconfigure(r, weight=1)
        for c in (0,1): grid.grid_columnconfigure(c, weight=1)

        for i in range(4):
            r, c = divmod(i, 2)
            slot = ttk.Frame(grid, padding=0); slot.grid(row=r, column=c, sticky="nsew", padx=GRID_GAP, pady=GRID_GAP)
            slot.grid_rowconfigure(0, weight=1); slot.grid_columnconfigure(0, weight=1)
            cvs = tk.Canvas(slot, highlightthickness=0, bg="#222"); cvs.grid(row=0, column=0, sticky="nsew")
            cvs.bind("<Configure>", lambda e, idx=i: self._on_canvas_resize(idx))
            cvs.bind("<Button-1>", lambda e, idx=i: self._on_click(idx))
            btn = ttk.Button(cvs, text="OTHER", style="Other.TButton", command=lambda idx=i: self._on_other(idx))
            def place_btn(canvas=cvs, b=btn):
                try:
                    w = max(1, canvas.winfo_width()); h = max(1, canvas.winfo_height())
                    b.place(x=w-10, y=h-10, anchor="se")
                except Exception: pass
            cvs.bind("<Configure>", lambda e, pb=place_btn: pb()); place_btn()
            self.frames.append(slot); self.labels.append(cvs); self.overlay_btns.append(btn)
            draw_title(cvs, "Loading…")

        self._update_progress()
        self._start_initial_prefetch()

    def _open_gallery(self):
        try: GalleryWindow(self)
        except Exception: pass

    def _open_progress3d(self):
        if not self.terms:
            messagebox.showwarning("No terms", "No terms loaded.")
            return
        def runner():
            try:
                from .progress_3d import show_progress_3d
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Matplotlib missing",
                    "The 3D view requires matplotlib.\n\nFix in PowerShell:\n"
                    "  .\\.venv\\Scripts\\Activate.ps1\n  pip install matplotlib\n\n"
                    f"Details: {e}"
                ))
                return
            show_progress_3d(self.terms)
        threading.Thread(target=runner, daemon=True).start()

    def _run_dedupe(self):
        try:
            from .dedupe import dedupe_ranked_images, THRESHOLD
        except Exception as e:
            messagebox.showerror("DeDupe", f"Couldn't import dedupe tool.\n\nDetails: {e}")
            return
        if not self.terms_path:
            messagebox.showwarning("DeDupe", "No terms file loaded.")
            return
        confirm = messagebox.askyesno(
            "DeDupe",
            f"This will permanently delete visually redundant images\n"
            f"(threshold = {THRESHOLD}).\n\nProceed?"
        )
        if not confirm:
            return
        def worker():
            try:
                base_dir = Path(self.terms_path).expanduser().resolve().parent
                summary = dedupe_ranked_images(base_dir)
                t = summary.get("totals", {})
                msg_lines = [
                    "Deduplication completed.",
                    f"Threshold: {THRESHOLD}",
                    f"Scanned:   {t.get('scanned', 0)}",
                    f"Kept:      {t.get('kept', 0)}",
                    f"Deleted:   {t.get('deleted', 0)}",
                ]
                self.root.after(0, lambda: messagebox.showinfo("DeDupe", "\\n".join(msg_lines)))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("DeDupe failed", str(e)))
        threading.Thread(target=worker, daemon=True).start()

    def _existing_norm_names(self) -> set[str]:
        from .persistence import normalize_term
        return {normalize_term(t.name) for t in self.terms}

    def add_terms_from_list(self, names: list[str]) -> tuple[int, list[str]]:
        existing = self._existing_norm_names(); added = 0; skipped: list[str] = []
        for raw in names:
            name = (raw or "").strip()
            if not name: continue
            from .persistence import normalize_term
            norm = normalize_term(name)
            if norm in existing: skipped.append(name); continue
            self.terms.append(Term(name)); existing.add(norm); added += 1
        try: save_terms(self.terms_path, self.terms)
        except Exception: pass
        self._update_progress()
        return added, skipped

    def _open_add_terms(self):
        win = tk.Toplevel(self.root); win.title("Add Terms (one per line)"); win.transient(self.root); win.grab_set()
        frm = ttk.Frame(win, padding=12); frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Paste new terms, one per line. Existing ones will be skipped.").pack(anchor="w", pady=(0,6))
        txt = tk.Text(frm, width=50, height=16); txt.pack(fill=tk.BOTH, expand=True)
        btns = ttk.Frame(frm); btns.pack(fill=tk.X, pady=(8,0))
        def on_add():
            raw = txt.get("1.0","end"); names = [s.strip() for s in raw.splitlines() if s.strip()]
            self.add_terms_from_list(names); win.destroy()
            with self.prefetch_lock: self.prefetch_queue.clear(); self._start_initial_prefetch()
        ttk.Button(btns, text="Add", command=on_add).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0,8))

    def _open_remove_terms(self):
        win = tk.Toplevel(self.root); win.title("Remove Terms"); win.transient(self.root); win.grab_set()
        frm = ttk.Frame(win, padding=12); frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Select one or more terms to remove:").pack(anchor="w")
        lb = tk.Listbox(frm, selectmode=tk.EXTENDED, width=40, height=20); lb.pack(fill=tk.BOTH, expand=True, pady=6)
        sorted_terms = sorted(self.terms, key=lambda t: t.name.casefold())
        for t in sorted_terms: lb.insert(tk.END, f"{t.name}  ({t.rating:.0f}, g={t.games})")
        btns = ttk.Frame(frm); btns.pack(fill=tk.X, pady=(8,0))
        def on_remove():
            sel = lb.curselection()
            if not sel: win.destroy(); return
            to_remove = [sorted_terms[i] for i in sel]
            names = [t.name for t in to_remove]
            self.terms = [t for t in self.terms if t not in to_remove]
            for nm in names:
                try: del self.used[nm]
                except Exception: pass
            try: save_terms(self.terms_path, self.terms); save_used(self.used_path, self.used)
            except Exception: pass
            win.destroy()
            with self.prefetch_lock: self.prefetch_queue.clear(); self._start_initial_prefetch()
            self._update_progress()
        ttk.Button(btns, text="Remove Selected", command=on_remove).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0,8))

    def _open_file(self):
        path = filedialog.askopenfilename(title="Select terms file", filetypes=[("Text/CSV","*.txt *.csv"),("All files","*.*")])
        if not path: return
        terms = load_terms(path)
        if not terms: messagebox.showerror("No Terms","That file is empty."); return
        self.terms_path = path; cfg = load_config(); cfg["last_terms_path"] = path; save_config(cfg)
        self.terms = terms
        self.used_path = used_cache_path(path); self.seen_hashes_path = seen_hashes_path(path)
        self.used = load_used(self.used_path); self.seen_hashes = load_seen_hashes(self.seen_hashes_path)
        self.current_four = []; self.current_urls = [None]*4; self.current_raw_bytes = [None]*4; self.current_pils = [None]*4
        self.recency_map.clear(); self.round_id = 0
        for lbl in self.labels: lbl.delete("all"); draw_title(lbl, "Loading…")
        with self.prefetch_lock: self.prefetch_queue.clear()
        self._update_progress()
        self._start_initial_prefetch()

    def _open_settings(self):
        cfg = cfg_get()

        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.transient(self.root)
        win.grab_set()

        outer = ttk.Frame(win, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(1, weight=1)

        def add_group(title):
            lab = ttk.Label(outer, text=title, font=("Segoe UI", 11, "bold"))
            lab.grid(column=0, row=add_group.r, columnspan=3, sticky="w", pady=(10, 4))
            add_group.r += 1
        add_group.r = 0

        def add_bool(label, key, note=None):
            var = tk.BooleanVar(value=bool(cfg.get(key, SETTINGS_DEFAULTS[key])))
            row = add_group.r
            ttk.Label(outer, text=label).grid(row=row, column=0, sticky="w", padx=(0,10))
            ttk.Checkbutton(outer, variable=var).grid(row=row, column=1, sticky="w")
            if note: ttk.Label(outer, text=note, foreground="#888").grid(row=row, column=2, sticky="w")
            add_group.r += 1
            return var

        def add_int(label, key, rng, note=None):
            val = int(cfg.get(key, SETTINGS_DEFAULTS[key]))
            var = tk.IntVar(value=val)
            row = add_group.r
            ttk.Label(outer, text=label).grid(row=row, column=0, sticky="w", padx=(0,10))
            sp = ttk.Spinbox(outer, from_=rng[0], to=rng[1], increment=1, textvariable=var, width=8)
            sp.grid(row=row, column=1, sticky="w")
            if note: ttk.Label(outer, text=note, foreground="#888").grid(row=row, column=2, sticky="w")
            add_group.r += 1
            return var

        def add_float(label, key, rng, step, note=None):
            val = float(cfg.get(key, SETTINGS_DEFAULTS[key]))
            var = tk.DoubleVar(value=val)
            row = add_group.r
            ttk.Label(outer, text=label).grid(row=row, column=0, sticky="w", padx=(0,10))
            sp = ttk.Spinbox(outer, from_=rng[0], to=rng[1], increment=step, textvariable=var, width=8)
            sp.grid(row=row, column=1, sticky="w")
            if note: ttk.Label(outer, text=note, foreground="#888").grid(row=row, column=2, sticky="w")
            add_group.r += 1
            return var

        add_group("A. Selection")
        v_cooldown = add_int("Cooldown window (1) [0..999]", "cooldown_window", (0, 999), "Lower = can reappear sooner")
        v_recency_cap = add_int("Recency cap factor (2) [1..100]", "recency_cap_factor", (1, 100), "Cap = factor × cooldown")

        add_group("B. Image filtering")
        v_enf_woman = add_bool("Require “woman” filter (4)", "enforce_woman")
        v_face_req = add_bool("Require face detection (5)", "face_required")
        v_face_frac = add_float("Face min size fraction (6) [0.05..0.6]", "face_min_frac", (0.05, 0.6), 0.01, "Higher = closer/larger faces")
        v_face_neigh = add_int("Face min neighbors (7) [1..10]", "face_min_neighbors", (1, 10), "Higher = stricter")
        v_face_scale = add_float("Face scale factor (8) [1.05..1.5]", "face_scale_factor", (1.05, 1.5), 0.01, "Lower = slower but more sensitive")
        v_min_dim = add_int("Minimum image dimension px (9) [0..2000]", "min_dim", (0, 2000))

        add_group("C. Search & networking")
        v_ddg_max = add_int("DuckDuckGo max results (10) [10..200]", "ddg_max_results", (10, 200))
        v_topk = add_int("Candidate top-K (11) [1..50]", "top_sample_k", (1, 50))
        v_timeout = add_int("HTTP timeout seconds (12) [1..60]", "timeout", (1, 60))

        add_group("D. Prefetch & caching")
        v_unique_host = add_bool("Enforce unique host per round (13)", "unique_hosts_per_round", "Avoid all 4 from same site")
        v_avoid_dup = add_bool("Avoid duplicate images (pHash) (14)", "avoid_dup_hashes", "Skip near-duplicates globally")
        v_pref_q = add_int("Prefetch queue max batches (15) [0..6]", "prefetch_max_queue", (0, 6), "Higher = smoother UX, more IO")
        v_use_used = add_bool("Use per-term used-URL cache (16)", "use_used_url_cache", "Remember URLs shown for a term")
        v_use_seen = add_bool("Use global seen-hash cache (17)", "use_seen_hash_cache", "Avoid dupes across sessions")

        add_group("E. Gallery (All ranked images)")
        v_g_batch = add_int("Thumbs 'load more' batch size (24) [4..40]", "gallery_batch_size", (4, 40), "* requires restart of Gallery window")
        v_g_vis = add_int("Scroll prefetch buffer px (25) [200..4000]", "gallery_vis_buffer", (200, 4000), "* requires restart of Gallery window")

        btns = ttk.Frame(outer); btns.grid(row=add_group.r, column=0, columnspan=3, sticky="e", pady=(12,0))
        def on_ok():
            cfg["cooldown_window"] = int(v_cooldown.get())
            cfg["recency_cap_factor"] = int(v_recency_cap.get())

            cfg["enforce_woman"] = bool(v_enf_woman.get())
            cfg["face_required"] = bool(v_face_req.get())
            cfg["face_min_frac"] = float(v_face_frac.get())
            cfg["face_min_neighbors"] = int(v_face_neigh.get())
            cfg["face_scale_factor"] = float(v_face_scale.get())
            cfg["min_dim"] = int(v_min_dim.get())

            cfg["ddg_max_results"] = int(v_ddg_max.get())
            cfg["top_sample_k"] = int(v_topk.get())
            cfg["timeout"] = int(v_timeout.get())

            cfg["unique_hosts_per_round"] = bool(v_unique_host.get())
            cfg["avoid_dup_hashes"] = bool(v_avoid_dup.get())
            cfg["prefetch_max_queue"] = int(v_pref_q.get())
            cfg["use_used_url_cache"] = bool(v_use_used.get())
            cfg["use_seen_hash_cache"] = bool(v_use_seen.get())

            cfg["gallery_batch_size"] = int(v_g_batch.get())
            cfg["gallery_vis_buffer"] = int(v_g_vis.get())

            cfg_save(cfg)

            try:
                core.COOLDOWN_WINDOW = cfg["cooldown_window"]
                images.ENFORCE_WOMAN = cfg["enforce_woman"]
                images.MIN_DIM = int(cfg["min_dim"])
                images.DDG_MAX_RESULTS = int(cfg["ddg_max_results"])
                images.TOP_SAMPLE_K = int(cfg["top_sample_k"])
                images.TIMEOUT = int(cfg["timeout"])
            except Exception:
                pass

            win.destroy()

        ttk.Button(btns, text="OK", command=on_ok).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0,8))

    def _start_initial_prefetch(self):
        def worker():
            batch = self._prefetch_round()
            self.root.after(0, lambda: self._display_batch(batch))
            threading.Thread(target=self._top_up_prefetch_async, daemon=True).start()
        threading.Thread(target=worker, daemon=True).start()

    def _top_up_prefetch_async(self):
        try:
            while True:
                batch = self._prefetch_round()
                max_q = int(cfg_get().get("prefetch_max_queue", 2))
                with self.prefetch_lock:
                    if len(self.prefetch_queue) >= max_q: break
                    self.prefetch_queue.append(batch)
        except Exception:
            pass

    def _consume_prefetch_or_fetch(self) -> PrefetchBatch:
        with self.prefetch_lock:
            batch = self.prefetch_queue.pop(0) if self.prefetch_queue else None
        if batch and len(batch.entries) == 4:
            threading.Thread(target=self._top_up_prefetch_async, daemon=True).start()
            return batch
        return self._prefetch_round()

    def _prefetch_round(self) -> PrefetchBatch:
        cfg = cfg_get()
        if len(self.terms) < 4: return PrefetchBatch(entries=[])

        four = weighted_sample_terms(self.terms, 4, recency_map=self.recency_map)
        extra = self.extra_var.get()
        round_hosts: set[str] = set()

        use_used = cfg.get("use_used_url_cache", True)
        used_sets = [self.used.get(t.name, set()) if use_used else set() for t in four]

        urls: list[str | None] = [None]*4
        datas: list[bytes | None] = [None]*4
        hashes: list[str | None] = [None]*4

        avoid_dup = cfg.get("avoid_dup_hashes", True)
        unique_hosts = cfg.get("unique_hosts_per_round", True)

        for i, term in enumerate(four):
            candidates = ddg_candidates(term.name, extra)
            if candidates:
                for u in candidates:
                    if use_used and u in used_sets[i]: continue
                    hst = host_of(u)
                    if unique_hosts and hst and hst in round_hosts: continue
                    d = download_image_bytes(u, u)
                    if not d: continue
                    if not looks_like_woman(d): continue
                    if cfg.get("face_required", True) and not _has_face_bytes(d): continue
                    hh = phash_bytes(d)
                    if avoid_dup and hh and hh in self.seen_hashes: continue
                    urls[i], datas[i], hashes[i] = u, d, hh
                    round_hosts.add(hst)
                    if use_used:
                        if term.name not in self.used: self.used[term.name] = set()
                        self.used[term.name].add(u)
                    if avoid_dup and hh: self.seen_hashes.add(hh)
                    break

        try:
            exclude = set(t.name for t in four)
            for i in range(4):
                if datas[i] is not None: continue
                alt_list = weighted_sample_terms(self.terms, 1, recency_map=self.recency_map, exclude=exclude)
                alt = alt_list[0] if alt_list else None
                if not alt: continue
                exclude.add(alt.name)
                used_set = self.used.get(alt.name, set()) if use_used else set()
                cand = ddg_candidates(alt.name, extra)
                url = None; data = None; hh = None
                if cand:
                    for u in cand:
                        if use_used and u in used_set: continue
                        hst = host_of(u)
                        if unique_hosts and hst and hst in round_hosts: continue
                        d = download_image_bytes(u, u)
                        if not d: continue
                        if not looks_like_woman(d): continue
                        if cfg.get("face_required", True) and not _has_face_bytes(d): continue
                        hh2 = phash_bytes(d)
                        if avoid_dup and hh2 and hh2 in self.seen_hashes: continue
                        url, data, hh = u, d, hh2
                        round_hosts.add(hst); break
                urls[i], datas[i], hashes[i] = url, data, hh
                if url:
                    if use_used:
                        if alt.name not in self.used: self.used[alt.name] = set()
                        self.used[alt.name].add(url)
                    if avoid_dup and hh: self.seen_hashes.add(hh)
                    four[i] = alt
        except Exception:
            pass

        try:
            if use_used: save_used(self.used_path, self.used)
            if avoid_dup: save_seen_hashes(self.seen_hashes_path, self.seen_hashes)
        except Exception:
            pass

        cap = int(cfg.get("recency_cap_factor", 10)) * int(core.COOLDOWN_WINDOW or 1)
        for k in list(self.recency_map.keys()):
            self.recency_map[k] = min(self.recency_map[k] + 1, cap)
        for t in four: self.recency_map[t.name] = 0
        self.round_id += 1

        entries = [PrefetchEntry(term=four[i], url=urls[i], data=datas[i]) for i in range(4)]
        return PrefetchBatch(entries=entries)

    def _display_batch(self, batch: PrefetchBatch):
        if not batch.entries or len(batch.entries) != 4:
            messagebox.showerror("Need more terms","At least 4 terms are required."); return

        self.current_four = [e.term for e in batch.entries]
        self.current_urls = [e.url for e in batch.entries]
        self.current_raw_bytes = [e.data for e in batch.entries]
        self.current_pils = [None]*4

        for i, entry in enumerate(batch.entries):
            term = entry.term
            lbl = self.labels[i]
            draw_title(lbl, term.name)
            if DEBUG_OVERLAY:
                host = host_of(entry.url) if entry.url else "-"
                draw_debug(lbl, host)
            data = entry.data
            if data:
                try:
                    pil = Image.open(io.BytesIO(data)).convert("RGB")
                    self.current_pils[i] = pil
                    render_to_canvas(lbl, pil)
                    draw_title(lbl, term.name)
                    if DEBUG_OVERLAY:
                        draw_debug(lbl, host_of(entry.url) if entry.url else "-")
                except Exception:
                    self.current_pils[i] = None
                    canvas_show_error(lbl, "Failed to load"); draw_title(lbl, term.name)
            else:
                self.current_pils[i] = None
                canvas_show_error(lbl, "No image"); draw_title(lbl, term.name)

        self._update_progress()

    def _calc_progress(self) -> tuple[float, float]:
        if not self.terms:
            return 0.0, 0.0
        sigmas = [t.sigma for t in self.terms]
        if not sigmas:
            return 0.0, 0.0
        sigma0 = max(sigmas) if max(sigmas) > 10 else 3.0
        total = 0.0
        for s in sigmas:
            total += max(0.0, min(1.0, 1.0 - (s / sigma0)))
        progress = total / max(1, len(sigmas))
        avg_sigma = sum(sigmas) / max(1, len(sigmas))
        return progress, avg_sigma

    def _update_progress(self):
        progress, avg_sigma = self._calc_progress()
        pct = int(round(progress * 100))
        self.progress_var.set(pct)
        if self.progress_label is not None:
            self.progress_label.config(text=f"Certainty: {pct}% (avg σ={avg_sigma:.2f})")

    def _on_canvas_resize(self, idx: int):
        try:
            pil = self.current_pils[idx]; cvs = self.labels[idx]
            if pil is not None: render_to_canvas(cvs, pil)
            if self.current_four and idx < len(self.current_four):
                draw_title(cvs, self.current_four[idx].name)
                if DEBUG_OVERLAY:
                    draw_debug(cvs, host_of(self.current_urls[idx]) if self.current_urls[idx] else "-")
        except Exception: pass

    def _on_click(self, idx: int):
        if not self.current_four or idx >= len(self.current_four): return
        winner = self.current_four[idx]; losers = [t for i, t in enumerate(self.current_four) if i != idx]
        self._save_winning_image_uncropped(idx, winner)
        elo_update(winner, losers)
        try: save_terms(self.terms_path, self.terms)
        except Exception: pass
        self._update_progress()
        batch = self._consume_prefetch_or_fetch()
        self._display_batch(batch)

    def _on_other(self, idx: int):
        if not self.current_four or idx >= len(self.current_four): return
        old_term = self.current_four[idx]; extra = self.extra_var.get()
        def worker():
            cfg = cfg_get()
            use_used = cfg.get("use_used_url_cache", True)
            avoid_dup = cfg.get("avoid_dup_hashes", True)
            unique_hosts = cfg.get("unique_hosts_per_round", True)

            if use_used:
                self.used[old_term.name] = set()
                try: save_used(self.used_path, self.used)
                except Exception: pass

            exclude = {t.name for i, t in enumerate(self.current_four) if i != idx}
            alt_list = weighted_sample_terms(self.terms, 1, recency_map=self.recency_map, exclude=exclude)
            new_term = alt_list[0] if alt_list else None
            if new_term is None: return

            other_hosts = {host_of(u) for i, u in enumerate(self.current_urls) if i != idx and u}
            url = None; data = None; hh = None
            candidates = ddg_candidates(new_term.name, extra)

            def try_candidates(cands, respect_hosts=True):
                nonlocal url, data, hh
                if not cands: return False
                used_set = self.used.get(new_term.name, set()) if use_used else set()
                for u in cands:
                    if use_used and u in used_set: continue
                    if respect_hosts and unique_hosts:
                        hst = host_of(u)
                        if hst and hst in other_hosts: continue
                    d = download_image_bytes(u, u)
                    if not d: continue
                    if not looks_like_woman(d): continue
                    if cfg.get("face_required", True) and not _has_face_bytes(d): continue
                    hhash = phash_bytes(d)
                    if avoid_dup and hhash and hhash in self.seen_hashes: continue
                    url, data, hh = u, d, hhash
                    return True
                return False

            ok = try_candidates(candidates, True)
            if not ok: ok = try_candidates(candidates, False)

            if url and use_used:
                if new_term.name not in self.used: self.used[new_term.name] = set()
                self.used[new_term.name].add(url)
                try: save_used(self.used_path, self.used)
                except Exception: pass
            if hh and cfg.get("avoid_dup_hashes", True):
                self.seen_hashes.add(hh)
                try: save_seen_hashes(self.seen_hashes_path, self.seen_hashes)
                except Exception: pass

            def apply():
                self.current_four[idx] = new_term
                self.current_urls[idx] = url
                self.current_raw_bytes[idx] = data
                lbl = self.labels[idx]
                draw_title(lbl, new_term.name)
                if DEBUG_OVERLAY:
                    draw_debug(lbl, host_of(url) if url else "-")
                if data:
                    try:
                        pil = Image.open(io.BytesIO(data)).convert("RGB")
                        self.current_pils[idx] = pil
                        render_to_canvas(lbl, pil)
                        draw_title(lbl, new_term.name)
                        if DEBUG_OVERLAY:
                            draw_debug(lbl, host_of(url) if url else "-")
                    except Exception:
                        self.current_pils[idx] = None
                self._update_progress()
            self.root.after(0, apply)
        threading.Thread(target=worker, daemon=True).start()

    def _save_winning_image_uncropped(self, idx: int, term: Term):
        data = self.current_raw_bytes[idx]
        if not data: return
        term_dir = os.path.join(os.path.dirname(os.path.abspath(self.terms_path)), "ranked_images", slugify(term.name))
        os.makedirs(term_dir, exist_ok=True)
        try:
            pil = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = f"{slugify(term.name)}_{int(round(term.rating))}_{ts}.jpg"
        path = os.path.join(term_dir, fname)
        try: pil.save(path, format="JPEG", quality=90)
        except Exception: pass

def main():
    arg_path = sys.argv[1] if len(sys.argv) >= 2 else None
    root = tk.Tk(); root.geometry("1300x1000"); root.minsize(900, 700)
    App(root, arg_path); root.mainloop()

class GalleryWindow(tk.Toplevel):
    BATCH_SIZE = SETTINGS_DEFAULTS["gallery_batch_size"]
    VIS_BUFFER = SETTINGS_DEFAULTS["gallery_vis_buffer"]

    def __init__(self, app: "App"):
        super().__init__(app.root)
        gcfg = cfg_get()
        type(self).BATCH_SIZE = int(gcfg.get("gallery_batch_size", SETTINGS_DEFAULTS["gallery_batch_size"]))
        type(self).VIS_BUFFER = int(gcfg.get("gallery_vis_buffer", SETTINGS_DEFAULTS["gallery_vis_buffer"]))

        self.title("All Ranked Images"); self.geometry("1200x800")
        self.app = app
        self._images: list[ImageTk.PhotoImage] = []
        self.thumb_height_var = tk.IntVar(value=200)
        self.sort_mode = tk.StringVar(value="rating_desc")
        self._thumb_cache: dict[tuple[str, int], ImageTk.PhotoImage] = {}
        self._row_widgets: list[tuple[ttk.Frame, dict]] = []

        container = ttk.Frame(self); container.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(container); top.pack(fill=tk.X, padx=10, pady=(8,0))
        ttk.Label(top, text="Height:").pack(side=tk.LEFT)
        for s in [50,150,200,250,300]:
            ttk.Button(top, text=str(s), command=lambda size=s: self._set_height(size)).pack(side=tk.LEFT, padx=2)
        ttk.Label(top, text="   Sort:").pack(side=tk.LEFT, padx=(12,0))
        sort_bar = ttk.Frame(top); sort_bar.pack(side=tk.LEFT)
        def add_sort(label, key):
            ttk.Button(sort_bar, text=label, command=lambda k=key: self._set_sort(k)).pack(side=tk.LEFT, padx=2)
        add_sort("Alphabetical", "name_asc")
        add_sort("Highest Rating", "rating_desc")
        add_sort("Lowest Rating", "rating_asc")
        add_sort("Games High", "games_desc")
        add_sort("Games Low", "games_asc")
        add_sort("Sigma High", "sigma_desc")
        add_sort("Sigma Low", "sigma_asc")
        ttk.Button(top, text="Refresh", command=self._refresh).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(container, highlightthickness=0, bg="#111")
        vbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y); self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = ttk.Frame(self.canvas); self.inner_id = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_frame_configure); self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_scroll, add="+")
        self.canvas.bind("<Button-4>", self._on_scroll); self.canvas.bind("<Button-5>", self._on_scroll)

        self._build_rows_shells()

    def _set_height(self, size: int):
        self.thumb_height_var.set(size); self._refresh()

    def _set_sort(self, mode: str):
        self.sort_mode.set(mode); self._refresh()

    def _on_frame_configure(self, event=None): self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.inner_id, width=event.width); self._ensure_visible_rows()

    def _on_scroll(self, event=None): self.after(10, self._ensure_visible_rows())

    def _refresh(self):
        for w in list(self.inner.children.values()):
            try: w.destroy()
            except Exception: pass
        self._images.clear(); self._row_widgets.clear()
        self._build_rows_shells()

    def _sorted_terms(self):
        key = self.sort_mode.get()
        terms = list(self.app.terms)
        if key == "name_asc": terms.sort(key=lambda t: t.name.casefold())
        elif key == "rating_asc": terms.sort(key=lambda t: t.rating)
        elif key == "rating_desc": terms.sort(key=lambda t: t.rating, reverse=True)
        elif key == "games_asc": terms.sort(key=lambda t: t.games)
        elif key == "games_desc": terms.sort(key=lambda t: t.games, reverse=True)
        elif key == "sigma_asc": terms.sort(key=lambda t: t.sigma)
        elif key == "sigma_desc": terms.sort(key=lambda t: t.sigma, reverse=True)
        else: terms.sort(key=lambda t: t.rating, reverse=True)
        return terms

    def _build_rows_shells(self):
        base_dir = os.path.dirname(os.path.abspath(self.app.terms_path))
        ranked_dir = os.path.join(base_dir, "ranked_images")
        terms_sorted = self._sorted_terms()

        for term in terms_sorted:
            term_dir = os.path.join(ranked_dir, slugify(term.name))
            if not os.path.isdir(term_dir): continue
            files = sorted([f for f in Path(term_dir).glob("*.jpg")], key=lambda p: p.name, reverse=True)
            if not files: continue

            row_frame = ttk.Frame(self.inner); row_frame.pack(fill=tk.X, padx=10, pady=(12, 4))
            header = ttk.Frame(row_frame); header.pack(fill=tk.X)
            stats = f"rating {term.rating:.0f} | games {term.games} | sigma {term.sigma:.1f} | images {len(files)}"
            ttk.Label(header, text=f"{term.name} — {stats}", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, anchor="w")
            more_btn_holder = ttk.Frame(header); more_btn_holder.pack(side=tk.RIGHT)
            thumbs_wrap = ttk.Frame(row_frame); thumbs_wrap.pack(fill=tk.X, padx=0, pady=(4, 0))

            meta = {"term": term, "dir": term_dir, "files": files, "wrap": thumbs_wrap,
                    "start": 0, "built": False, "more_btn_holder": more_btn_holder}
            self._row_widgets.append((row_frame, meta))

        self.after(0, self._ensure_visible_rows)

    def _ensure_visible_rows(self):
        try:
            top = int(self.canvas.canvasy(0)); bottom = int(self.canvas.canvasy(self.canvas.winfo_height()))
        except Exception: return
        buf = int(cfg_get().get("gallery_vis_buffer", 800))
        top -= buf; bottom += buf
        for row_frame, meta in self._row_widgets:
            try:
                ry0 = row_frame.winfo_y(); ry1 = ry0 + row_frame.winfo_height()
            except Exception: continue
            if ry1 < top or ry0 > bottom: continue
            if not meta.get("built"): self._build_row_content(meta)

    def _build_row_content(self, meta: dict):
        files = meta["files"]; wrap: ttk.Frame = meta["wrap"]
        start = 0; page = int(cfg_get().get("gallery_batch_size", 8)); subset = files[start:start+page]
        self._render_thumbs_async(wrap, subset)
        meta["start"] = start + page; meta["built"] = True
        if len(files) > page:
            holder: ttk.Frame = meta["more_btn_holder"]
            btn = ttk.Button(holder, text="Load more…"); btn.pack()
            btn.configure(command=lambda m=meta, b=btn: self._on_load_more(m, b))

    def _on_load_more(self, meta: dict, btn: ttk.Button | None):
        files = meta["files"]; start = meta.get("start", 0); page = int(cfg_get().get("gallery_batch_size", 8))
        subset = files[start:start+page]
        self._render_thumbs_async(meta["wrap"], subset)
        start += page; meta["start"] = start
        if start >= len(files) and btn is not None:
            try: btn.destroy()
            except Exception: pass

    def _render_thumbs_async(self, parent: ttk.Frame, file_list: list[Path]):
        if not file_list: return
        target_h = int(self.thumb_height_var.get())
        def worker():
            items = []
            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = Image.ANTIALIAS  # type: ignore
            for fp in file_list:
                try:
                    key = (str(fp), target_h)
                    if key in self._thumb_cache:
                        tkimg = self._thumb_cache[key]; items.append((fp, tkimg)); continue
                    pil = Image.open(fp).convert("RGB")
                    ratio = target_h / max(1, pil.height)
                    new_w = max(1, int(round(pil.width * ratio)))
                    thumb_img = pil.resize((new_w, target_h), resample=resample)
                    tkimg = ImageTk.PhotoImage(thumb_img)
                    if len(self._thumb_cache) > 800:
                        try: self._thumb_cache.pop(next(iter(self._thumb_cache)))
                        except Exception: pass
                    self._thumb_cache[key] = tkimg
                    items.append((fp, tkimg))
                except Exception:
                    continue
            def apply():
                c = 0
                for fp, tkimg in items:
                    lbl = tk.Label(parent, image=tkimg, bg="#111", cursor="hand2")
                    lbl.image = tkimg
                    lbl.bind("<Button-1>", lambda e, path=str(fp): self._open_fullscreen(path))
                    lbl.grid(row=0, column=c, padx=6, pady=6); c += 1
                    self._images.append(tkimg)
            self.after(0, apply)
        threading.Thread(target=worker, daemon=True).start()

    def _open_fullscreen(self, path: str):
        FullscreenImageWindow(self, path)

class FullscreenImageWindow(tk.Toplevel):
    def __init__(self, parent: GalleryWindow, path: str):
        super().__init__(parent)
        self.title(os.path.basename(path)); self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda e: self.destroy()); self.bind("<Button-1>", lambda e: self.destroy())
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0); self.canvas.pack(fill=tk.BOTH, expand=True)
        self._pil = Image.open(path).convert("RGB"); self._tk = None
        self.bind("<Configure>", self._on_resize); self._on_resize()
    def _on_resize(self, event=None):
        w = self.winfo_width(); h = self.winfo_height()
        if w <= 2 or h <= 2: return
        try: resample = Image.Resampling.LANCZOS
        except Exception: resample = Image.ANTIALIAS  # type: ignore
        fitted = ImageOps.contain(self._pil, (w, h), method=resample)
        self._tk = ImageTk.PhotoImage(fitted)
        self.canvas.delete("img"); self.canvas.create_image(w//2, h//2, image=self._tk, anchor="center", tags="img")
