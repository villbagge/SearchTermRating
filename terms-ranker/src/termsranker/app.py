from __future__ import annotations
import io, os, sys, threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageOps

from .core import Term, elo_update, weighted_sample_terms, COOLDOWN_WINDOW, COOLDOWN_MIN_FACTOR
from .persistence import (
    load_terms, save_terms, used_cache_path, seen_hashes_path, load_used, save_used,
    load_seen_hashes, save_seen_hashes, slugify
)
from .images import (
    ddg_candidates, download_image_bytes, looks_like_woman, phash_bytes, host_of,
    HOST_DIVERSITY_IN_FALLBACK
)

PADDING = 12
THUMB_SIZE = (520, 520)
SELF_TEST_EACH_BATCH = False  # keep off by default

def canvas_show_message(canvas: "tk.Canvas", msg: str, tag: str = "status", color: str = "#ddd"):
    try:
        canvas.delete(tag)
        w = max(1, int(canvas.winfo_width())); h = max(1, int(canvas.winfo_height()))
        canvas.create_text(10, 10, text=msg, anchor="nw", fill=color, tags=tag)
    except Exception:
        pass

def canvas_show_error(canvas: "tk.Canvas", msg: str):
    try:
        canvas.delete("img"); canvas.delete("err"); canvas.delete("status")
        w = max(1, int(canvas.winfo_width())); h = max(1, int(canvas.winfo_height()))
        canvas.create_text(w//2, h//2, text=msg, fill="#ddd", tags="err")
    except Exception:
        pass

def render_to_canvas(canvas: "tk.Canvas", pil_img: Image.Image):
    try:
        w = max(1, int(canvas.winfo_width())); h = max(1, int(canvas.winfo_height()))
    except Exception:
        w = h = 0
    if w < 10 or h < 10:
        w, h = THUMB_SIZE
    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = Image.ANTIALIAS  # type: ignore
    fitted = ImageOps.contain(pil_img.convert("RGB"), (w, h), method=resample)
    tkimg = ImageTk.PhotoImage(fitted)
    canvas.delete("img"); canvas.delete("status")
    canvas.create_image(w//2, h//2, image=tkimg, anchor="center", tags="img")
    canvas.image = tkimg

class PrefetchBatch:
    __slots__ = ("terms", "urls", "bytes_list")
    def __init__(self, terms, urls, bytes_list):
        self.terms = terms
        self.urls = urls
        self.bytes_list = bytes_list

class App:
    def __init__(self, root: tk.Tk, terms_path: str):
        self.root = root
        self.terms_path = terms_path
        self.terms: list[Term] = load_terms(terms_path)
        if not self.terms:
            messagebox.showerror("No Terms", "The file is empty. Add terms (one per line) or 'term,rating[,games,sigma]'.")
            sys.exit(1)

        # caches
        self.used_path = used_cache_path(terms_path)
        self.used = load_used(self.used_path)
        self.seen_hashes_path = seen_hashes_path(terms_path)
        self.seen_hashes = load_seen_hashes(self.seen_hashes_path)

        # selection recency tracking
        self.round_id = 0
        self.recency_map: dict[str, int] = {}  # name -> rounds since last shown

        # UI state
        self.extra_var = tk.StringVar(value="")
        self.status = tk.StringVar(value="")  # keep but don't spam with hints
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

        # --- UI ---
        self.root.title("Terms Ranker")
        top = ttk.Frame(self.root, padding=PADDING); top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Click the best image").pack(side=tk.LEFT)
        ttk.Label(top, text="  |  Extra terms:").pack(side=tk.LEFT, padx=(8,2))
        self.extra_entry = ttk.Entry(top, textvariable=self.extra_var, width=30); self.extra_entry.pack(side=tk.LEFT, padx=(0,10))

        ttk.Button(top, text="Add Terms…", command=self._open_add_terms).pack(side=tk.RIGHT, padx=(0,8))
        ttk.Button(top, text="Remove Terms…", command=self._open_remove_terms).pack(side=tk.RIGHT, padx=(0,8))
        ttk.Button(top, text="Open Terms File", command=self._open_file).pack(side=tk.RIGHT)
        ttk.Button(top, text="Settings…", command=self._open_settings).pack(side=tk.RIGHT, padx=(0,8))
        ttk.Button(top, text="Gallery", command=self._open_gallery).pack(side=tk.RIGHT, padx=(0,8))

        statusbar = ttk.Label(self.root, textvariable=self.status, anchor="w")
        statusbar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=6)

        grid = ttk.Frame(self.root); grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        try: grid.pack_propagate(False)
        except Exception: pass

        for i in range(4):
            slot = ttk.Frame(grid, padding=8)
            r, c = divmod(i, 2)
            slot.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
            slot.rowconfigure(0, weight=1); slot.columnconfigure(0, weight=1)

            cvs = tk.Canvas(slot, highlightthickness=0, bg="#222")
            cvs.grid(row=0, column=0, sticky="nsew")
            # No "Loading..."—will show term name per round
            cvs.bind("<Configure>", lambda e, idx=i: self._on_canvas_resize(idx))

            row2 = ttk.Frame(slot); row2.grid(row=1, column=0, sticky="ew", pady=(6, 0))
            row2.columnconfigure(0, weight=1)

            # Remove captions entirely: keep a placeholder but don't display text
            cap = ttk.Label(row2, text="", anchor="w")
            cap.grid(row=0, column=0, sticky="w")
            other_btn = ttk.Button(row2, text="OTHER", width=8, command=lambda idx=i: self._on_other(idx))
            other_btn.grid(row=0, column=1, sticky="e", padx=(6, 0))

            cvs.bind("<Button-1>", lambda e, idx=i: self._on_click(idx))

            self.frames.append(slot); self.labels.append(cvs); self.current_pils.append(None)
            self.captions.append(cap); self.other_btns.append(other_btn)

        for r in range(2): grid.rowconfigure(r, weight=1, uniform='gallery')
        for c in range(2): grid.columnconfigure(c, weight=1, uniform='gallery')

        self._start_initial_prefetch()

    # ---- Add / Remove ----
    def _existing_norm_names(self) -> set[str]:
        from .persistence import normalize_term
        return {normalize_term(t.name) for t in self.terms}

    def add_terms_from_list(self, names: list[str]) -> tuple[int, list[str]]:
        existing = self._existing_norm_names()
        added = 0; skipped: list[str] = []
        for raw in names:
            name = (raw or "").strip()
            if not name: continue
            from .persistence import normalize_term
            norm = normalize_term(name)
            if norm in existing:
                skipped.append(name); continue
            self.terms.append(Term(name)); existing.add(norm); added += 1
        try: save_terms(self.terms_path, self.terms)
        except Exception: pass
        return added, skipped

    def _open_add_terms(self):
        win = tk.Toplevel(self.root); win.title("Add Terms (one per line)"); win.transient(self.root); win.grab_set()
        frm = ttk.Frame(win, padding=12); frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Paste new terms, one per line. Existing ones will be skipped.").pack(anchor="w", pady=(0,6))
        txt = tk.Text(frm, width=50, height=16); txt.pack(fill=tk.BOTH, expand=True)
        btns = ttk.Frame(frm); btns.pack(fill=tk.X, pady=(8,0))
        def on_add():
            raw = txt.get("1.0", "end"); names = [s.strip() for s in raw.splitlines() if s.strip()]
            added, skipped = self.add_terms_from_list(names)
            msg = f"Added {added} term(s)." + (f" Skipped {len(skipped)} duplicate(s)." if skipped else "")
            self.status.set(msg); win.destroy()
            with self.prefetch_lock: self.prefetch_queue.clear()
            self._start_initial_prefetch()
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
            selected_indices = lb.curselection()
            if not selected_indices: win.destroy(); return
            to_remove = [sorted_terms[i] for i in selected_indices]
            removed_names = [t.name for t in to_remove]
            self.terms = [t for t in self.terms if t not in to_remove]
            used = self.used
            for name in removed_names:
                if name in used:
                    try: del used[name]
                    except Exception: pass
            try:
                save_terms(self.terms_path, self.terms); save_used(self.used_path, used)
            except Exception: pass
            self.status.set(f"Removed {len(to_remove)} term(s)."); win.destroy()
            with self.prefetch_lock: self.prefetch_queue.clear()
            self._start_initial_prefetch()
        ttk.Button(btns, text="Remove Selected", command=on_remove).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0,8))

    # ---- File / Settings / Gallery ----
    def _open_file(self):
        path = filedialog.askopenfilename(title="Select terms file", filetypes=[("Text/CSV", "*.txt *.csv"), ("All files", "*.*")])
        if not path: return
        terms = load_terms(path)
        if not terms:
            messagebox.showerror("No Terms", "That file is empty. Add terms (one per line) or 'term,rating[,games,sigma]'."); return
        self.terms_path = path; self.terms = terms
        self.used_path = used_cache_path(path); self.seen_hashes_path = seen_hashes_path(path)
        self.used = load_used(self.used_path); self.seen_hashes = load_seen_hashes(self.seen_hashes_path)
        self.current_four = []; self.current_urls = [None]*4; self.current_raw_bytes = [None]*4; self.current_photos = [None]*4
        self.recency_map.clear(); self.round_id = 0
        for lbl in self.labels:
            lbl.delete("all")
        with self.prefetch_lock: self.prefetch_queue.clear()
        self._start_initial_prefetch()
        self.status.set(f"Loaded {len(self.terms)} terms from {os.path.basename(path)}")

    def _open_settings(self):
        import termsranker.core as core
        import termsranker.images as images
        win = tk.Toplevel(self.root); win.title("Settings"); win.transient(self.root); win.grab_set()
        frm = ttk.Frame(win, padding=12); frm.pack(fill=tk.BOTH, expand=True)
        # variables
        enforce_woman_var = tk.BooleanVar(value=images.ENFORCE_WOMAN)
        strict_err_var   = tk.BooleanVar(value=images.DEEPFACE_STRICT_ON_ERROR)
        enforce_detect_var = tk.BooleanVar(value=images.DEEPFACE_ENFORCE_DETECTION)
        host_diverse_var = tk.BooleanVar(value=images.HOST_DIVERSITY_IN_FALLBACK)
        woman_thresh_var = tk.DoubleVar(value=images.WOMAN_PROB_THRESHOLD)
        safesearch_var = tk.StringVar(value=images.SAFESEARCH)
        ddg_max_var = tk.IntVar(value=images.DDG_MAX_RESULTS)
        topk_var = tk.IntVar(value=images.TOP_SAMPLE_K)
        min_dim_var = tk.IntVar(value=images.MIN_DIM)
        timeout_var = tk.IntVar(value=images.TIMEOUT)
        unranked_boost_var = tk.DoubleVar(value=core.UNRANKED_BOOST * 100.0)
        highspread_boost_var = tk.DoubleVar(value=core.HIGH_SPREAD_BOOST * 100.0)
        spread_thresh_var = tk.DoubleVar(value=core.HIGH_SPREAD_THRESHOLD)
        cooldown_window_var = tk.IntVar(value=core.COOLDOWN_WINDOW)
        cooldown_min_factor_var = tk.DoubleVar(value=core.COOLDOWN_MIN_FACTOR)

        def add_row(r, label, widget):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3)
            widget.grid(row=r, column=1, sticky="ew", pady=3)
        frm.columnconfigure(1, weight=1)
        r = 0
        add_row(r, "Require woman filter", ttk.Checkbutton(frm, variable=enforce_woman_var)); r+=1
        add_row(r, "Woman prob threshold", ttk.Entry(frm, textvariable=woman_thresh_var)); r+=1
        add_row(r, "DeepFace: enforce detection", ttk.Checkbutton(frm, variable=enforce_detect_var)); r+=1
        add_row(r, "DeepFace: strict on errors", ttk.Checkbutton(frm, variable=strict_err_var)); r+=1
        add_row(r, "DuckDuckGo SafeSearch", ttk.Combobox(frm, textvariable=safesearch_var, values=["off","moderate","strict"], state="readonly")); r+=1
        add_row(r, "DDG max results", ttk.Entry(frm, textvariable=ddg_max_var)); r+=1
        add_row(r, "Top sample K", ttk.Entry(frm, textvariable=topk_var)); r+=1
        add_row(r, "Minimum image dim", ttk.Entry(frm, textvariable=min_dim_var)); r+=1
        add_row(r, "HTTP timeout (s)", ttk.Entry(frm, textvariable=timeout_var)); r+=1
        add_row(r, "Fallback: enforce host diversity", ttk.Checkbutton(frm, variable=host_diverse_var)); r+=1
        sep = ttk.Separator(frm, orient="horizontal"); sep.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(8,6)); r+=1
        add_row(r, "Unranked boost (%)", ttk.Entry(frm, textvariable=unranked_boost_var)); r+=1
        add_row(r, "High-spread boost (%)", ttk.Entry(frm, textvariable=highspread_boost_var)); r+=1
        add_row(r, "High-spread threshold (0–1)", ttk.Entry(frm, textvariable=spread_thresh_var)); r+=1
        sep2 = ttk.Separator(frm, orient="horizontal"); sep2.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(8,6)); r+=1
        add_row(r, "Cooldown window (rounds)", ttk.Entry(frm, textvariable=cooldown_window_var)); r+=1
        add_row(r, "Cooldown min factor (0–1)", ttk.Entry(frm, textvariable=cooldown_min_factor_var)); r+=1
        btns = ttk.Frame(frm); btns.grid(row=r, column=0, columnspan=2, pady=(10,0), sticky="e")
        def apply_and_close():
            images.ENFORCE_WOMAN = bool(enforce_woman_var.get())
            try: images.WOMAN_PROB_THRESHOLD = float(woman_thresh_var.get())
            except Exception: pass
            images.DEEPFACE_ENFORCE_DETECTION = bool(enforce_detect_var.get())
            images.DEEPFACE_STRICT_ON_ERROR = bool(strict_err_var.get())
            images.SAFESEARCH = str(safesearch_var.get())
            try: images.DDG_MAX_RESULTS = max(10, int(ddg_max_var.get()))
            except Exception: pass
            try: images.TOP_SAMPLE_K = max(1, int(topk_var.get()))
            except Exception: pass
            try: images.MIN_DIM = max(0, int(min_dim_var.get()))
            except Exception: pass
            try: images.TIMEOUT = max(1, int(timeout_var.get()))
            except Exception: pass
            try: core.UNRANKED_BOOST = max(0.0, float(unranked_boost_var.get()) / 100.0)
            except Exception: pass
            try: core.HIGH_SPREAD_BOOST = max(0.0, float(highspread_boost_var.get()) / 100.0)
            except Exception: pass
            try: core.HIGH_SPREAD_THRESHOLD = max(0.0, min(1.0, float(spread_thresh_var.get())))
            except Exception: pass
            try: core.COOLDOWN_WINDOW = max(0, int(cooldown_window_var.get()))
            except Exception: pass
            try: core.COOLDOWN_MIN_FACTOR = max(0.0, min(1.0, float(cooldown_min_factor_var.get())))
            except Exception: pass
            win.destroy()
        ttk.Button(btns, text="OK", command=apply_and_close).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=(0,8))

    def _open_gallery(self):
        try: GalleryWindow(self)
        except Exception: pass

    # ---- Prefetch orchestration ----
    def _start_initial_prefetch(self):
        def worker():
            for _ in range(2):
                try: batch = self._prefetch_round()
                except Exception: batch = PrefetchBatch([], [], [])
                with self.prefetch_lock: self.prefetch_queue.append(batch)
            first = None
            with self.prefetch_lock:
                if self.prefetch_queue: first = self.prefetch_queue.pop(0)
            if first is None: first = PrefetchBatch([], [], [])
            self.root.after(0, lambda: self._display_batch(first))
            threading.Thread(target=self._top_up_prefetch_async, daemon=True).start()
        threading.Thread(target=worker, daemon=True).start()

    def _top_up_prefetch_async(self):
        try:
            while True:
                batch = self._prefetch_round()
                with self.prefetch_lock:
                    if len(self.prefetch_queue) >= 2: break
                    self.prefetch_queue.append(batch)
        except Exception:
            pass

    def _consume_prefetch_or_fetch(self) -> PrefetchBatch:
        with self.prefetch_lock:
            batch = self.prefetch_queue.pop(0) if self.prefetch_queue else None
        if batch is not None:
            threading.Thread(target=self._top_up_prefetch_async, daemon=True).start()
            return batch
        return self._prefetch_round()

    # ---- Candidate picking & downloading ----
    def _prefetch_round(self) -> PrefetchBatch:
        if len(self.terms) < 4: return PrefetchBatch([], [], [])
        # sample 4 with recency cooldown
        four = weighted_sample_terms(self.terms, 4, recency_map=self.recency_map)
        # Show term names immediately on each canvas
        for i, term in enumerate(four):
            canvas_show_message(self.labels[i], term.name, tag="status")
        extra = self.extra_var.get()
        urls = [None]*4; bytes_list = [None]*4
        round_hosts: set[str] = set()
        used_sets = [self.used.get(t.name, set()) for t in four]
        for i, term in enumerate(four):
            candidates = ddg_candidates(term.name, extra)
            url = None; data = None; h = None
            if candidates:
                for u in candidates:
                    if u in used_sets[i]: continue
                    hst = host_of(u)
                    if hst and hst in round_hosts: continue
                    d = download_image_bytes(u, u)
                    if not d: continue
                    if not looks_like_woman(d): continue
                    hh = phash_bytes(d)
                    if hh and hh in self.seen_hashes: continue
                    url, data, h = u, d, hh; break
            urls[i] = url; bytes_list[i] = data
            if url:
                round_hosts.add(host_of(url))
                if four[i].name not in self.used: self.used[four[i].name] = set()
                self.used[four[i].name].add(url)
                if h: self.seen_hashes.add(h)
        # Second pass for failed slots (try different terms)
        try:
            exclude = set(t.name for t in four)
            for i in range(4):
                if bytes_list[i] is not None: continue
                alt_list = weighted_sample_terms(self.terms, 1, recency_map=self.recency_map, exclude=exclude)
                alt = alt_list[0] if alt_list else None
                if not alt: continue
                exclude.add(alt.name)
                # update canvas text for replaced term
                canvas_show_message(self.labels[i], alt.name, tag="status")
                used_set = self.used.get(alt.name, set())
                url = None; data = None; h = None
                candidates = ddg_candidates(alt.name, extra)
                if candidates:
                    for u in candidates:
                        if u in used_set: continue
                        hst = host_of(u)
                        if hst and hst in round_hosts: continue
                        d = download_image_bytes(u, u)
                        if not d: continue
                        if not looks_like_woman(d): continue
                        hh = phash_bytes(d)
                        if hh and hh in self.seen_hashes: continue
                        url, data, h = u, d, hh; break
                urls[i] = url; bytes_list[i] = data
                if url:
                    round_hosts.add(host_of(url))
                    if alt.name not in self.used: self.used[alt.name] = set()
                    self.used[alt.name].add(url)
                    if h: self.seen_hashes.add(h)
                    four[i] = alt
        except Exception:
            pass
        # persist caches (best-effort)
        try:
            save_used(self.used_path, self.used)
            save_seen_hashes(self.seen_hashes_path, self.seen_hashes)
        except Exception:
            pass
        # update recency map: increment all; set 0 for the ones shown this round
        for k in list(self.recency_map.keys()):
            self.recency_map[k] = min(self.recency_map[k] + 1, 10 * COOLDOWN_WINDOW if COOLDOWN_WINDOW else 999999)
        for t in four:
            self.recency_map[t.name] = 0
        self.round_id += 1
        return PrefetchBatch(four, urls, bytes_list)

    # ---- Display ----
    def _display_batch(self, batch: PrefetchBatch):
        if len(batch.terms) < 4:
            messagebox.showerror("Need more terms", "At least 4 terms are required."); return
        self.current_four = batch.terms
        self.current_urls = batch.urls
        self.current_photos = [None]*4
        self.current_raw_bytes = [None]*4
        for i in range(4):
            data = batch.bytes_list[i]; lbl = self.labels[i]
            if data:
                try:
                    pil = Image.open(io.BytesIO(data)).convert("RGB")
                    self.current_raw_bytes[i] = data; self.current_urls[i] = batch.urls[i]; self.current_pils[i] = pil
                    render_to_canvas(lbl, pil)
                    try: self.root.after_idle(lambda idx=i: self._on_canvas_resize(idx))
                    except Exception: pass
                except Exception:
                    canvas_show_error(lbl, "Failed to load"); self.current_pils[i] = None
            else:
                canvas_show_error(lbl, "Failed to load"); lbl.image = None; self.current_pils[i] = None
        # Do not write captions under images and no hint in status
        for i in range(4):
            self.captions[i].configure(text="")
        # self.status.set("")  # keep whatever last message was

    # ---- Events ----
    def _on_canvas_resize(self, idx: int):
        try:
            pil = self.current_pils[idx]; cvs = self.labels[idx]
            if pil is not None: render_to_canvas(cvs, pil)
        except Exception:
            pass

    def _on_click(self, idx: int):
        if not self.current_four or idx >= len(self.current_four): return
        winner = self.current_four[idx]; losers = [t for i, t in enumerate(self.current_four) if i != idx]
        self.status.set(f"You picked: {winner.name}")
        self._save_winning_image_uncropped(idx, winner)
        elo_update(winner, losers)
        try:
            save_terms(self.terms_path, self.terms); self.status.set(f"Updated: '{winner.name}' won. Rankings saved.")
        except Exception as e:
            self.status.set(f"Updated (save failed: {e})")
        batch = self._consume_prefetch_or_fetch(); self._display_batch(batch)

    def _on_other(self, idx: int):
        if not self.current_four or idx >= len(self.current_four): return
        old_term = self.current_four[idx]; extra = self.extra_var.get()
        def worker():
            self.used[old_term.name] = set()
            try: save_used(self.used_path, self.used)
            except Exception: pass
            exclude = {t.name for i, t in enumerate(self.current_four) if i != idx}
            alt_list = weighted_sample_terms(self.terms, 1, recency_map=self.recency_map, exclude=exclude)
            new_term = alt_list[0] if alt_list else None
            if new_term is None: return
            canvas_show_message(self.labels[idx], new_term.name, tag="status")
            other_hosts = {host_of(u) for i, u in enumerate(self.current_urls) if i != idx and u}
            url = None; data = None; h = None
            candidates = ddg_candidates(new_term.name, extra)
            if candidates:
                for u in candidates:
                    if u in self.used.get(new_term.name, set()): continue
                    hst = host_of(u)
                    if hst and hst in other_hosts: continue
                    d = download_image_bytes(u, u)
                    if not d: continue
                    if not looks_like_woman(d): continue
                    hh = phash_bytes(d)
                    if hh and hh in self.seen_hashes: continue
                    url, data, h = u, d, hh; break
            if url:
                if new_term.name not in self.used: self.used[new_term.name] = set()
                self.used[new_term.name].add(url)
                try: save_used(self.used_path, self.used)
                except Exception: pass
            if h:
                self.seen_hashes.add(h)
                try: save_seen_hashes(self.seen_hashes_path, self.seen_hashes)
                except Exception: pass
            def apply():
                self.current_four[idx] = new_term
                if data:
                    try:
                        pil = Image.open(io.BytesIO(data)).convert("RGB")
                        self.current_raw_bytes[idx] = data; self.current_urls[idx] = url; self.current_pils[idx] = pil
                        lbl = self.labels[idx]; render_to_canvas(lbl, pil)
                        try: self.root.after_idle(lambda idx=idx: self._on_canvas_resize(idx))
                        except Exception: pass
                    except Exception:
                        canvas_show_error(self.labels[idx], "Failed to load"); self.current_pils[idx]=None; self.current_urls[idx]=None
                        self.current_raw_bytes[idx]=None; self.current_photos[idx]=None
                else:
                    canvas_show_error(self.labels[idx], "Failed to load"); self.labels[idx].image=None
                    self.current_urls[idx]=None; self.current_raw_bytes[idx]=None; self.current_photos[idx]=None
                self.captions[idx].configure(text="")
                self.status.set(f"Replaced with new term: {new_term.name}")
            self.root.after(0, apply)
        threading.Thread(target=worker, daemon=True).start()

    # ---- Save winning image ----
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
        try:
            pil.save(path, format="JPEG", quality=90); self.status.set(f"Saved: {path}")
        except Exception:
            pass

class GalleryWindow(tk.Toplevel):
    BATCH_SIZE = 8  # images per "page" per term

    def __init__(self, app: "App"):
        super().__init__(app.root)
        self.title("All Ranked Images"); self.geometry("1200x800")
        self.app = app
        self._images: list[ImageTk.PhotoImage] = []
        self.thumb_size_var = tk.IntVar(value=240)
        self.sort_mode = tk.StringVar(value="rating_desc")  # default sort
        self._term_offsets: dict[str, int] = {}  # term_dir -> next start index
        self._thumb_cache: dict[tuple[str, int], ImageTk.PhotoImage] = {}  # (path, size) -> tk image

        # container
        container = ttk.Frame(self); container.pack(fill=tk.BOTH, expand=True)

        # top controls: size buttons and sorting buttons
        top = ttk.Frame(container); top.pack(fill=tk.X, padx=10, pady=(8,0))
        ttk.Label(top, text="Size:").pack(side=tk.LEFT)
        for s in range(100, 1001, 100):
            ttk.Button(top, text=str(s), command=lambda size=s: self._set_size(size)).pack(side=tk.LEFT, padx=2)
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

        # scrollable canvas
        self.canvas = tk.Canvas(container, highlightthickness=0, bg="#111")
        vbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y); self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = ttk.Frame(self.canvas); self.inner_id = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_frame_configure); self.canvas.bind("<Configure>", self._on_canvas_configure)

        self._build_rows_chunked()

    def _set_size(self, size: int):
        self.thumb_size_var.set(size)
        self._refresh()

    def _set_sort(self, mode: str):
        self.sort_mode.set(mode)
        self._refresh()

    def _on_frame_configure(self, event=None): self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def _on_canvas_configure(self, event): self.canvas.itemconfig(self.inner_id, width=event.width)

    def _refresh(self):
        for w in list(self.inner.children.values()):
            try: w.destroy()
            except Exception: pass
        self._images.clear()
        self._term_offsets.clear()
        self._build_rows_chunked()

    def _sorted_terms(self):
        key = self.sort_mode.get()
        terms = list(self.app.terms)
        if key == "name_asc":
            terms.sort(key=lambda t: t.name.casefold())
        elif key == "rating_asc":
            terms.sort(key=lambda t: t.rating)
        elif key == "rating_desc":
            terms.sort(key=lambda t: t.rating, reverse=True)
        elif key == "games_asc":
            terms.sort(key=lambda t: t.games)
        elif key == "games_desc":
            terms.sort(key=lambda t: t.games, reverse=True)
        elif key == "sigma_asc":
            terms.sort(key=lambda t: t.sigma)
        elif key == "sigma_desc":
            terms.sort(key=lambda t: t.sigma, reverse=True)
        else:
            terms.sort(key=lambda t: t.rating, reverse=True)
        return terms

    def _build_rows_chunked(self):
        base_dir = os.path.dirname(os.path.abspath(self.app.terms_path))
        ranked_dir = os.path.join(base_dir, "ranked_images")
        terms_sorted = self._sorted_terms()
        self._rows = []  # list of callables to build each term row lazily

        for term in terms_sorted:
            term_dir = os.path.join(ranked_dir, slugify(term.name))
            if not os.path.isdir(term_dir): 
                continue
            files = sorted([f for f in Path(term_dir).glob("*.jpg")], key=lambda p: p.name, reverse=True)
            if not files:
                continue
            def make_row_builder(term=term, term_dir=term_dir, files=files):
                def builder():
                    self._build_term_row(term, term_dir, files)
                return builder
            self._rows.append(make_row_builder())

        self._build_next_row(0)

    def _build_next_row(self, idx: int):
        if idx >= len(self._rows):
            return
        try:
            self._rows[idx]()  # build one row
        except Exception:
            pass
        self.after(5, lambda: self._build_next_row(idx + 1))

    def _build_term_row(self, term: Term, term_dir: str, files: list[Path]):
        row = len(self.inner.grid_slaves())
        count = len(files)
        stats = f"rating {term.rating:.0f} | games {term.games} | sigma {term.sigma:.1f} | images {count}"
        hdr = ttk.Label(self.inner, text=f"{term.name} — {stats}", font=("Segoe UI", 12, "bold"))
        hdr.grid(row=row, column=0, sticky="w", padx=10, pady=(12, 4)); row += 1

        wrap = ttk.Frame(self.inner); wrap.grid(row=row, column=0, sticky="ew", padx=10); row += 1
        start = 0
        page = self.BATCH_SIZE
        self._term_offsets[term_dir] = start + page
        subset = files[start:start+page]
        self._render_thumbs_into(wrap, subset)

        if len(files) > page:
            btn = ttk.Button(self.inner, text="Load more…", command=lambda td=term_dir, f=files, w=wrap, b=None: self._on_load_more(td, f, w, b))
            btn.grid(row=row, column=0, sticky="w", padx=10, pady=(6, 0))
            btn.configure(command=lambda td=term_dir, f=files, w=wrap, b=btn: self._on_load_more(td, f, w, b))

    def _on_load_more(self, term_dir: str, files: list[Path], wrap: ttk.Frame, btn: ttk.Button | None):
        start = self._term_offsets.get(term_dir, 0)
        page = self.BATCH_SIZE
        subset = files[start:start+page]
        self._render_thumbs_into(wrap, subset)
        start += page
        self._term_offsets[term_dir] = start
        if start >= len(files) and btn is not None:
            try: btn.destroy()
            except Exception: pass

    def _render_thumbs_into(self, parent: ttk.Frame, file_list: list[Path]):
        if not file_list: return
        size = int(self.thumb_size_var.get())
        try: resample = Image.Resampling.LANCZOS
        except Exception: resample = Image.ANTIALIAS  # type: ignore
        c = 0
        for fp in file_list:
            try:
                key = (str(fp), size)
                if key in self._thumb_cache:
                    tkimg = self._thumb_cache[key]
                else:
                    pil = Image.open(fp).convert("RGB")
                    thumb_img = ImageOps.contain(pil, (size, size), method=resample)
                    tkimg = ImageTk.PhotoImage(thumb_img)
                    if len(self._thumb_cache) > 800:
                        try: self._thumb_cache.pop(next(iter(self._thumb_cache)))
                        except Exception: pass
                    self._thumb_cache[key] = tkimg
                lbl = tk.Label(parent, image=tkimg, bg="#111", cursor="hand2")
                lbl.image = tkimg
                lbl.bind("<Button-1>", lambda e, path=str(fp): self._open_fullscreen(path))
                lbl.grid(row=0, column=c, padx=6, pady=6); c += 1
                self._images.append(tkimg)
            except Exception:
                continue

    def _open_fullscreen(self, path: str):
        FullscreenImageWindow(self, path)

class FullscreenImageWindow(tk.Toplevel):
    def __init__(self, parent: GalleryWindow, path: str):
        super().__init__(parent)
        self.title(os.path.basename(path))
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Button-1>", lambda e: self.destroy())
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # load image
        self._pil = Image.open(path).convert("RGB")
        self._tk = None
        self.bind("<Configure>", self._on_resize)
        self._on_resize()

    def _on_resize(self, event=None):
        w = self.winfo_width(); h = self.winfo_height()
        if w <= 2 or h <= 2: return
        try: resample = Image.Resampling.LANCZOS
        except Exception: resample = Image.ANTIALIAS  # type: ignore
        fitted = ImageOps.contain(self._pil, (w, h), method=resample)
        self._tk = ImageTk.PhotoImage(fitted)
        self.canvas.delete("img")
        self.canvas.create_image(w//2, h//2, image=self._tk, anchor="center", tags="img")

def main():
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = filedialog.askopenfilename(title="Select terms file", filetypes=[("Text/CSV", "*.txt *.csv"), ("All files", "*.*")])
        if not path:
            print("No file selected."); return
    root = tk.Tk(); root.geometry("1300x1000"); root.minsize(900, 700)
    App(root, path); root.mainloop()
