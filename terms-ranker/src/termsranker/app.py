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
