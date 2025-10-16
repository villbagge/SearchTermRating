from __future__ import annotations
import sys, tkinter as tk
from .app import App, GalleryWindow

def main():
    # Start minimal host window and open Gallery directly
    arg_path = sys.argv[1] if len(sys.argv) >= 2 else None
    root = tk.Tk(); root.geometry("1200x800")
    app = App(root, arg_path)
    # Open gallery immediately
    GalleryWindow(app)
    root.mainloop()
