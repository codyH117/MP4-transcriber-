import os
import sys
import time
import queue
import shutil
import threading
from datetime import datetime
from tkinter import (
    N, S, E, W,
    BOTH, END,
    filedialog, StringVar, ttk
)

# TkinterDnD adds native drag & drop to Tk
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
except Exception:
    TkinterDnD = None  # we will error nicely later

# ---------- Helpers ----------

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            "FFmpeg/FFprobe not found.\n\n"
            "Install FFmpeg and add its 'bin' to PATH so 'ffmpeg' and 'ffprobe' work."
        )

def media_duration_seconds(path):
    import subprocess
    try:
        out = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=nw=1:nk=1', path],
            stderr=subprocess.STDOUT
        ).decode('utf-8', 'replace').strip()
        dur = float(out)
        return max(dur, 1.0)
    except Exception:
        return 600.0  # fallback estimate

def safe_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_transcript_{ts}"

# ---------- Transcriber (background thread) ----------

class TranscriberWorker(threading.Thread):
    """
    Runs transcriptions in the background so the UI remains responsive.
    Consumes file paths from a queue and posts status messages/results back
    via a thread-safe callback.
    """
    def __init__(self, in_q, on_update, get_outdir, model_name_env="WHISPER_MODEL"):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.on_update = on_update
        self.get_outdir = get_outdir
        self.model_name_env = model_name_env
        self._stop = threading.Event()
        self._model = None
        self._device = "cpu"

    def stop(self):
        self._stop.set()

    def _lazy_init(self):
        # Import here so UI can launch even if deps missing
        import torch
        import whisper
        self._device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
        model_name = os.environ.get(self.model_name_env, "base")
        self.on_update(f"[Whisper] Loading model: {model_name} on {self._device} ...")
        self._model = whisper.load_model(model_name, device=self._device)
        return whisper

    def run(self):
        whisper = None
        try:
            ensure_ffmpeg()
        except Exception as e:
            self.on_update(f"[Error] {e}")
            return

        while not self._stop.is_set():
            try:
                path = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if not os.path.exists(path):
                self.on_update(f"[Skip] Not found: {path}")
                self.in_q.task_done()
                continue

            try:
                if self._model is None:
                    try:
                        whisper = self._lazy_init()
                    except Exception as e:
                        self.on_update(
                            "[Error] Missing dependencies.\n"
                            "Install with: pip install -U openai-whisper torch tkinterdnd2"
                        )
                        self.on_update(str(e))
                        self.in_q.task_done()
                        continue

                outdir = self.get_outdir() or os.path.dirname(path)
                os.makedirs(outdir, exist_ok=True)
                base_out = os.path.join(outdir, safe_stem(path))

                # Estimated progress ticker based on wall time vs media duration
                total = media_duration_seconds(path)
                start = time.time()
                self.on_update(f"[Transcribing] {os.path.basename(path)}")

                # Do the transcription
                result = self._model.transcribe(
                    path,
                    fp16=False,       # avoid FP16 warning on CPU
                    verbose=False,    # keep output clean
                    # language="en",  # uncomment to force English
                )

                # Write .txt
                txt_path = base_out + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(result.get("text", "").strip() + "\n")

                # Final update
                elapsed = time.time() - start
                self.on_update(f"[Done] -> {txt_path}  ({int(elapsed)}s)")

            except Exception as e:
                self.on_update(f"[Error] {e}")
            finally:
                self.in_q.task_done()

# ---------- UI ----------

class App(TkinterDnD.Tk if TkinterDnD else object):
    def __init__(self):
        if not TkinterDnD:
            raise RuntimeError(
                "tkinterdnd2 is required for drag & drop.\n"
                "Install with: pip install tkinterdnd2\n"
                "If that fails on your system, you can still click 'Add files…'."
            )
        super().__init__()

        self.title("Whisper Transcriber")
        self.geometry("880x420")
        self.minsize(800, 360)
        self.configure(bg="white")

        # Queues
        self.in_q = queue.Queue()

        # Layout: 3 columns (left list, middle button, right list)
        self.columnconfigure(0, weight=1, uniform="cols")
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=1, uniform="cols")
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # Left: input list + controls
        left_frame = ttk.Frame(self)
        left_frame.grid(row=0, column=0, sticky=N+S+E+W, padx=(16, 8), pady=16)

        ttk.Label(left_frame, text="Drop files to transcribe", font=("Segoe UI", 11)).pack(anchor="w")

        self.input_list = ttk.Listbox(left_frame, height=12)
        self.input_list.pack(fill=BOTH, expand=True, pady=(6, 8))
        self.input_list.drop_target_register(DND_FILES)
        self.input_list.dnd_bind("<<Drop>>", self.on_drop)

        btns = ttk.Frame(left_frame)
        btns.pack(fill="x", pady=(0, 8))
        ttk.Button(btns, text="Add files…", command=self.add_files).pack(side="left")
        ttk.Button(btns, text="Clear", command=lambda: self.input_list.delete(0, END)).pack(side="left", padx=(6, 0))

        # Output folder selector (bottom-left)
        out_frame = ttk.Frame(left_frame)
        out_frame.pack(fill="x", pady=(6, 0))
        ttk.Label(out_frame, text="Output location:").pack(anchor="w")
        self.outdir_var = StringVar(value="")
        out_row = ttk.Frame(out_frame)
        out_row.pack(fill="x", pady=(4, 0))
        self.out_entry = ttk.Entry(out_row, textvariable=self.outdir_var)
        self.out_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="Browse…", command=self.pick_outdir).pack(side="left", padx=(6, 0))

        # Middle: Transcribe button
        mid_frame = ttk.Frame(self)
        mid_frame.grid(row=0, column=1, sticky=N+S, padx=8, pady=16)
        self.transcribe_btn = ttk.Button(mid_frame, text="Transcribe", command=self.start_transcribe, width=14)
        self.transcribe_btn.pack(pady=(40, 8))

        # Right: results / status
        right_frame = ttk.Frame(self)
        right_frame.grid(row=0, column=2, sticky=N+S+E+W, padx=(8, 16), pady=16)
        ttk.Label(right_frame, text="Results", font=("Segoe UI", 11)).pack(anchor="w")
        self.output_list = ttk.Listbox(right_frame, height=16)
        self.output_list.pack(fill=BOTH, expand=True, pady=(6, 0))

        # Bottom status bar
        self.status_var = StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_bar.grid(row=1, column=0, columnspan=3, sticky=E+W, padx=16, pady=(0, 12))

        # Worker
        self.worker = TranscriberWorker(
            in_q=self.in_q,
            on_update=self.on_update_safe,
            get_outdir=lambda: self.outdir_var.get().strip()
        )
        self.worker.start()

        # Styles (keep it clean & white)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background="white")
        style.configure("TLabel", background="white")
        style.configure("TFrame", background="white")

    # ---- UI Callbacks ----

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select media files",
            filetypes=[("Media", "*.mp4 *.mov *.m4a *.mp3 *.wav *.mkv *.aac *.flac *.ogg *.webm"),
                       ("All files", "*.*")]
        )
        for p in paths:
            self._add_path(p)

    def on_drop(self, event):
        # event.data may contain {path1} {path2}... possibly with braces
        raw = event.data
        parts = self._split_dnd_paths(raw)
        for p in parts:
            self._add_path(p)

    def _split_dnd_paths(self, data):
        # Handles Windows-style brace-wrapped paths with spaces
        out = []
        buf = []
        in_brace = False
        for ch in data:
            if ch == "{":
                in_brace = True
                buf = []
            elif ch == "}":
                in_brace = False
                out.append("".join(buf))
                buf = []
            elif ch == " " and not in_brace:
                if buf:
                    out.append("".join(buf))
                    buf = []
            else:
                buf.append(ch)
        if buf:
            out.append("".join(buf))
        return [p.strip() for p in out if p.strip()]

    def _add_path(self, p):
        if os.path.isfile(p):
            self.input_list.insert(END, p)

    def pick_outdir(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if d:
            self.outdir_var.set(d)

    def start_transcribe(self):
        count = self.input_list.size()
        if count == 0:
            self.on_update_safe("[Info] Add or drop at least one file.")
            return
        # Queue all items
        for i in range(count):
            self.in_q.put(self.input_list.get(i))
        self.on_update_safe(f"[Queued] {count} file(s). Starting…")
        self.status_var.set("Transcribing…")
        # Disable button during work (re-enabled when queue empties)
        self.transcribe_btn.config(state="disabled")
        # Watcher thread to re-enable when done
        threading.Thread(target=self._watch_queue, daemon=True).start()

    def _watch_queue(self):
        self.in_q.join()  # waits until worker marks all tasks done
        self.on_update_safe("[All done]")
        self.status_var.set("Ready")
        self.transcribe_btn.config(state="normal")

    def on_update_safe(self, msg):
        # Marshal to main thread
        self.after(0, self._append_output, msg)

    def _append_output(self, msg):
        self.output_list.insert(END, msg)
        self.output_list.see(END)

# ---------- Entry ----------

def main():
    # Use pythonw.exe to avoid a console window.
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        # If tkinterdnd2 is missing, show a GUI error dialog
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Whisper Transcriber - Error", str(e))

if __name__ == "__main__":
    main()
