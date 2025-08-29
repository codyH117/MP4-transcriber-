# transcriber_gui.pyw — no console popups (avoid Whisper's ffmpeg), smooth progress, instant output list, green checks
import os, sys, threading, queue, subprocess, shutil, time, wave
from datetime import datetime
from tkinter import Tk, StringVar, END, BOTH, LEFT, N, messagebox, filedialog
from tkinter import ttk

import numpy as np  # whisper depends on numpy; safe to import

# Optional drag & drop
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False

ALLOWED_EXT = {".mp4",".mov",".mkv",".m4a",".mp3",".wav",".aac",".flac",".ogg",".webm"}

CHUNK_SECONDS = 90  # shorter chunks for silkier progress

# ---- Fully hide ffmpeg/ffprobe consoles on Windows ----
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
STARTUPINFO = None
if os.name == "nt":
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    STARTUPINFO = si

def run_quiet_check_output(cmd):
    return subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=CREATE_NO_WINDOW,
        startupinfo=STARTUPINFO
    )

def run_quiet(cmd):
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        creationflags=CREATE_NO_WINDOW,
        startupinfo=STARTUPINFO
    )

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("FFmpeg/ffprobe not found. Install and ensure both are on PATH.")

def safe_stem(path):
    base = os.path.splitext(os.path.basename(path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_transcript_{ts}"

def ffprobe_duration_seconds(path) -> float:
    try:
        out = run_quiet_check_output([
            "ffprobe","-v","error",
            "-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1",
            path
        ]).strip()
        dur = float(out)
        if dur > 0: return dur
    except Exception:
        pass
    try:
        out = run_quiet_check_output([
            "ffprobe","-v","error","-select_streams","a:0",
            "-show_entries","stream=duration",
            "-of","default=noprint_wrappers=1:nokey=1", path
        ]).strip()
        return max(float(out), 0.0)
    except Exception:
        return 0.0

def extract_audio_chunk_to_wav(src, start_s, dur_s, out_wav):
    # Re-encode slice to 16k mono PCM WAV for stable decoding (no console shown)
    run_quiet([
        "ffmpeg","-y",
        "-ss", str(start_s),
        "-t", str(dur_s),
        "-i", src,
        "-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",
        out_wav
    ])

def read_wav_as_float32_mono_16k(path):
    """Read our 16k mono s16le WAV without spawning any external process."""
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        sampwidth = wf.getsampwidth()
        # Expecting nch=1, sr=16000, sampwidth=2
        raw = wf.readframes(nframes)
    # Convert int16 -> float32 in [-1, 1]
    audio_i16 = np.frombuffer(raw, dtype=np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32768.0
    return audio_f32, sr

def tk_listbox(parent, **kwargs):
    import tkinter as tk
    lb = tk.Listbox(parent, activestyle="none", highlightthickness=0, borderwidth=1, relief="solid",
                    background="#ffffff", selectbackground="#eaf6ef", selectforeground="#000000",
                    height=kwargs.get("height", 10))
    return lb

class TranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcriber")
        self.root.minsize(820, 420)

        style = ttk.Style()
        try: style.theme_use("clam")
        except Exception: pass
        style.configure(".", background="#ffffff")
        style.configure("TFrame", background="#ffffff")
        style.configure("TLabel", background="#ffffff")
        style.configure("Box.TLabelframe", background="#ffffff")
        style.configure("Box.TLabelframe.Label", background="#ffffff", font=("Segoe UI", 10, "bold"))
        style.configure("Green.Horizontal.TProgressbar", troughcolor="#f1f1f1", background="#45b36b")

        self.files = []
        self.files_set = set()
        self.output_dir = StringVar(value="")
        self.is_running = False

        # progress animator state
        self.curr_pct = 0.0
        self.target_pct = 0.0
        self.catchup_mode = False
        self.animating = False

        # list bookkeeping for ✅ toggles
        self.input_names = []
        self.output_names = []

        self.btn_text = StringVar(value="Transcribe")

        wrapper = ttk.Frame(root, padding=12)
        wrapper.pack(fill=BOTH, expand=True)

        wrapper.columnconfigure(0, weight=3)
        wrapper.columnconfigure(1, weight=1)
        wrapper.columnconfigure(2, weight=3)
        wrapper.rowconfigure(0, weight=1)
        wrapper.rowconfigure(1, weight=0)

        # Left: inputs
        left = ttk.Labelframe(wrapper, text="Input Files", style="Box.TLabelframe", padding=(10, 8))
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.drop_hint = ttk.Label(left, text="Drag & drop files here\n(or click Add Files)", anchor="center")
        self.drop_hint.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.input_list = tk_listbox(left, height=12)
        self.input_list.grid(row=1, column=0, sticky="nsew")
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        btn_row = ttk.Frame(left)
        btn_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(btn_row, text="Add Files", command=self.add_files_dialog).pack(side=LEFT)
        ttk.Button(btn_row, text="Clear", command=self.clear_files).pack(side=LEFT, padx=(8, 0))

        # Center: go
        center = ttk.Frame(wrapper, padding=(4, 0))
        center.grid(row=0, column=1, sticky="nsew")
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)
        self.go_btn = ttk.Button(center, textvariable=self.btn_text, command=self.start_transcription, width=16)
        self.go_btn.grid(row=0, column=0, sticky="n")

        # Right: outputs
        right = ttk.Labelframe(wrapper, text="Output", style="Box.TLabelframe", padding=(10, 8))
        right.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        right.columnconfigure(1, weight=1)

        ttk.Label(right, text="Folder:").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.out_entry = ttk.Entry(right, textvariable=self.output_dir)
        self.out_entry.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        ttk.Button(right, text="Browse…", command=self.browse_output_dir).grid(row=0, column=2, sticky="w", padx=(8, 0), pady=(0, 6))

        ttk.Label(right, text="Files being created:").grid(row=1, column=0, columnspan=3, sticky="w")
        self.output_list = tk_listbox(right, height=12)
        self.output_list.grid(row=2, column=0, columnspan=3, sticky="nsew")
        right.rowconfigure(2, weight=1)

        # Bottom: progress bars
        bottom = ttk.Frame(wrapper, padding=(0, 8, 0, 0))
        bottom.grid(row=1, column=0, columnspan=3, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        self.curr_label = ttk.Label(bottom, text="Current file:")
        self.curr_bar = ttk.Progressbar(bottom, mode="determinate", style="Green.Horizontal.TProgressbar", maximum=100)
        self.total_label = ttk.Label(bottom, text="All files:")
        self.total_bar = ttk.Progressbar(bottom, mode="determinate", style="Green.Horizontal.TProgressbar")

        self.curr_label.grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.curr_bar.grid(row=1, column=0, sticky="ew")

        # DnD
        if DND_AVAILABLE and isinstance(self.root, TkinterDnD.Tk):
            self.input_list.drop_target_register(DND_FILES)
            self.input_list.dnd_bind("<<Drop>>", self.on_drop)
            self.drop_hint.drop_target_register(DND_FILES)
            self.drop_hint.dnd_bind("<<Drop>>", self.on_drop)
            self.drop_hint.configure(foreground="#777777")
        else:
            self.drop_hint.configure(text="Drag & drop (pip install tkinterdnd2)\n—or click Add Files—")

        self.msg_q = queue.Queue()

        try:
            ensure_ffmpeg()
        except Exception as e:
            messagebox.showwarning("FFmpeg not found", str(e))

    # ---- UI helpers ----
    def on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        self.add_files(paths)

    def add_files_dialog(self):
        paths = filedialog.askopenfilenames(
            title="Select audio/video files",
            filetypes=[("Media files", "*.mp4 *.mov *.mkv *.m4a *.mp3 *.wav *.aac *.flac *.ogg *.webm"),
                       ("All files", "*.*")],
        )
        if paths: self.add_files(paths)

    def add_files(self, paths):
        added = 0
        for p in paths:
            p = p.strip()
            if not p: continue
            if p.startswith("{") and p.endswith("}"): p = p[1:-1]
            if not os.path.isfile(p): continue
            ext = os.path.splitext(p)[1].lower()
            if ext not in ALLOWED_EXT: continue
            if p not in self.files_set:
                self.files.append(p)
                self.files_set.add(p)
                base = os.path.basename(p)
                self.input_names.append(base)
                self.input_list.insert(END, base)
                added += 1
        if added and not self.output_dir.get():
            self.output_dir.set(os.path.dirname(self.files[0]))

    def clear_files(self):
        if self.is_running:
            messagebox.showinfo("Busy", "Transcription is running; please wait until it finishes.")
            return
        self.files.clear()
        self.files_set.clear()
        self.input_names.clear()
        self.output_names.clear()
        self.input_list.delete(0, END)
        self.output_list.delete(0, END)
        self.curr_bar["value"] = 0
        self.total_bar["value"] = 0
        self.btn_text.set("Transcribe")

    def browse_output_dir(self):
        out = filedialog.askdirectory(title="Choose output folder")
        if out: self.output_dir.set(out)

    def set_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.go_btn.config(state=state)
        self.out_entry.config(state=state)

    # ---- Animator (always moving, ease + catch-up) ----
    def _ensure_animator(self):
        if self.animating: return
        self.animating = True
        self._anim_tick()

    def _anim_tick(self):
        delta = self.target_pct - self.curr_pct
        dist = abs(delta)
        if hasattr(self, "catchup_mode") and self.catchup_mode:
            alpha = 0.25
        else:
            alpha = 0.05 + min(0.20, dist * 0.01)
        self.curr_pct += alpha * delta
        if dist < 0.2:
            self.curr_pct = self.target_pct
            self.catchup_mode = False
        self.curr_bar["value"] = max(0, min(100, self.curr_pct))
        if self.is_running or abs(self.curr_pct - self.target_pct) > 0.01:
            self.root.after(30, self._anim_tick)
        else:
            self.animating = False

    # ---- Flow ----
    def start_transcription(self):
        if self.is_running: return
        if not self.files:
            messagebox.showinfo("No files", "Add at least one media file to transcribe.")
            return

        out_dir = self.output_dir.get().strip()
        if not out_dir:
            messagebox.showinfo("No output folder", "Choose an output folder on the right.")
            return
        if not os.path.isdir(out_dir):
            try: os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Output error", f"Cannot create output folder:\n{e}")
                return

        self.is_running = True
        self.set_controls_enabled(False)
        self.output_list.delete(0, END)
        self.output_names.clear()
        self.btn_text.set("Transcribe")  # reset in case it had a ✅

        total = len(self.files)
        if total > 1:
            self.total_label.grid(row=2, column=0, sticky="w", pady=(8, 2))
            self.total_bar.grid(row=3, column=0, sticky="ew")
            self.total_bar.config(value=0, maximum=total)
        else:
            self.total_label.grid_forget()
            self.total_bar.grid_forget()

        # Populate "Files being created" immediately
        for src in self.files:
            base_out = safe_stem(src) + ".txt"
            self.output_names.append(base_out)
            self.output_list.insert(END, base_out)

        self.curr_pct = 0.0
        self.target_pct = 0.0
        self.curr_bar["value"] = 0
        self._ensure_animator()

        t = threading.Thread(target=self._run_transcriptions, args=(out_dir,), daemon=True)
        t.start()
        self.root.after(50, self._drain_messages)

    def _mark_done_in_list(self, listbox, names_list, match_name):
        # exact match first
        try:
            idx = names_list.index(match_name)
        except ValueError:
            # fallback: startswith (for timestamped names)
            idx = next((i for i, n in enumerate(names_list) if n.startswith(match_name)), None)
            if idx is None: return
        current = listbox.get(idx)
        if not current.startswith("✅ "):
            listbox.delete(idx)
            listbox.insert(idx, "✅ " + current)

    def _drain_messages(self):
        try:
            while True:
                kind, payload = self.msg_q.get_nowait()
                if kind == "chunk-target":
                    self.catchup_mode = False
                    self.target_pct = float(payload)
                    self._ensure_animator()
                elif kind == "chunk-done":
                    if self.curr_pct < self.target_pct - 0.2:
                        self.catchup_mode = True
                        self._ensure_animator()
                    else:
                        self.catchup_mode = False
                elif kind == "file-done":
                    in_base, out_name = payload
                    self._mark_done_in_list(self.input_list, self.input_names, in_base)
                    self._mark_done_in_list(self.output_list, self.output_names, out_name)
                    if self.total_bar.winfo_ismapped():
                        self.total_bar["value"] = min(self.total_bar["value"] + 1, self.total_bar["maximum"] or 1)
                elif kind == "done-all":
                    self.is_running = False
                    self.set_controls_enabled(True)
                    self.target_pct = 100.0
                    self._ensure_animator()
                    def finalize():
                        # reset bars & show ✅ on button
                        self.curr_pct = 0.0
                        self.target_pct = 0.0
                        self.curr_bar["value"] = 0
                        if self.total_bar.winfo_ismapped():
                            self.total_bar["value"] = 0
                        self.btn_text.set("Transcribe ✅")
                    self.root.after(500, finalize)
                    return
                elif kind == "error":
                    self.is_running = False
                    self.set_controls_enabled(True)
                    messagebox.showerror("Error", payload)
                    return
        except queue.Empty:
            pass
        self.root.after(50, self._drain_messages)

    def _run_transcriptions(self, out_dir):
        try:
            ensure_ffmpeg()
            try:
                import whisper
            except Exception:
                raise RuntimeError("Whisper is not installed. In PowerShell run:\n\npip install -U openai-whisper")

            model_name = os.environ.get("WHISPER_MODEL", "base")
            model = whisper.load_model(model_name)

            tmp_dir = os.path.join(out_dir, "_tmp_chunks")
            os.makedirs(tmp_dir, exist_ok=True)

            try:
                for src in list(self.files):
                    in_base = os.path.basename(src)
                    planned = safe_stem(src) + ".txt"
                    txt_path = os.path.join(out_dir, planned)

                    dur = ffprobe_duration_seconds(src)
                    if dur <= 0:
                        # fallback single-shot (still no ffmpeg from whisper; we decode ourselves)
                        # Make a full-length WAV temp and read it
                        tmp_wav = os.path.join(tmp_dir, "full.wav")
                        extract_audio_chunk_to_wav(src, 0, 1e9, tmp_wav)
                        audio, sr = read_wav_as_float32_mono_16k(tmp_wav)
                        res = model.transcribe(audio=audio)
                        text = res.get("text", "").strip()
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(text + "\n")
                        try: os.remove(tmp_wav)
                        except Exception: pass
                        self.msg_q.put(("chunk-target", 100.0))
                        self.msg_q.put(("chunk-done", None))
                        self.msg_q.put(("file-done", (in_base, planned)))
                        continue

                    text_parts = []
                    start = 0.0
                    while start < dur - 0.01:
                        chunk_len = min(CHUNK_SECONDS, dur - start)
                        target_pct = ((start + chunk_len) / dur) * 100.0
                        self.msg_q.put(("chunk-target", target_pct))

                        tmp_wav = os.path.join(tmp_dir, f"chunk_{int(start)}_{int(start+chunk_len)}.wav")
                        extract_audio_chunk_to_wav(src, start, chunk_len, tmp_wav)

                        # Read WAV locally (no external decoder) and transcribe array
                        audio, sr = read_wav_as_float32_mono_16k(tmp_wav)
                        res = model.transcribe(audio=audio)
                        text_parts.append(res.get("text", "").strip())

                        # Done with this chunk
                        self.msg_q.put(("chunk-done", None))
                        try: os.remove(tmp_wav)
                        except Exception: pass

                        start += chunk_len

                    final_text = ("\n").join(t for t in text_parts if t).strip()
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(final_text + "\n")

                    self.msg_q.put(("file-done", (in_base, planned)))

            finally:
                # Clean tmp dir
                try:
                    for name in os.listdir(tmp_dir):
                        if name.lower().endswith(".wav"):
                            os.remove(os.path.join(tmp_dir, name))
                    if not os.listdir(tmp_dir):
                        os.rmdir(tmp_dir)
                except Exception:
                    pass

            self.msg_q.put(("done-all", None))

        except Exception as e:
            self.msg_q.put(("error", str(e)))

def main():
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = Tk()
    app = TranscriberApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
