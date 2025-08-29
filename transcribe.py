# transcribe.py
# Drag-and-drop friendly: accepts a video path as argv[1], lets you pick an output folder,
# runs Whisper locally (base model by default), and writes a .txt (and optional .srt).

import sys
import os
import subprocess
import shutil
from datetime import datetime

def ensure_ffmpeg():
    # Whisper shells out to ffmpeg; fail fast if missing.
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg not found. Install it and make sure 'ffmpeg' is on your PATH."
        )

def pick_output_dir(default_dir):
    # If a second argument is provided, use that. Otherwise show a folder picker.
    if len(sys.argv) >= 3 and sys.argv[2].strip():
        return os.path.abspath(sys.argv[2])

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        out_dir = filedialog.askdirectory(
            title="Select output folder for transcript",
            initialdir=default_dir
        )
        if out_dir:
            return out_dir
    except Exception:
        pass
    # Fallback: same folder as the media file
    return default_dir

def safe_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    # Optionally timestamp to avoid overwrites
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_transcript_{ts}"

def main():
    if len(sys.argv) < 2:
        print("Usage:\n  python transcribe.py <video_or_audio_path> [output_folder]")
        print("\nTip: You can also drag and drop the file onto the .bat shortcut.")
        sys.exit(1)

    media_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(media_path):
        print(f"File not found: {media_path}")
        sys.exit(1)

    ensure_ffmpeg()

    # Lazy import to show nicer errors if whisper isn't installed
    try:
        import whisper
    except Exception as e:
        print("Whisper is not installed. Run:\n  pip install -U openai-whisper")
        sys.exit(1)

    # Choose a model: 'base' is a good speed/quality tradeoff for English.
    model_name = os.environ.get("WHISPER_MODEL", "base")
    print(f"[Whisper] Loading model: {model_name} ...")
    model = whisper.load_model(model_name)

    out_dir = pick_output_dir(default_dir=os.path.dirname(media_path))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Whisper] Transcribing: {media_path}")
    # You can set language='en' if you know the language; otherwise Whisper will detect.
    result = model.transcribe(media_path)  # , language='en'

    base_out = os.path.join(out_dir, safe_stem(media_path))

    # Write plain text
    txt_path = base_out + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result.get("text", "").strip() + "\n")
    print(f"[Done] Transcript saved: {txt_path}")

    # Optional: also write SRT captions (uncomment to enable)
    # segments = result.get("segments", [])
    # srt_path = base_out + ".srt"
    # def fmt_time(t):
    #     # SRT 00:00:00,000
    #     from math import floor
    #     ms = int((t - floor(t)) * 1000)
    #     t = int(floor(t))
    #     h, t = divmod(t, 3600)
    #     m, s = divmod(t, 60)
    #     return f"{h:02}:{m:02}:{s:02},{ms:03}"
    # with open(srt_path, "w", encoding="utf-8") as srt:
    #     for i, seg in enumerate(segments, 1):
    #         srt.write(f"{i}\n")
    #         srt.write(f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}\n")
    #         srt.write(seg["text"].strip() + "\n\n")
    # print(f"[Done] SRT saved: {srt_path}")

if __name__ == "__main__":
    main()
