"""Minimal Tkinter GUI for memoryball-autocrop."""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from .face_cropper import FaceCropper
from .image_pipeline import process_image
from .video_pipeline import process_video
from .utils import ProcessingOptions, ensure_dir, iter_media_files, is_image, is_video, setup_environment


class Application(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("memoryball-autocrop")
        self.geometry("480x300")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.size_var = tk.IntVar(value=480)
        self.crf_var = tk.IntVar(value=20)
        self.quality_var = tk.IntVar(value=90)
        self.face_var = tk.BooleanVar(value=True)
        self.progress_var = tk.DoubleVar(value=0.0)
        self._build_widgets()

    def _build_widgets(self) -> None:
        pad = {"padx": 10, "pady": 5}

        tk.Label(self, text="Eingabe").grid(row=0, column=0, sticky="w", **pad)
        tk.Entry(self, textvariable=self.input_var, width=40).grid(row=0, column=1, **pad)
        tk.Button(self, text="…", command=self._choose_input).grid(row=0, column=2, **pad)

        tk.Label(self, text="Ausgabe").grid(row=1, column=0, sticky="w", **pad)
        tk.Entry(self, textvariable=self.output_var, width=40).grid(row=1, column=1, **pad)
        tk.Button(self, text="…", command=self._choose_output).grid(row=1, column=2, **pad)

        tk.Checkbutton(self, text="Gesichtserkennung", variable=self.face_var).grid(row=2, column=0, columnspan=3, sticky="w", **pad)

        tk.Label(self, text="Größe").grid(row=3, column=0, sticky="w", **pad)
        tk.Scale(self, from_=256, to=1080, orient=tk.HORIZONTAL, variable=self.size_var).grid(row=3, column=1, columnspan=2, sticky="ew", **pad)

        tk.Label(self, text="CRF").grid(row=4, column=0, sticky="w", **pad)
        tk.Scale(self, from_=10, to=35, orient=tk.HORIZONTAL, variable=self.crf_var).grid(row=4, column=1, columnspan=2, sticky="ew", **pad)

        tk.Label(self, text="Qualität").grid(row=5, column=0, sticky="w", **pad)
        tk.Scale(self, from_=50, to=100, orient=tk.HORIZONTAL, variable=self.quality_var).grid(row=5, column=1, columnspan=2, sticky="ew", **pad)

        tk.Button(self, text="Start", command=self._start).grid(row=6, column=0, columnspan=3, **pad)

        tk.Label(self, textvariable=self.progress_var).grid(row=7, column=0, columnspan=3, **pad)

    def _choose_input(self) -> None:
        selection = filedialog.askdirectory()
        if selection:
            self.input_var.set(selection)

    def _choose_output(self) -> None:
        selection = filedialog.askdirectory()
        if selection:
            self.output_var.set(selection)

    def _start(self) -> None:
        input_path = Path(self.input_var.get())
        output_dir = Path(self.output_var.get())
        if not input_path.exists():
            messagebox.showerror("Fehler", "Bitte einen gültigen Eingabeordner wählen")
            return
        ensure_dir(output_dir)
        options = ProcessingOptions(
            input_path=input_path,
            output_dir=output_dir,
            size=self.size_var.get(),
            crf=self.crf_var.get(),
            quality=self.quality_var.get(),
            face_detection_enabled=self.face_var.get(),
        )
        threading.Thread(target=self._run_batch, args=(options,), daemon=True).start()

    def _run_batch(self, options: ProcessingOptions) -> None:
        logger = setup_environment(options)
        files = list(iter_media_files(options.input_path))
        total = len(files)
        if total == 0:
            messagebox.showinfo("Info", "Keine unterstützten Dateien gefunden")
            return
        face_cropper = FaceCropper(min_face=options.min_face, face_priority=options.face_priority) if options.face_detection_enabled else None
        processed = 0
        for item in files:
            try:
                if is_image(item):
                    process_image(item, options, face_cropper)
                elif is_video(item):
                    process_video(item, options, face_cropper)
                processed += 1
                self.progress_var.set(processed / total * 100)
            except Exception as exc:  # pragma: no cover - GUI side effects
                logger.exception("Fehler bei %s: %s", item, exc)
        if face_cropper:
            face_cropper.close()
        messagebox.showinfo("Fertig", "Batch abgeschlossen")


def launch_gui() -> None:
    app = Application()
    app.mainloop()
