"""Interaktive Tkinter-Oberfläche für MemoryBall Studio."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from PIL import Image, ImageTk

from .face_cropper import FaceCropper
from .image_pipeline import determine_crop_box, process_image
from .utils import (
    CropBox,
    ProcessingOptions,
    clamp,
    ensure_dir,
    iter_media_files,
    is_image,
    is_video,
    setup_environment,
)
from .video_pipeline import process_video


class Application(tk.Tk):
    """Tkinter-Anwendung mit Vorschau und manueller Zuschnittssteuerung."""

    CANVAS_SIZE = 520
    DETECTION_CHOICES = [
        ("face", "Gesichtserkennung"),
        ("person", "Menscherkennung"),
        ("none", "Keine Erkennung"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.title("MemoryBall Studio")
        self.geometry("960x640")
        self.minsize(900, 600)

        self._configure_style()

        self.input_path: Optional[Path] = None
        self.media_files: list[Path] = []
        self.image_files: list[Path] = []
        self.manual_crops: dict[Path, CropBox] = {}
        self.current_path: Optional[Path] = None
        self.current_image: Optional[Image.Image] = None
        self._tk_image: Optional[ImageTk.PhotoImage] = None
        self._preview_cropper: Optional[FaceCropper] = None
        self._updating_controls = False

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self._detection_value_by_label = {label: value for value, label in self.DETECTION_CHOICES}
        self._detection_label_by_value = {value: label for value, label in self.DETECTION_CHOICES}
        self.detection_mode_var = tk.StringVar(value=self._detection_label_by_value["face"])
        self._last_detection_mode = "face"
        self.size_var = tk.IntVar(value=480)
        self.size_ratio = tk.DoubleVar(value=1.0)
        self.offset_x = tk.DoubleVar(value=0.0)
        self.offset_y = tk.DoubleVar(value=0.0)
        self.progress_var = tk.StringVar(value="Bereit.")
        self.crop_info_var = tk.StringVar(value="Kein Bild ausgewählt.")

        self._build_layout()
        self.detection_mode_var.trace_add("write", self._on_detection_change)

    # ------------------------------------------------------------------
    # Layout & UI
    # ------------------------------------------------------------------
    def _configure_style(self) -> None:
        background = "#0f111a"
        card_background = "#161a27"
        accent = "#3f8efc"
        self.configure(background=background)
        self.option_add("*Font", "Segoe UI 10")
        self.option_add("*Label.font", "Segoe UI 10")
        self.option_add("*Entry.background", card_background)
        self.option_add("*Entry.foreground", "#f5f7fa")
        self.option_add("*Entry.insertBackground", "#f5f7fa")
        self.option_add("*Spinbox.background", card_background)
        self.option_add("*Spinbox.foreground", "#f5f7fa")
        self.option_add("*Spinbox.insertBackground", "#f5f7fa")
        self.option_add("*TCombobox*Listbox.background", card_background)
        self.option_add("*TCombobox*Listbox.foreground", "#f5f7fa")
        self.option_add("*TCombobox*Listbox.selectBackground", accent)
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=background)
        style.configure("Card.TFrame", background=card_background, relief="flat")
        style.configure("TLabel", background=background, foreground="#f5f7fa")
        style.configure("Section.TLabel", background=background, foreground="#9aa0b5", font=("Segoe UI", 9, "bold"))
        style.configure("Heading.TLabel", background=background, foreground="#f5f7fa", font=("Segoe UI", 12, "bold"))
        style.configure("TButton", padding=8)
        style.configure(
            "Accent.TButton",
            background=accent,
            foreground="#ffffff",
            padding=10,
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#1f6feb"), ("disabled", "#1b1f2b")],
            foreground=[("disabled", "#9aa0b5")],
        )
        style.configure(
            "Modern.TCombobox",
            fieldbackground=card_background,
            background=card_background,
            foreground="#f5f7fa",
        )
        style.map(
            "Modern.TCombobox",
            fieldbackground=[("readonly", card_background)],
            background=[("readonly", card_background)],
        )
        style.configure("Horizontal.TScale", background=background, troughcolor=card_background)
        style.configure(
            "Modern.TSpinbox",
            fieldbackground=card_background,
            background=card_background,
            foreground="#f5f7fa",
        )
        style.map("Modern.TSpinbox", fieldbackground=[("readonly", card_background)])

    def _current_detection_mode(self) -> str:
        return self._detection_value_by_label.get(
            self.detection_mode_var.get(),
            "none",
        )

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main = ttk.Frame(self, padding=16)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        top = ttk.Frame(main)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        top.columnconfigure(0, weight=1)

        input_card = ttk.Frame(top, style="Card.TFrame", padding=16)
        input_card.grid(row=0, column=0, sticky="ew")
        input_card.columnconfigure(1, weight=1)

        ttk.Label(input_card, text="Ordner", style="Heading.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(input_card, text="Eingabeordner", style="Section.TLabel").grid(
            row=1, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Entry(input_card, textvariable=self.input_var).grid(
            row=1, column=1, sticky="ew", padx=(12, 8), pady=(12, 0)
        )
        ttk.Button(input_card, text="Wählen…", command=self._choose_input).grid(
            row=1, column=2, sticky="ew", pady=(12, 0)
        )

        ttk.Label(input_card, text="Ausgabeordner", style="Section.TLabel").grid(
            row=2, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Entry(input_card, textvariable=self.output_var).grid(
            row=2, column=1, sticky="ew", padx=(12, 8), pady=(12, 0)
        )
        ttk.Button(input_card, text="Wählen…", command=self._choose_output).grid(
            row=2, column=2, sticky="ew", pady=(12, 0)
        )

        options = ttk.Frame(input_card)
        options.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(16, 0))
        options.columnconfigure(3, weight=1)

        ttk.Label(options, text="Motiv-Erkennung", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        detection_labels = list(self._detection_value_by_label.keys())
        ttk.Combobox(
            options,
            textvariable=self.detection_mode_var,
            values=detection_labels,
            state="readonly",
            style="Modern.TCombobox",
            width=18,
        ).grid(row=0, column=1, sticky="w", padx=(8, 16))

        ttk.Label(options, text="Zielgröße", style="Section.TLabel").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            options,
            from_=256,
            to=1080,
            increment=16,
            textvariable=self.size_var,
            width=7,
            style="Modern.TSpinbox",
        ).grid(row=0, column=3, sticky="w")

        list_frame = ttk.Frame(main, style="Card.TFrame", padding=16)
        list_frame.grid(row=1, column=0, sticky="nswe")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(1, weight=1)
        ttk.Label(list_frame, text="Bilder", style="Heading.TLabel").grid(row=0, column=0, sticky="w")

        self.listbox = tk.Listbox(list_frame, exportselection=False, height=20)
        self.listbox.grid(row=1, column=0, sticky="nswe", pady=(6, 0))
        self.listbox.bind("<<ListboxSelect>>", lambda _event: self._on_listbox_select())
        self.listbox.configure(
            background="#0b0f1c",
            foreground="#f5f7fa",
            borderwidth=0,
            highlightthickness=0,
            selectbackground="#1f6feb",
            selectforeground="#ffffff",
            activestyle="none",
        )

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scrollbar.set)

        preview = ttk.Frame(main, style="Card.TFrame", padding=16)
        preview.grid(row=1, column=1, sticky="nsew", padx=(16, 0))
        preview.columnconfigure(0, weight=1)

        ttk.Label(preview, text="Vorschau", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(
            preview,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            background="#0b0f1c",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.grid(row=1, column=0, sticky="n", pady=(12, 16))

        sliders = ttk.Frame(preview)
        sliders.grid(row=2, column=0, sticky="ew")
        sliders.columnconfigure(1, weight=1)
        sliders.columnconfigure(3, weight=1)

        ttk.Label(sliders, text="Zoom").grid(row=0, column=0, sticky="w")
        self.size_scale = ttk.Scale(sliders, from_=0.25, to=1.0, variable=self.size_ratio, command=self._on_slider_change)
        self.size_scale.grid(row=0, column=1, sticky="ew", padx=(6, 6))

        ttk.Label(sliders, text="X-Position").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.x_scale = ttk.Scale(sliders, from_=0.0, to=1.0, variable=self.offset_x, command=self._on_slider_change)
        self.x_scale.grid(row=1, column=1, sticky="ew", padx=(6, 6), pady=(6, 0))

        ttk.Label(sliders, text="Y-Position").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.y_scale = ttk.Scale(sliders, from_=0.0, to=1.0, variable=self.offset_y, command=self._on_slider_change)
        self.y_scale.grid(row=2, column=1, sticky="ew", padx=(6, 6), pady=(6, 0))

        ttk.Button(sliders, text="Auto", command=self._reset_crop_to_auto).grid(
            row=0,
            column=2,
            rowspan=3,
            sticky="ns",
            padx=(12, 0),
        )

        ttk.Label(preview, textvariable=self.crop_info_var, style="Section.TLabel").grid(
            row=3,
            column=0,
            sticky="w",
            pady=(12, 0),
        )

        bottom = ttk.Frame(main)
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        bottom.columnconfigure(0, weight=1)

        ttk.Label(bottom, textvariable=self.progress_var, style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.convert_button = ttk.Button(bottom, text="Konvertieren", command=self._on_convert, style="Accent.TButton")
        self.convert_button.grid(row=0, column=1, sticky="e")

        self._set_controls_enabled(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_path(self, path: Path) -> Path:
        try:
            return path.resolve()
        except OSError:
            return path

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for scale in (self.size_scale, self.x_scale, self.y_scale):
            if enabled:
                scale.state(["!disabled"])
            else:
                scale.state(["disabled"])
        self.listbox.configure(state=state)

    def _default_output_for(self, path: Path) -> Path:
        return path.parent / f"Converted {path.name}"

    def _resolve_output_dir(self) -> Optional[Path]:
        raw = self.output_var.get().strip()
        if not raw and self.input_path:
            return self._default_output_for(self.input_path)
        if not raw:
            return None
        return Path(raw).expanduser()

    def _get_preview_cropper(self) -> Optional[FaceCropper]:
        mode = self._current_detection_mode()
        if mode == "none":
            return None
        if self._preview_cropper is None or getattr(self._preview_cropper, "mode", None) != mode:
            if self._preview_cropper is not None:
                self._preview_cropper.close()
            self._preview_cropper = FaceCropper(mode=mode)
        return self._preview_cropper

    def destroy(self) -> None:  # pragma: no cover - GUI shutdown
        if self._preview_cropper is not None:
            self._preview_cropper.close()
            self._preview_cropper = None
        super().destroy()

    # ------------------------------------------------------------------
    # Folder handling
    # ------------------------------------------------------------------
    def _choose_input(self) -> None:
        selection = filedialog.askdirectory(title="Eingabeordner wählen")
        if selection:
            self._set_input_path(Path(selection))

    def _choose_output(self) -> None:
        selection = filedialog.askdirectory(title="Ausgabeordner wählen")
        if selection:
            self.output_var.set(selection)

    def _set_input_path(self, path: Path) -> None:
        self.input_path = self._normalize_path(path)
        self.input_var.set(str(self.input_path))
        default_output = self._default_output_for(self.input_path)
        self.output_var.set(str(default_output))
        self.manual_crops.clear()
        self._load_media_files()

    def _load_media_files(self) -> None:
        self.media_files.clear()
        self.image_files.clear()
        self.listbox.delete(0, tk.END)
        self.canvas.delete("all")
        self.current_path = None
        self.current_image = None
        self._tk_image = None
        self.crop_info_var.set("Kein Bild ausgewählt.")
        self._set_controls_enabled(False)

        if not self.input_path:
            return

        files = [self._normalize_path(path) for path in iter_media_files(self.input_path)]
        files.sort()
        self.media_files = files
        self.image_files = [f for f in files if is_image(f)]

        for image in self.image_files:
            display = image.relative_to(self.input_path)
            self.listbox.insert(tk.END, str(display))

        if self.image_files:
            self.listbox.selection_set(0)
            self._on_listbox_select()
            self.progress_var.set(f"{len(self.image_files)} Bilder geladen.")
        else:
            self.progress_var.set("Keine unterstützten Bilder gefunden.")

    # ------------------------------------------------------------------
    # Preview & manual crop
    # ------------------------------------------------------------------
    def _on_listbox_select(self) -> None:
        if not self.image_files:
            return
        selection = self.listbox.curselection()
        if not selection:
            return
        index = selection[0]
        path = self.image_files[index]
        if not path.exists():
            self.progress_var.set("Datei nicht gefunden.")
            return
        self._load_preview(path)

    def _load_preview(self, path: Path) -> None:
        self.current_path = path
        with Image.open(path) as img:
            self.current_image = img.copy()
        crop = self.manual_crops.get(path)
        if crop is None:
            crop = self._auto_crop_current()
            self.manual_crops[path] = crop
        self._apply_crop_to_controls(crop)
        self._set_controls_enabled(True)

    def _auto_crop_current(self) -> CropBox:
        assert self.current_image is not None and self.input_path is not None
        detection_mode = self._current_detection_mode()
        options = ProcessingOptions(
            input_path=self.input_path,
            output_dir=self._resolve_output_dir() or self._default_output_for(self.input_path),
            size=self.size_var.get(),
            face_detection_enabled=detection_mode != "none",
            detection_mode=detection_mode,
        )
        cropper = self._get_preview_cropper()
        return determine_crop_box(self.current_image, options, cropper)

    def _apply_crop_to_controls(self, crop: CropBox) -> None:
        if self.current_image is None or self.current_path is None:
            return
        width, height = self.current_image.size
        min_side = max(1, min(width, height))
        size_ratio = clamp(crop.size / min_side, 0.01, 1.0)
        max_x = max(width - crop.size, 1e-9)
        max_y = max(height - crop.size, 1e-9)
        offset_x = clamp(crop.x / max_x if max_x else 0.0, 0.0, 1.0)
        offset_y = clamp(crop.y / max_y if max_y else 0.0, 0.0, 1.0)

        self._updating_controls = True
        self.size_ratio.set(size_ratio)
        self.offset_x.set(offset_x)
        self.offset_y.set(offset_y)
        self.manual_crops[self.current_path] = crop
        self._updating_controls = False
        self._render_preview(crop)

    def _on_slider_change(self, _value: float | str) -> None:
        if self._updating_controls or self.current_image is None or self.current_path is None:
            return
        width, height = self.current_image.size
        min_side = max(1, min(width, height))
        ratio = clamp(self.size_ratio.get(), 0.01, 1.0)
        size = ratio * min_side
        max_x = max(0.0, width - size)
        max_y = max(0.0, height - size)
        x = clamp(self.offset_x.get(), 0.0, 1.0) * max_x
        y = clamp(self.offset_y.get(), 0.0, 1.0) * max_y
        crop = CropBox(x=x, y=y, size=size)
        self.manual_crops[self.current_path] = crop
        self._render_preview(crop)

    def _render_preview(self, crop: CropBox) -> None:
        if self.current_image is None:
            return
        width, height = self.current_image.size
        scale = min(self.CANVAS_SIZE / width, self.CANVAS_SIZE / height, 1.0)
        display_width = int(width * scale)
        display_height = int(height * scale)
        resized = self.current_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        offset_x = (self.CANVAS_SIZE - display_width) / 2
        offset_y = (self.CANVAS_SIZE - display_height) / 2
        self.canvas.create_image(self.CANVAS_SIZE / 2, self.CANVAS_SIZE / 2, image=self._tk_image)

        rect = (
            offset_x + crop.x * scale,
            offset_y + crop.y * scale,
            offset_x + (crop.x + crop.size) * scale,
            offset_y + (crop.y + crop.size) * scale,
        )
        self.canvas.create_rectangle(*rect, outline="#00ff88", width=2)
        self.crop_info_var.set(
            f"Ausschnitt: {int(crop.size)}px – Position ({int(crop.x)}, {int(crop.y)})"
        )

    def _reset_crop_to_auto(self) -> None:
        if self.current_image is None or self.current_path is None:
            return
        crop = self._auto_crop_current()
        self.manual_crops[self.current_path] = crop
        self._apply_crop_to_controls(crop)

    def _on_detection_change(self, *_args: object) -> None:
        mode = self._current_detection_mode()
        if mode == self._last_detection_mode:
            return
        self._last_detection_mode = mode
        if self._preview_cropper is not None:
            self._preview_cropper.close()
            self._preview_cropper = None
        if self.current_path is not None and self.current_image is not None:
            self.manual_crops.pop(self.current_path, None)
            self._reset_crop_to_auto()

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def _on_convert(self) -> None:
        if self.input_path is None:
            messagebox.showerror("Fehler", "Bitte zuerst einen Eingabeordner wählen.")
            return
        output_dir = self._resolve_output_dir()
        if output_dir is None:
            messagebox.showerror("Fehler", "Bitte einen Ausgabeordner wählen oder eingeben.")
            return
        ensure_dir(output_dir)
        self.convert_button.state(["disabled"])
        self.progress_var.set("Konvertierung läuft…")
        manual_overrides = {path: crop for path, crop in self.manual_crops.items()}
        thread = threading.Thread(
            target=self._run_batch,
            args=(output_dir, manual_overrides),
            daemon=True,
        )
        thread.start()

    def _run_batch(self, output_dir: Path, manual_overrides: dict[Path, CropBox]) -> None:
        assert self.input_path is not None
        detection_mode = self._current_detection_mode()
        options = ProcessingOptions(
            input_path=self.input_path,
            output_dir=output_dir,
            size=self.size_var.get(),
            face_detection_enabled=detection_mode != "none",
            detection_mode=detection_mode,
        )
        try:
            logger = setup_environment(options)
        except SystemExit:
            self.after(0, lambda: self._handle_error("ffmpeg/ffprobe nicht gefunden. Bitte installieren."))
            return

        files = list(iter_media_files(options.input_path))
        files = [self._normalize_path(path) for path in files]
        total = len(files)
        if total == 0:
            self.after(0, lambda: self._handle_error("Keine unterstützten Dateien gefunden."))
            return

        face_cropper = (
            FaceCropper(
                min_face=options.min_face,
                face_priority=options.face_priority,
                mode=options.detection_mode,
            )
            if options.face_detection_enabled
            else None
        )
        manual_map = {self._normalize_path(path): crop for path, crop in manual_overrides.items()}

        processed = 0
        for path in files:
            try:
                if is_image(path):
                    override = manual_map.get(path)
                    process_image(path, options, face_cropper, manual_crop=override)
                elif is_video(path):
                    process_video(path, options, face_cropper)
                processed += 1
                self.after(
                    0,
                    lambda done=processed, total=total, name=path.name: self.progress_var.set(
                        f"{done}/{total} verarbeitet – {name}"
                    ),
                )
            except Exception as exc:  # pragma: no cover - Fehlerdialog im GUI-Thread
                logger.exception("Fehler bei %s", path)
                self.after(0, lambda: messagebox.showerror("Fehler", f"Verarbeitung fehlgeschlagen: {path.name}\n{exc}"))

        if face_cropper is not None:
            face_cropper.close()

        self.after(0, self._finish_batch)

    def _handle_error(self, message: str) -> None:
        self.progress_var.set(message)
        messagebox.showerror("Fehler", message)
        self.convert_button.state(["!disabled"])

    def _finish_batch(self) -> None:
        self.progress_var.set("Fertig.")
        self.convert_button.state(["!disabled"])
        messagebox.showinfo("Fertig", "Alle Dateien wurden konvertiert.")


def launch_gui() -> None:
    app = Application()
    app.mainloop()
