"""Interaktive Tkinter-Oberfläche für MemoryBall Studio."""

from __future__ import annotations

import os
import subprocess
import sys
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
    ManualCrop,
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
        self.manual_crops: dict[Path, ManualCrop] = {}
        self._list_paths: list[Path] = []
        self.current_path: Optional[Path] = None
        self.current_image: Optional[Image.Image] = None
        self._tk_image: Optional[ImageTk.PhotoImage] = None
        self._preview_cropper: Optional[FaceCropper] = None
        self._updating_controls = False
        self.output_media_files: list[Path] = []

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
        self.motion_enabled_var = tk.BooleanVar(value=True)
        self.active_crop_var = tk.StringVar(value="end")
        self.progress_var = tk.StringVar(value="Bereit.")
        self.crop_info_var = tk.StringVar(value="Kein Bild ausgewählt.")
        self.position_var = tk.StringVar(value="0/0")
        self.output_info_var = tk.StringVar(value="Keine Ausgabedateien.")

        self._canvas_scale = 1.0
        self._canvas_offset = (0.0, 0.0)
        self._manual_display: dict[str, tuple[float, float, float, float]] = {}
        self._drag_state: Optional[dict[str, object]] = None
        self._conversion_active = False

        self._build_layout()
        self.detection_mode_var.trace_add("write", self._on_detection_change)
        self.active_crop_var.trace_add("write", self._on_active_crop_change)

    # ------------------------------------------------------------------
    # Layout & UI
    # ------------------------------------------------------------------
    def _configure_style(self) -> None:
        background = "#0f111a"
        card_background = "#161a27"
        accent = "#3f8efc"
        self.configure(background=background)
        self.option_add("*Font", "{Segoe UI} 10")
        self.option_add("*Label.font", "{Segoe UI} 10")
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
        main.columnconfigure(2, weight=0)
        main.rowconfigure(1, weight=1)

        top = ttk.Frame(main)
        top.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 16))
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
        ttk.Label(list_frame, text="Bilder & Videos", style="Heading.TLabel").grid(row=0, column=0, sticky="w")

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
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        nav = ttk.Frame(preview)
        nav.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        nav.columnconfigure(1, weight=1)
        self.prev_button = ttk.Button(nav, text="◀", width=4, command=self._show_previous_image)
        self.prev_button.grid(row=0, column=0, sticky="w")
        ttk.Label(nav, textvariable=self.position_var, style="Section.TLabel").grid(row=0, column=1)
        self.next_button = ttk.Button(nav, text="▶", width=4, command=self._show_next_image)
        self.next_button.grid(row=0, column=2, sticky="e")

        motion_controls = ttk.Frame(preview)
        motion_controls.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        motion_controls.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            motion_controls,
            text="Bewegung aktiv",
            variable=self.motion_enabled_var,
            command=self._on_motion_toggle,
        ).grid(row=0, column=0, sticky="w")
        radio_frame = ttk.Frame(motion_controls)
        radio_frame.grid(row=0, column=1, sticky="e")
        self.start_radio = ttk.Radiobutton(
            radio_frame,
            text="Start (Rot)",
            value="start",
            variable=self.active_crop_var,
        )
        self.start_radio.grid(row=0, column=0, padx=(0, 12))
        self.end_radio = ttk.Radiobutton(
            radio_frame,
            text="Ende (Grün)",
            value="end",
            variable=self.active_crop_var,
        )
        self.end_radio.grid(row=0, column=1)

        sliders = ttk.Frame(preview)
        sliders.grid(row=4, column=0, sticky="ew")
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
            row=5,
            column=0,
            sticky="w",
            pady=(12, 0),
        )

        output = ttk.Frame(main, style="Card.TFrame", padding=16)
        output.grid(row=1, column=2, sticky="nsw", padx=(16, 0))
        output.columnconfigure(0, weight=1)
        output.rowconfigure(1, weight=1)
        ttk.Label(output, text="Ausgabe", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        self.output_listbox = tk.Listbox(output, exportselection=False, height=20)
        self.output_listbox.grid(row=1, column=0, sticky="nswe", pady=(6, 0))
        self.output_listbox.configure(
            background="#0b0f1c",
            foreground="#f5f7fa",
            borderwidth=0,
            highlightthickness=0,
            selectbackground="#1f6feb",
            selectforeground="#ffffff",
            activestyle="none",
        )
        self.output_listbox.bind("<Double-Button-1>", self._open_output_file)
        output_scroll = ttk.Scrollbar(output, orient="vertical", command=self.output_listbox.yview)
        output_scroll.grid(row=1, column=1, sticky="ns")
        self.output_listbox.configure(yscrollcommand=output_scroll.set)
        ttk.Label(output, textvariable=self.output_info_var, style="Section.TLabel").grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(12, 0),
        )

        bottom = ttk.Frame(main)
        bottom.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(16, 0))
        bottom.columnconfigure(0, weight=1)

        ttk.Label(bottom, textvariable=self.progress_var, style="Section.TLabel").grid(row=0, column=0, sticky="w")
        buttons = ttk.Frame(bottom)
        buttons.grid(row=0, column=1, sticky="e")
        self.convert_selected_button = ttk.Button(
            buttons,
            text="Nur aktuelles Bild",
            command=self._on_convert_selected,
        )
        self.convert_selected_button.grid(row=0, column=0, sticky="e", padx=(0, 8))
        self.convert_button = ttk.Button(buttons, text="Alle konvertieren", command=self._on_convert, style="Accent.TButton")
        self.convert_button.grid(row=0, column=1, sticky="e")

        self._set_controls_enabled(False)
        self._refresh_output_list()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_path(self, path: Path) -> Path:
        try:
            return path.resolve()
        except OSError:
            return path

    def _set_controls_enabled(self, enabled: bool) -> None:
        for scale in (self.size_scale, self.x_scale, self.y_scale):
            if enabled:
                scale.state(["!disabled"])
            else:
                scale.state(["disabled"])
        self.prev_button.state(["!disabled"] if enabled else ["disabled"])
        self.next_button.state(["!disabled"] if enabled else ["disabled"])
        if enabled:
            if self.motion_enabled_var.get():
                self.start_radio.state(["!disabled"])
            else:
                self.start_radio.state(["disabled"])
            self.end_radio.state(["!disabled"])
            self._refresh_selected_button_state()
        else:
            self.start_radio.state(["disabled"])
            self.end_radio.state(["disabled"])
            self.convert_selected_button.state(["disabled"])
        self.listbox.configure(state="normal")

    def _refresh_selected_button_state(self) -> None:
        if self._conversion_active:
            self.convert_selected_button.state(["disabled"])
            return
        if self.current_path is not None and self.current_path in self.image_files:
            self.convert_selected_button.state(["!disabled"])
        else:
            self.convert_selected_button.state(["disabled"])

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
            self._refresh_output_list()

    def _set_input_path(self, path: Path) -> None:
        self.input_path = self._normalize_path(path)
        self.input_var.set(str(self.input_path))
        default_output = self._default_output_for(self.input_path)
        self.output_var.set(str(default_output))
        self.manual_crops.clear()
        self._load_media_files()
        self._refresh_output_list()

    def _load_media_files(self) -> None:
        self.media_files.clear()
        self.image_files.clear()
        self.listbox.delete(0, tk.END)
        self._list_paths.clear()
        self.canvas.delete("all")
        self.current_path = None
        self.current_image = None
        self._tk_image = None
        self.crop_info_var.set("Kein Bild ausgewählt.")
        self._set_controls_enabled(False)
        self.position_var.set("0/0")

        if not self.input_path:
            return

        files = [self._normalize_path(path) for path in iter_media_files(self.input_path)]
        files.sort()
        self.media_files = files
        for media in self.media_files:
            display = media.relative_to(self.input_path)
            prefix = "🖼️ " if is_image(media) else "🎞️ "
            self.listbox.insert(tk.END, f"{prefix}{display}")
            self._list_paths.append(media)
            if is_image(media):
                self.image_files.append(media)

        if self.image_files:
            first_image = self.image_files[0]
            index = self._list_paths.index(first_image)
            self.listbox.selection_set(index)
            self.listbox.see(index)
            self._on_listbox_select()
            video_count = len(self.media_files) - len(self.image_files)
            if video_count:
                self.progress_var.set(
                    f"{len(self.media_files)} Dateien geladen – {len(self.image_files)} Bilder, {video_count} Videos."
                )
            else:
                self.progress_var.set(f"{len(self.image_files)} Bilder geladen.")
        else:
            self.progress_var.set("Keine unterstützten Bilder gefunden.")
            if self.media_files:
                self.progress_var.set(
                    f"{len(self.media_files)} Videos gefunden. Bitte Bilder hinzufügen, um Zuschnitte zu bearbeiten."
                )
            self._show_placeholder("Keine Bilder verfügbar.")
        self._update_navigation_state()

    # ------------------------------------------------------------------
    # Preview & manual crop
    # ------------------------------------------------------------------
    def _on_listbox_select(self) -> None:
        if not self._list_paths:
            return
        selection = self.listbox.curselection()
        if not selection:
            return
        index = selection[0]
        path = self._list_paths[index]
        if not path.exists():
            self.progress_var.set("Datei nicht gefunden.")
            return
        if is_image(path):
            self._load_preview(path)
        else:
            self._show_placeholder("Video ausgewählt – keine Vorschau verfügbar.")
            self._set_controls_enabled(False)
            self.current_path = None
            self.current_image = None
        self._update_navigation_state()

    def _load_preview(self, path: Path) -> None:
        self.current_path = path
        with Image.open(path) as img:
            self.current_image = img.copy()
        manual = self.manual_crops.get(path)
        if manual is None:
            auto_crop = self._auto_crop_current()
            manual = self._create_manual_from_auto(auto_crop)
        else:
            manual = self._normalize_manual(manual)
        self.manual_crops[path] = manual
        self._apply_manual_to_controls(manual)
        self._set_controls_enabled(True)
        self._update_navigation_state()

    def _auto_crop_current(self) -> CropBox:
        assert self.current_image is not None and self.input_path is not None
        detection_mode = self._current_detection_mode()
        options = ProcessingOptions(
            input_path=self.input_path,
            output_dir=self._resolve_output_dir() or self._default_output_for(self.input_path),
            size=self.size_var.get(),
            face_detection_enabled=detection_mode != "none",
            detection_mode=detection_mode,
            motion_enabled=self.motion_enabled_var.get(),
        )
        cropper = self._get_preview_cropper()
        return determine_crop_box(self.current_image, options, cropper)

    def _scale_crop(self, crop: CropBox, factor: float, width: int, height: int) -> CropBox:
        factor = clamp(factor, 0.01, 10.0)
        size = clamp(crop.size * factor, 1.0, float(min(width, height)))
        center_x = crop.x + crop.size / 2
        center_y = crop.y + crop.size / 2
        max_x = max(0.0, width - size)
        max_y = max(0.0, height - size)
        x = clamp(center_x - size / 2, 0.0, max_x)
        y = clamp(center_y - size / 2, 0.0, max_y)
        return CropBox(x=x, y=y, size=size)

    def _create_manual_from_auto(self, crop: CropBox) -> ManualCrop:
        assert self.current_image is not None
        width, height = self.current_image.size
        end = CropBox(crop.x, crop.y, crop.size)
        has_margin = (
            crop.x > 1.0
            and crop.y > 1.0
            and crop.x + crop.size < width - 1.0
            and crop.y + crop.size < height - 1.0
        )
        if self.motion_enabled_var.get() and has_margin:
            start = self._scale_crop(end, 0.9, width, height)
        else:
            start = CropBox(end.x, end.y, end.size)
        return ManualCrop(start=start, end=end)

    def _normalize_crop_box(self, crop: CropBox, width: int, height: int) -> CropBox:
        size = clamp(crop.size, 1.0, float(min(width, height)))
        max_x = max(0.0, width - size)
        max_y = max(0.0, height - size)
        x = clamp(crop.x, 0.0, max_x)
        y = clamp(crop.y, 0.0, max_y)
        return CropBox(x=x, y=y, size=size)

    def _normalize_manual(self, manual: ManualCrop) -> ManualCrop:
        assert self.current_image is not None
        width, height = self.current_image.size
        start = self._normalize_crop_box(manual.start, width, height)
        end = self._normalize_crop_box(manual.end, width, height)
        return ManualCrop(start=start, end=end)

    def _active_manual_crop(self, manual: ManualCrop) -> CropBox:
        if self.motion_enabled_var.get() and self.active_crop_var.get() == "start":
            return manual.start
        return manual.end

    def _sync_sliders_with_active(self, manual: ManualCrop) -> None:
        if self.current_image is None:
            return
        crop = self._active_manual_crop(manual)
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
        self._updating_controls = False

    def _update_crop_info(self, manual: ManualCrop) -> None:
        if self.motion_enabled_var.get():
            start = manual.start
            end = manual.end
            self.crop_info_var.set(
                " | ".join(
                    [
                        f"Start: {int(start.size)}px – ({int(start.x)}, {int(start.y)})",
                        f"Ende: {int(end.size)}px – ({int(end.x)}, {int(end.y)})",
                    ]
                )
            )
        else:
            end = manual.end
            self.crop_info_var.set(
                f"Ausschnitt: {int(end.size)}px – Position ({int(end.x)}, {int(end.y)})"
            )

    def _update_current_manual(self, manual: ManualCrop, sync_controls: bool = True) -> None:
        if self.current_image is None or self.current_path is None:
            return
        normalized = self._normalize_manual(manual)
        self.manual_crops[self.current_path] = normalized
        if sync_controls:
            self._sync_sliders_with_active(normalized)
        self._render_preview(normalized)
        self._update_position_label()

    def _apply_manual_to_controls(self, manual: ManualCrop) -> None:
        self._update_current_manual(manual, sync_controls=True)

    def _on_motion_toggle(self) -> None:
        enabled = self.motion_enabled_var.get()
        if enabled:
            self.start_radio.state(["!disabled"])
        else:
            self.start_radio.state(["disabled"])
            if self.active_crop_var.get() == "start":
                self.active_crop_var.set("end")
        if self.current_path is None or self.current_path not in self.manual_crops or self.current_image is None:
            self._refresh_selected_button_state()
            return
        manual = self.manual_crops[self.current_path]
        if enabled:
            if (
                abs(manual.start.x - manual.end.x) < 1e-3
                and abs(manual.start.y - manual.end.y) < 1e-3
                and abs(manual.start.size - manual.end.size) < 1e-3
            ):
                manual = self._create_manual_from_auto(manual.end)
                self.manual_crops[self.current_path] = manual
        self._apply_manual_to_controls(manual)
        self._refresh_selected_button_state()

    def _on_active_crop_change(self, *_args: object) -> None:
        if not self.motion_enabled_var.get():
            if self.active_crop_var.get() != "end":
                self.active_crop_var.set("end")
            return
        if self.current_path is None or self.current_path not in self.manual_crops:
            return
        manual = self.manual_crops[self.current_path]
        self._sync_sliders_with_active(manual)
        self._render_preview(manual)

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
        new_crop = CropBox(x=x, y=y, size=size)
        manual = self.manual_crops.get(self.current_path)
        if manual is None:
            auto = self._auto_crop_current()
            manual = self._create_manual_from_auto(auto)
        start = CropBox(manual.start.x, manual.start.y, manual.start.size)
        end = CropBox(manual.end.x, manual.end.y, manual.end.size)
        if self.motion_enabled_var.get():
            if self.active_crop_var.get() == "start":
                start = new_crop
            else:
                end = new_crop
        else:
            start = new_crop
            end = new_crop
        self._update_current_manual(ManualCrop(start=start, end=end), sync_controls=True)

    def _canvas_rect(self, crop: CropBox) -> tuple[float, float, float, float]:
        offset_x, offset_y = self._canvas_offset
        scale = self._canvas_scale
        return (
            offset_x + crop.x * scale,
            offset_y + crop.y * scale,
            offset_x + (crop.x + crop.size) * scale,
            offset_y + (crop.y + crop.size) * scale,
        )

    def _detect_handle(self, rect: tuple[float, float, float, float], x: float, y: float) -> Optional[str]:
        handle_range = 10.0
        x1, y1, x2, y2 = rect
        corners = {
            "nw": (x1, y1),
            "ne": (x2, y1),
            "sw": (x1, y2),
            "se": (x2, y2),
        }
        for name, (cx, cy) in corners.items():
            if abs(x - cx) <= handle_range and abs(y - cy) <= handle_range:
                return name
        return None

    def _draw_handles(self, rect: tuple[float, float, float, float], color: str) -> None:
        handle = 6
        x1, y1, x2, y2 = rect
        for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
            self.canvas.create_rectangle(
                x - handle,
                y - handle,
                x + handle,
                y + handle,
                outline=color,
                fill=color,
                tags=("handle",),
            )

    def _render_preview(self, manual: ManualCrop) -> None:
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

        self._canvas_scale = scale
        self._canvas_offset = (offset_x, offset_y)
        self._manual_display = {}

        active_target = "start" if self.motion_enabled_var.get() and self.active_crop_var.get() == "start" else "end"

        if self.motion_enabled_var.get():
            start_rect = self._canvas_rect(manual.start)
            end_rect = self._canvas_rect(manual.end)
            self._manual_display["start"] = start_rect
            self._manual_display["end"] = end_rect
            self.canvas.create_rectangle(*start_rect, outline="#ff5555", width=2)
            self.canvas.create_rectangle(*end_rect, outline="#00ff88", width=3 if active_target == "end" else 2)
            if active_target == "start":
                self.canvas.create_rectangle(*start_rect, outline="#ff5555", width=3)
                self._draw_handles(start_rect, "#ff5555")
            else:
                self._draw_handles(end_rect, "#00ff88")
        else:
            end_rect = self._canvas_rect(manual.end)
            self._manual_display["end"] = end_rect
            self.canvas.create_rectangle(*end_rect, outline="#00ff88", width=3)
            self._draw_handles(end_rect, "#00ff88")

        self._update_crop_info(manual)
        has_prev, has_next = self._navigation_flags()
        self._update_canvas_navigation(has_prev, has_next)

    def _show_placeholder(self, message: str) -> None:
        self.canvas.delete("all")
        self._tk_image = None
        self._manual_display = {}
        self.crop_info_var.set(message)

    def _navigation_flags(self) -> tuple[bool, bool]:
        if not self.image_files or self.current_path not in self.image_files:
            return False, False
        index = self.image_files.index(self.current_path)
        has_prev = index > 0
        has_next = index < len(self.image_files) - 1
        return has_prev, has_next

    def _update_canvas_navigation(self, has_prev: bool, has_next: bool) -> None:
        self.canvas.delete("nav")
        self.canvas.tag_unbind("nav_prev", "<Button-1>")
        self.canvas.tag_unbind("nav_next", "<Button-1>")
        if self._tk_image is None:
            return

        radius = 18
        center_y = self.CANVAS_SIZE / 2
        left_center = radius + 6
        right_center = self.CANVAS_SIZE - radius - 6

        def draw_arrow(center_x: float, enabled: bool, tag: str) -> None:
            background = "#1f6feb" if enabled else "#101321"
            foreground = "#ffffff" if enabled else "#2f3548"
            tags = ("nav", tag)
            self.canvas.create_oval(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                fill=background,
                outline="",
                tags=tags,
            )
            if tag == "nav_prev":
                points = [
                    center_x + 6,
                    center_y - 10,
                    center_x - 6,
                    center_y,
                    center_x + 6,
                    center_y + 10,
                ]
            else:
                points = [
                    center_x - 6,
                    center_y - 10,
                    center_x + 6,
                    center_y,
                    center_x - 6,
                    center_y + 10,
                ]
            self.canvas.create_polygon(points, fill=foreground, outline="", tags=tags)
            if enabled:
                if tag == "nav_prev":
                    self.canvas.tag_bind("nav_prev", "<Button-1>", lambda _e: self._show_previous_image())
                else:
                    self.canvas.tag_bind("nav_next", "<Button-1>", lambda _e: self._show_next_image())

        draw_arrow(left_center, has_prev, "nav_prev")
        draw_arrow(right_center, has_next, "nav_next")

    def _resize_crop_with_handle(
        self, crop: CropBox, handle: str, dx: float, dy: float, width: int, height: int
    ) -> CropBox:
        x1 = crop.x
        y1 = crop.y
        x2 = crop.x + crop.size
        y2 = crop.y + crop.size
        if handle == "se":
            new_x2 = x2 + dx
            new_y2 = y2 + dy
            size = max(1.0, max(new_x2 - x1, new_y2 - y1))
            x = x1
            y = y1
        elif handle == "sw":
            new_x1 = x1 + dx
            new_y2 = y2 + dy
            size = max(1.0, max(x2 - new_x1, new_y2 - y1))
            x = x2 - size
            y = y1
        elif handle == "ne":
            new_x2 = x2 + dx
            new_y1 = y1 + dy
            size = max(1.0, max(new_x2 - x1, y2 - new_y1))
            x = x1
            y = y2 - size
        else:  # "nw"
            new_x1 = x1 + dx
            new_y1 = y1 + dy
            size = max(1.0, max(x2 - new_x1, y2 - new_y1))
            x = x2 - size
            y = y2 - size
        resized = CropBox(x=x, y=y, size=size)
        return self._normalize_crop_box(resized, width, height)

    def _on_canvas_press(self, event: tk.Event) -> None:
        tags = self.canvas.gettags("current")
        if "nav_prev" in tags:
            self._show_previous_image()
            return
        if "nav_next" in tags:
            self._show_next_image()
            return
        if self.current_path is None or self.current_path not in self.manual_crops or self.current_image is None:
            return
        manual = self.manual_crops[self.current_path]
        candidates = []
        if self.motion_enabled_var.get():
            candidates.extend(["start", "end"])
        else:
            candidates.append("end")
        # Prefer the currently active crop if available
        if self.motion_enabled_var.get() and self.active_crop_var.get() in candidates:
            candidates.remove(self.active_crop_var.get())
            candidates.insert(0, self.active_crop_var.get())

        target: Optional[str] = None
        mode = "move"
        handle: Optional[str] = None
        for key in candidates:
            rect = self._manual_display.get(key)
            if rect is None:
                continue
            handle = self._detect_handle(rect, event.x, event.y)
            inside = rect[0] <= event.x <= rect[2] and rect[1] <= event.y <= rect[3]
            if handle or inside:
                target = key
                mode = "resize" if handle else "move"
                break
        if target is None:
            self._drag_state = None
            return
        if target != self.active_crop_var.get():
            self.active_crop_var.set(target)
        crop = manual.start if target == "start" else manual.end
        self._drag_state = {
            "target": target,
            "mode": mode,
            "handle": handle,
            "start": CropBox(crop.x, crop.y, crop.size),
            "event": (event.x, event.y),
        }

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if not self._drag_state or self.current_image is None or self.current_path is None:
            return
        if self._conversion_active:
            return
        manual = self.manual_crops.get(self.current_path)
        if manual is None:
            return
        target = self._drag_state.get("target")
        if target not in ("start", "end"):
            return
        start_crop: CropBox = self._drag_state["start"]  # type: ignore[index]
        mode: str = self._drag_state["mode"]  # type: ignore[index]
        handle = self._drag_state.get("handle")
        start_event_x, start_event_y = self._drag_state["event"]  # type: ignore[index]
        dx_canvas = event.x - start_event_x
        dy_canvas = event.y - start_event_y
        scale = self._canvas_scale or 1.0
        dx = dx_canvas / scale
        dy = dy_canvas / scale
        width, height = self.current_image.size
        if mode == "move":
            new_crop = CropBox(
                x=start_crop.x + dx,
                y=start_crop.y + dy,
                size=start_crop.size,
            )
            new_crop = self._normalize_crop_box(new_crop, width, height)
        else:
            if handle is None:
                return
            new_crop = self._resize_crop_with_handle(start_crop, handle, dx, dy, width, height)
        start = CropBox(manual.start.x, manual.start.y, manual.start.size)
        end = CropBox(manual.end.x, manual.end.y, manual.end.size)
        if target == "start":
            start = new_crop
        else:
            end = new_crop
        if not self.motion_enabled_var.get():
            start = new_crop
            end = new_crop
        self._update_current_manual(ManualCrop(start=start, end=end), sync_controls=True)

    def _on_canvas_release(self, _event: tk.Event) -> None:
        self._drag_state = None

    def _update_position_label(self) -> None:
        if not self.image_files or self.current_path is None:
            self.position_var.set("0/0")
            return
        try:
            index = self.image_files.index(self.current_path)
        except ValueError:
            self.position_var.set("0/0")
            return
        self.position_var.set(f"{index + 1}/{len(self.image_files)}")

    def _update_navigation_state(self) -> None:
        if not self.image_files:
            self.prev_button.state(["disabled"])
            self.next_button.state(["disabled"])
            self.position_var.set("0/0")
            self._update_canvas_navigation(False, False)
            return
        if self.current_path is None or self.current_path not in self.image_files:
            self.prev_button.state(["disabled"])
            self.next_button.state(["disabled"])
            self.position_var.set(f"0/{len(self.image_files)}")
            self._update_canvas_navigation(False, False)
            return
        index = self.image_files.index(self.current_path)
        has_prev = index > 0
        has_next = index < len(self.image_files) - 1
        self.prev_button.state(["!disabled"] if has_prev else ["disabled"])
        self.next_button.state(["!disabled"] if has_next else ["disabled"])
        self.position_var.set(f"{index + 1}/{len(self.image_files)}")
        self._update_canvas_navigation(has_prev, has_next)

    def _show_previous_image(self) -> None:
        if self.current_path is None or self.current_path not in self.image_files:
            return
        index = self.image_files.index(self.current_path)
        if index == 0:
            return
        next_path = self.image_files[index - 1]
        list_index = self._list_paths.index(next_path)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(list_index)
        self.listbox.see(list_index)
        self._load_preview(next_path)

    def _show_next_image(self) -> None:
        if self.current_path is None or self.current_path not in self.image_files:
            return
        index = self.image_files.index(self.current_path)
        if index >= len(self.image_files) - 1:
            return
        next_path = self.image_files[index + 1]
        list_index = self._list_paths.index(next_path)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(list_index)
        self.listbox.see(list_index)
        self._load_preview(next_path)

    def _reset_crop_to_auto(self) -> None:
        if self.current_image is None or self.current_path is None:
            return
        crop = self._auto_crop_current()
        manual = self._create_manual_from_auto(crop)
        self.manual_crops[self.current_path] = manual
        self._apply_manual_to_controls(manual)

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

    def _refresh_output_list(self) -> None:
        self.output_media_files.clear()
        self.output_listbox.delete(0, tk.END)
        output_dir = self._resolve_output_dir()
        if output_dir is None:
            self.output_info_var.set("Kein Ausgabeordner gewählt.")
            return
        if not output_dir.exists():
            self.output_info_var.set("Ausgabeordner wird beim Konvertieren erstellt.")
            return
        videos = sorted(path for path in output_dir.iterdir() if is_video(path))
        for video in videos:
            self.output_listbox.insert(tk.END, f"🎬 {video.name}")
            self.output_media_files.append(video)
        if videos:
            self.output_info_var.set(f"{len(videos)} Videos im Ausgabeordner.")
        else:
            self.output_info_var.set("Keine Videos im Ausgabeordner.")

    def _open_output_file(self, _event: tk.Event) -> None:
        if not self.output_media_files:
            return
        selection = self.output_listbox.curselection()
        if not selection:
            return
        path = self.output_media_files[selection[0]]
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Fehler", f"Datei kann nicht geöffnet werden: {path.name}\n{exc}")

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def _manual_overrides_copy(self) -> dict[Path, ManualCrop]:
        return {path: crop.copy() for path, crop in self.manual_crops.items()}

    def _start_conversion(
        self,
        output_dir: Path,
        manual_overrides: dict[Path, ManualCrop],
        files_subset: Optional[list[Path]] = None,
    ) -> None:
        ensure_dir(output_dir)
        self._conversion_active = True
        self.convert_button.state(["disabled"])
        self.convert_selected_button.state(["disabled"])
        self.progress_var.set("Konvertierung läuft…")
        thread = threading.Thread(
            target=self._run_batch,
            args=(output_dir, manual_overrides, files_subset),
            daemon=True,
        )
        thread.start()

    def _on_convert(self) -> None:
        if self.input_path is None:
            messagebox.showerror("Fehler", "Bitte zuerst einen Eingabeordner wählen.")
            return
        output_dir = self._resolve_output_dir()
        if output_dir is None:
            messagebox.showerror("Fehler", "Bitte einen Ausgabeordner wählen oder eingeben.")
            return
        self._start_conversion(output_dir, self._manual_overrides_copy())

    def _on_convert_selected(self) -> None:
        if self.input_path is None:
            messagebox.showerror("Fehler", "Bitte zuerst einen Eingabeordner wählen.")
            return
        output_dir = self._resolve_output_dir()
        if output_dir is None:
            messagebox.showerror("Fehler", "Bitte einen Ausgabeordner wählen oder eingeben.")
            return
        selection = list(self.listbox.curselection())
        selected_paths: list[Path] = []
        for index in selection:
            try:
                candidate = self._list_paths[index]
            except IndexError:
                continue
            if candidate.exists() and is_image(candidate):
                selected_paths.append(self._normalize_path(candidate))
        if not selected_paths and self.current_path is not None and is_image(self.current_path):
            selected_paths.append(self._normalize_path(self.current_path))
        if not selected_paths:
            messagebox.showinfo("Hinweis", "Bitte ein Bild auswählen, das konvertiert werden soll.")
            return
        unique_paths = []
        seen = set()
        for path in selected_paths:
            if path not in seen:
                unique_paths.append(path)
                seen.add(path)
        self._start_conversion(output_dir, self._manual_overrides_copy(), unique_paths)

    def _run_batch(
        self,
        output_dir: Path,
        manual_overrides: dict[Path, ManualCrop],
        files_subset: Optional[list[Path]] = None,
    ) -> None:
        assert self.input_path is not None
        detection_mode = self._current_detection_mode()
        options = ProcessingOptions(
            input_path=self.input_path,
            output_dir=output_dir,
            size=self.size_var.get(),
            face_detection_enabled=detection_mode != "none",
            detection_mode=detection_mode,
            motion_enabled=self.motion_enabled_var.get(),
        )
        try:
            logger = setup_environment(options)
        except SystemExit:
            self.after(0, lambda: self._handle_error("ffmpeg/ffprobe nicht gefunden. Bitte installieren."))
            return

        if files_subset is not None:
            files = [self._normalize_path(path) for path in files_subset]
        else:
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
        self._conversion_active = False
        self.convert_button.state(["!disabled"])
        self._refresh_selected_button_state()

    def _finish_batch(self) -> None:
        self.progress_var.set("Fertig.")
        self._conversion_active = False
        self.convert_button.state(["!disabled"])
        self._refresh_selected_button_state()
        messagebox.showinfo("Fertig", "Alle Dateien wurden konvertiert.")
        self._refresh_output_list()


def launch_gui() -> None:
    app = Application()
    app.mainloop()
