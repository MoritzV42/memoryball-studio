"""Interaktive Tkinter-Oberfläche für MemoryBall Studio."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from PIL import Image, ImageDraw, ImageTk
try:  # Pillow 9.1+
    from PIL import ImageOps
except ImportError:  # pragma: no cover - fallback for very old Pillow
    ImageOps = None  # type: ignore[assignment]

try:  # Pillow 9.1+
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - Pillow < 9.1
    RESAMPLE_LANCZOS = Image.LANCZOS

from .face_cropper import FaceCropper
from .image_pipeline import determine_crop_box, determine_motion_manual, process_image
from .utils import (
    CropBox,
    ManualCrop,
    ProcessingOptions,
    ORIENTATION_CIRCLE_MARGIN,
    clamp,
    crop_position_bounds,
    ensure_dir,
    iter_media_files,
    is_image,
    is_video,
    max_crop_size,
    normalize_crop_with_overflow,
    setup_environment,
)
from .video_pipeline import process_video


@dataclass
class MemoryCard:
    path: Path
    face_image: ImageTk.PhotoImage
    button: tk.Button
    revealed: bool = False
    matched: bool = False


@dataclass
class MemoryGameState:
    frame: tk.Frame
    cards: list[MemoryCard]
    token: object
    progressbar: ttk.Progressbar
    progress_var: tk.StringVar
    back_image: ImageTk.PhotoImage
    total: int = 0
    first_index: Optional[int] = None
    lock: bool = False
    completed: bool = False
    analysis_done: bool = False


class Application(tk.Tk):
    """Tkinter-Anwendung mit Vorschau und manueller Zuschnittssteuerung."""

    CANVAS_SIZE = 520
    CIRCLE_MARGIN = ORIENTATION_CIRCLE_MARGIN
    PREVIEW_OVERFLOW_RATIO = 0.02
    MOTION_DIRECTION_CHOICES = [
        ("in", "Reinzoomen"),
        ("out", "Rauszoomen"),
    ]
    OFFSET_STEP = 0.03
    ZOOM_STEP = 0.05
    MIN_ZOOM_RATIO = 0.25
    MAX_ZOOM_RATIO = 1.0
    CENTER_VALUE = 0.5
    MIN_MEMORY_CARD_SIZE = 56
    MEMORY_CARD_GAP = 8

    def __init__(self) -> None:
        super().__init__()
        self.title("MemoryBall Studio")
        self.geometry("960x640")
        self.minsize(900, 600)
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda _event: self.attributes("-fullscreen", False))

        self._configure_style()

        self.input_path: Optional[Path] = None
        self.media_files: list[Path] = []
        self.image_files: list[Path] = []
        self.manual_crops: dict[Path, ManualCrop] = {}
        self._auto_generated_paths: set[Path] = set()
        self._list_paths: list[Path] = []
        self._list_iids: list[str] = []
        self._thumbnail_cache: dict[Path, ImageTk.PhotoImage] = {}
        self._video_thumbnail: Optional[ImageTk.PhotoImage] = None
        self.current_path: Optional[Path] = None
        self.current_image: Optional[Image.Image] = None
        self._tk_image: Optional[ImageTk.PhotoImage] = None
        self._preview_cropper: Optional[FaceCropper] = None
        self._updating_controls = False
        self.output_media_files: list[Path] = []
        self._legend_items: dict[str, dict[str, object]] = {}
        self._start_color = "#00ff88"
        self._end_color = "#ff5555"
        self._legend_colors = {"start": self._start_color, "end": self._end_color}
        self._crop_buttons: dict[str, ttk.Button] = {}
        self._crop_buttons_enabled = True
        self._tutorial_window: Optional[tk.Toplevel] = None
        self._tutorial_steps: list[dict[str, object]] = []
        self._tutorial_index = -1
        self._tutorial_running = False
        self._tutorial_completed = self._load_tutorial_completed()
        self._control_mode_var = tk.StringVar(value="regulators")
        self._regulator_frame: Optional[ttk.Frame] = None
        self._dpad_frame: Optional[ttk.Frame] = None
        self._compact_control_buttons: list[tk.Widget] = []
        self.crop_button_frame: Optional[ttk.Frame] = None
        self.legend_frame: Optional[tk.Widget] = None
        self._loading_overlay: Optional[tk.Frame] = None
        self._loading_spinner: Optional[ttk.Progressbar] = None
        self._loading_message_var = tk.StringVar(value="")
        self._auto_task_token: Optional[object] = None
        self._bulk_auto_token: Optional[object] = None
        self._bulk_auto_thread: Optional[threading.Thread] = None
        self._memory_container: Optional[ttk.Frame] = None
        self._memory_game_state: Optional[MemoryGameState] = None
        self._memory_flip_job: Optional[str] = None
        self._settings_collapsed = False
        self._settings_container: Optional[ttk.Frame] = None
        self._settings_toggle_button: Optional[ttk.Button] = None

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.size_var = tk.IntVar(value=480)
        self.size_ratio = tk.DoubleVar(value=1.0)
        self.offset_x = tk.DoubleVar(value=0.0)
        self.offset_y = tk.DoubleVar(value=0.0)
        self.motion_enabled_var = tk.BooleanVar(value=True)
        self.active_crop_var = tk.StringVar(value="end")
        self.motion_direction_var = tk.StringVar(value="in")
        self._motion_direction_label_by_value = {
            value: label for value, label in self.MOTION_DIRECTION_CHOICES
        }
        self._motion_direction_value_by_label = {
            label: value for value, label in self.MOTION_DIRECTION_CHOICES
        }
        self.motion_direction_label_var = tk.StringVar(
            value=self._motion_direction_label_by_value["in"]
        )
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
        self._update_motion_direction_state()
        self._load_reference_preview()
        self.after(1000, self._maybe_start_tutorial)
        self.active_crop_var.trace_add("write", self._on_active_crop_change)

    # ------------------------------------------------------------------
    # Layout & UI
    # ------------------------------------------------------------------
    def _configure_style(self) -> None:
        background = "#050b18"
        card_background = "#0e1629"
        accent = "#4f78ff"
        success = "#2fdf84"
        danger = "#ff6b6b"
        self._background_color = background
        self._card_background = card_background
        self._accent_color = accent
        self._success_color = success
        self._danger_color = danger
        self.configure(background=background)
        self.option_add("*Font", "{Segoe UI} 10")
        self.option_add("*Label.font", "{Segoe UI} 10")
        self.option_add("*Entry.background", card_background)
        self.option_add("*Entry.foreground", "#e4ebff")
        self.option_add("*Entry.insertBackground", "#e4ebff")
        self.option_add("*Spinbox.background", card_background)
        self.option_add("*Spinbox.foreground", "#e4ebff")
        self.option_add("*Spinbox.insertBackground", "#e4ebff")
        self.option_add("*TCombobox*Listbox.background", card_background)
        self.option_add("*TCombobox*Listbox.foreground", "#e4ebff")
        self.option_add("*TCombobox*Listbox.selectBackground", accent)
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=background)
        style.configure("Card.TFrame", background=card_background, relief="flat")
        style.configure("TLabel", background=background, foreground="#e4ebff")
        style.configure("Title.TLabel", background=background, foreground="#f4f7ff", font=("Segoe UI", 16, "bold"))
        style.configure("Subtitle.TLabel", background=background, foreground="#8d9ac0", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=card_background, foreground="#9aa7c6", font=("Segoe UI", 9, "bold"))
        style.configure("Heading.TLabel", background=card_background, foreground="#f4f7ff", font=("Segoe UI", 12, "bold"))
        style.configure("Body.TLabel", background=card_background, foreground="#cbd5ff")
        style.configure("Status.TLabel", background=background, foreground="#8d9ac0", font=("Segoe UI", 9))
        style.configure("Tutorial.TLabel", background=card_background, foreground="#f4f7ff")
        style.configure("TutorialHeading.TLabel", background=card_background, foreground="#f4f7ff", font=("Segoe UI", 12, "bold"))
        style.configure("TButton", padding=8, background=card_background, foreground="#e4ebff")
        style.configure("TCheckbutton", background=card_background, foreground="#cbd5ff")
        style.configure("TRadiobutton", background=card_background, foreground="#cbd5ff")
        style.configure(
            "Accent.TButton",
            background=accent,
            foreground="#ffffff",
            padding=10,
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#325df6"), ("pressed", "#284fe3"), ("disabled", "#1b2033")],
            foreground=[("disabled", "#5f6a8e")],
        )
        style.configure(
            "Start.TButton",
            background=success,
            foreground="#02131f",
            padding=(12, 6),
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 11, "bold"),
        )
        style.map(
            "Start.TButton",
            background=[("active", "#4af29a"), ("pressed", "#24c375"), ("disabled", "#1b2033")],
            relief=[("pressed", "sunken"), ("!pressed", "flat")],
        )
        style.configure(
            "StartActive.TButton",
            background="#47ffad",
            foreground="#02131f",
            padding=(12, 6),
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 11, "bold"),
        )
        style.map(
            "StartActive.TButton",
            background=[("active", "#60ffba"), ("pressed", "#32d98f"), ("disabled", "#1b2033")],
            relief=[("pressed", "sunken"), ("!pressed", "flat")],
        )
        style.configure(
            "End.TButton",
            background=danger,
            foreground="#02131f",
            padding=(12, 6),
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 11, "bold"),
        )
        style.map(
            "End.TButton",
            background=[("active", "#ff8080"), ("pressed", "#ff5252"), ("disabled", "#1b2033")],
            relief=[("pressed", "sunken"), ("!pressed", "flat")],
        )
        style.configure(
            "EndActive.TButton",
            background="#ff8a8a",
            foreground="#02131f",
            padding=(12, 6),
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 11, "bold"),
        )
        style.map(
            "EndActive.TButton",
            background=[("active", "#ff9c9c"), ("pressed", "#ff6b6b"), ("disabled", "#1b2033")],
            relief=[("pressed", "sunken"), ("!pressed", "flat")],
        )
        style.configure(
            "Nav.TButton",
            background=card_background,
            foreground="#c7d3ff",
            padding=(10, 6),
            borderwidth=1,
            relief="solid",
            focusthickness=0,
        )
        style.map(
            "Nav.TButton",
            background=[("active", "#1a2642"), ("pressed", "#233354")],
            foreground=[("disabled", "#5f6a8e")],
        )
        style.configure(
            "Modern.TCombobox",
            fieldbackground=card_background,
            background=card_background,
            foreground="#e4ebff",
        )
        style.map(
            "Modern.TCombobox",
            fieldbackground=[("readonly", card_background)],
            background=[("readonly", card_background)],
        )
        style.configure("Horizontal.TScale", background=card_background, troughcolor="#1b2945")
        style.configure(
            "Modern.TSpinbox",
            fieldbackground=card_background,
            background=card_background,
            foreground="#e4ebff",
        )
        style.map("Modern.TSpinbox", fieldbackground=[("readonly", card_background)])
        style.configure(
            "Media.Treeview",
            background="#0b0f1c",
            fieldbackground="#0b0f1c",
            foreground="#f5f7fa",
            bordercolor="#0b0f1c",
            rowheight=56,
        )
        style.map(
            "Media.Treeview",
            background=[("selected", "#1f6feb")],
            foreground=[("selected", "#ffffff")],
        )

    def _find_reference_image(self) -> Optional[Path]:
        base_dir = Path(__file__).resolve().parent.parent
        reference_dir = base_dir / "Referenz"
        if reference_dir.is_dir():
            for candidate in sorted(reference_dir.iterdir()):
                if candidate.is_file() and is_image(candidate):
                    return candidate
        return None

    def _load_reference_preview(self) -> None:
        reference = self._find_reference_image()
        if reference is None:
            self._show_placeholder("Kein Bild ausgewählt.")
            return
        try:
            self._load_preview(self._normalize_path(reference))
        except Exception:
            self._show_placeholder("Kein Bild ausgewählt.")
            return
        self.progress_var.set("Referenzbild geladen. Wähle eine Datei oder einen Ordner aus.")
        self.crop_info_var.set("Passe den Ausschnitt oder lade eigene Dateien.")
        if self.input_path is None and self.current_path is not None:
            self.after(
                400,
                lambda path=self.current_path: self._start_auto_detection(
                    path, message="Analysiere Referenz…"
                ),
            )

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main = ttk.Frame(self, padding=20)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=0)
        main.rowconfigure(1, weight=1)
        main.rowconfigure(2, weight=0)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, columnspan=3, sticky="ew")
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=1)
        title_frame = ttk.Frame(header)
        title_frame.grid(row=0, column=0, rowspan=2, sticky="w")
        ttk.Label(title_frame, text="Memory Ball Studio", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            title_frame,
            text="Intelligente Ausrichtung für Fotos & Videos",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        io_header = ttk.Frame(header)
        io_header.grid(row=0, column=1, rowspan=2, sticky="new", padx=(24, 0))
        io_header.columnconfigure(1, weight=1)

        ttk.Label(io_header, text="Eingabequelle", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.input_entry = ttk.Entry(io_header, textvariable=self.input_var, width=32)
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(io_header, text="Wählen…", command=self._choose_input).grid(
            row=0, column=2, sticky="ew"
        )

        ttk.Label(io_header, text="Ausgabeordner", style="Section.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.output_entry = ttk.Entry(io_header, textvariable=self.output_var, width=32)
        self.output_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Button(io_header, text="Wählen…", command=self._choose_output).grid(
            row=1, column=2, sticky="ew", pady=(8, 0)
        )

        self.tutorial_button = ttk.Button(
            header,
            text="❔ Tutorial",
            command=self._start_tutorial,
            style="Accent.TButton",
        )
        self.tutorial_button.grid(row=0, column=2, rowspan=2, sticky="ne", padx=(12, 0))

        list_frame = ttk.Frame(main, style="Card.TFrame", padding=20)
        list_frame.grid(row=1, column=0, sticky="nswe", pady=(16, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(1, weight=1)
        ttk.Label(list_frame, text="Bilder & Videos", style="Heading.TLabel").grid(row=0, column=0, sticky="w")

        self.listbox = ttk.Treeview(
            list_frame,
            show="tree",
            selectmode="browse",
            height=20,
            style="Media.Treeview",
        )
        self.listbox.grid(row=1, column=0, sticky="nswe", pady=(6, 0))
        self.listbox.column("#0", anchor="w", stretch=True)
        self.listbox.bind("<<TreeviewSelect>>", lambda _event: self._on_listbox_select())

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scrollbar.set)

        preview = ttk.Frame(main, style="Card.TFrame", padding=20)
        preview.grid(row=1, column=1, sticky="nsew", padx=(16, 0), pady=(16, 0))
        preview.columnconfigure(0, weight=0)
        preview.columnconfigure(1, weight=1)
        preview.rowconfigure(1, weight=1)

        settings_header = ttk.Frame(preview)
        settings_header.grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Label(settings_header, text="Einstellungen", style="Heading.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self._settings_toggle_button = ttk.Button(
            settings_header,
            text="Einklappen",
            command=self._toggle_settings_panel,
            width=12,
        )
        self._settings_toggle_button.grid(row=0, column=1, sticky="w", padx=(12, 0))

        ttk.Label(preview, text="Vorschau", style="Heading.TLabel").grid(
            row=0, column=1, sticky="w"
        )

        controls_container = ttk.Frame(preview)
        controls_container.grid(row=1, column=0, sticky="nsw", pady=(12, 0), padx=(0, 12))
        controls_container.columnconfigure(0, weight=1)
        controls_container.rowconfigure(0, weight=1)
        self._settings_container = controls_container

        controls_canvas = tk.Canvas(
            controls_container,
            background=self._card_background,
            highlightthickness=0,
            bd=0,
            width=260,
        )
        controls_canvas.grid(row=0, column=0, sticky="nsw")
        controls_scrollbar = ttk.Scrollbar(
            controls_container, orient="vertical", command=controls_canvas.yview
        )
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        controls_column = ttk.Frame(controls_canvas)
        controls_window = controls_canvas.create_window(
            (0, 0), window=controls_column, anchor="nw", tags=("controls",)
        )

        def _on_controls_configure(event) -> None:
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        def _resize_controls(event) -> None:
            controls_canvas.itemconfigure(controls_window, width=event.width)

        controls_column.bind("<Configure>", _on_controls_configure)
        controls_canvas.bind("<Configure>", _resize_controls)
        controls_column.columnconfigure(0, weight=1)

        button_frame = ttk.Frame(controls_column)
        button_frame.grid(row=0, column=0, sticky="w")
        self.crop_button_frame = button_frame
        self.legend_frame = button_frame
        self._crop_buttons = {
            "start": ttk.Button(
                button_frame,
                text="Start 1",
                width=8,
                style="Start.TButton",
                command=lambda: self._select_crop("start"),
            ),
            "end": ttk.Button(
                button_frame,
                text="Ende 2",
                width=8,
                style="End.TButton",
                command=lambda: self._select_crop("end"),
            ),
        }
        self._crop_buttons["start"].grid(row=0, column=0, padx=(0, 8))
        self._crop_buttons["end"].grid(row=0, column=1)
        self._refresh_crop_buttons()

        self._compact_control_buttons.clear()

        compact_controls = ttk.Frame(controls_column)
        compact_controls.grid(row=1, column=0, sticky="we", pady=(12, 0))
        compact_controls.columnconfigure(0, weight=0)
        compact_controls.columnconfigure(1, weight=1)

        toggle_frame = ttk.Frame(compact_controls)
        toggle_frame.grid(row=0, column=0, columnspan=2, sticky="w")
        regulators_toggle = ttk.Radiobutton(
            toggle_frame,
            text="Regler",
            value="regulators",
            variable=self._control_mode_var,
            command=self._update_control_mode,
        )
        regulators_toggle.grid(row=0, column=0, sticky="w")
        cross_toggle = ttk.Radiobutton(
            toggle_frame,
            text="Kreuz",
            value="dpad",
            variable=self._control_mode_var,
            command=self._update_control_mode,
        )
        cross_toggle.grid(row=0, column=1, sticky="w", padx=(12, 0))
        self._compact_control_buttons.extend([regulators_toggle, cross_toggle])

        control_stack = ttk.Frame(compact_controls)
        control_stack.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        control_stack.columnconfigure(0, weight=1)

        self._regulator_frame = ttk.Frame(control_stack)
        self._regulator_frame.grid(row=0, column=0, sticky="w")
        self._regulator_frame.columnconfigure(1, weight=1)
        slider_length = 220

        ttk.Label(self._regulator_frame, text="Zoom", style="Body.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.size_scale = ttk.Scale(
            self._regulator_frame,
            from_=0.25,
            to=1.0,
            variable=self.size_ratio,
            command=self._on_slider_change,
        )
        self.size_scale.configure(length=slider_length)
        self.size_scale.grid(row=0, column=1, sticky="w", padx=(6, 6))

        ttk.Label(self._regulator_frame, text="X-Position", style="Body.TLabel").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self.x_scale = ttk.Scale(
            self._regulator_frame,
            from_=0.0,
            to=1.0,
            variable=self.offset_x,
            command=self._on_slider_change,
        )
        self.x_scale.configure(length=slider_length)
        self.x_scale.grid(row=1, column=1, sticky="w", padx=(6, 6), pady=(6, 0))

        ttk.Label(self._regulator_frame, text="Y-Position", style="Body.TLabel").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        self.y_scale = ttk.Scale(
            self._regulator_frame,
            from_=0.0,
            to=1.0,
            variable=self.offset_y,
            command=self._on_slider_change,
        )
        self.y_scale.configure(length=slider_length)
        self.y_scale.grid(row=2, column=1, sticky="w", padx=(6, 6), pady=(6, 0))

        self._dpad_frame = ttk.Frame(control_stack)
        self._dpad_frame.grid(row=0, column=0, sticky="w")
        for index in range(3):
            self._dpad_frame.columnconfigure(index, weight=1)

        def add_dpad_button(
            text: str, grid_row: int, grid_column: int, command: Callable[[], None]
        ) -> ttk.Button:
            button = ttk.Button(self._dpad_frame, text=text, width=3, command=command)
            button.grid(row=grid_row, column=grid_column, padx=2, pady=2)
            self._compact_control_buttons.append(button)
            return button

        add_dpad_button("↑", 0, 1, lambda: self._adjust_offset(0.0, -self.OFFSET_STEP))
        add_dpad_button("+", 0, 2, lambda: self._adjust_zoom(self.ZOOM_STEP))
        add_dpad_button("←", 1, 0, lambda: self._adjust_offset(-self.OFFSET_STEP, 0.0))
        add_dpad_button("●", 1, 1, self._center_offset)
        add_dpad_button("→", 1, 2, lambda: self._adjust_offset(self.OFFSET_STEP, 0.0))
        add_dpad_button("↓", 2, 1, lambda: self._adjust_offset(0.0, self.OFFSET_STEP))
        add_dpad_button("−", 2, 2, lambda: self._adjust_zoom(-self.ZOOM_STEP))

        auto_button = ttk.Button(
            compact_controls, text="Auto", command=self._reset_crop_to_auto
        )
        auto_button.grid(row=2, column=0, sticky="w", pady=(12, 0))
        self._compact_control_buttons.append(auto_button)
        analyze_all_button = ttk.Button(
            compact_controls,
            text="Alle Bilder analysieren",
            command=self._analyze_all_images,
            width=24,
        )
        analyze_all_button.grid(row=2, column=1, sticky="ew", padx=(12, 0), pady=(12, 0))
        self._compact_control_buttons.append(analyze_all_button)

        self._update_control_mode()

        nav = ttk.Frame(controls_column)
        nav.grid(row=2, column=0, sticky="w", pady=(16, 0))
        nav.columnconfigure(1, weight=0)
        nav.columnconfigure(2, weight=0)
        nav.rowconfigure(0, weight=0)
        nav.rowconfigure(1, weight=0)
        nav.rowconfigure(2, weight=0)
        self.prev_button = ttk.Button(
            nav, text="◀", width=3, command=self._show_previous_image, style="Nav.TButton"
        )
        self.prev_button.grid(row=1, column=0, sticky="w")
        ttk.Label(nav, textvariable=self.position_var, style="Section.TLabel").grid(
            row=1, column=1, padx=(8, 12), sticky="w"
        )
        self.next_button = ttk.Button(
            nav, text="▶", width=3, command=self._show_next_image, style="Nav.TButton"
        )
        self.next_button.grid(row=1, column=2, sticky="w", padx=(12, 0))

        motion_controls = ttk.Frame(controls_column)
        motion_controls.grid(row=3, column=0, sticky="w", pady=(12, 0))
        motion_controls.columnconfigure(0, weight=1)
        motion_controls.columnconfigure(1, weight=0)
        ttk.Checkbutton(
            motion_controls,
            text="Bewegung aktiv",
            variable=self.motion_enabled_var,
            command=self._on_motion_toggle,
        ).grid(row=0, column=0, sticky="w")
        self.motion_direction_combo = ttk.Combobox(
            motion_controls,
            values=[label for _value, label in self.MOTION_DIRECTION_CHOICES],
            state="readonly",
            textvariable=self.motion_direction_label_var,
            width=14,
            style="Modern.TCombobox",
        )
        self.motion_direction_combo.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.motion_direction_combo.bind(
            "<<ComboboxSelected>>", self._on_motion_direction_change
        )
        size_frame = ttk.Frame(motion_controls)
        size_frame.grid(row=2, column=0, sticky="w", pady=(12, 0))
        ttk.Label(size_frame, text="Zielgröße", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            size_frame,
            from_=256,
            to=1080,
            increment=16,
            textvariable=self.size_var,
            width=7,
            style="Modern.TSpinbox",
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(controls_column, textvariable=self.crop_info_var, style="Section.TLabel").grid(
            row=6,
            column=0,
            sticky="w",
            pady=(12, 0),
        )

        self.canvas = tk.Canvas(
            preview,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            background="#060d1d",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.grid(row=1, column=1, sticky="n", pady=(12, 12))
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        self._create_loading_overlay(preview)
        output = ttk.Frame(main, style="Card.TFrame", padding=20)
        output.grid(row=1, column=2, sticky="nsw", padx=(16, 0), pady=(16, 0))
        output.columnconfigure(0, weight=1)
        output.rowconfigure(1, weight=1)
        ttk.Label(output, text="Ausgabe", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        self.output_listbox = tk.Listbox(output, exportselection=False, height=20)
        self.output_listbox.grid(row=1, column=0, sticky="nswe", pady=(6, 0))
        self.output_listbox.configure(
            background="#0a1326",
            foreground="#ecf0ff",
            borderwidth=0,
            highlightthickness=0,
            selectbackground="#3f71ff",
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

        ttk.Label(bottom, textvariable=self.progress_var, style="Status.TLabel").grid(row=0, column=0, sticky="w")
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
        self._refresh_legend_state()
        self._update_settings_toggle_button()

    def _toggle_settings_panel(self) -> None:
        self._set_settings_collapsed(not self._settings_collapsed)

    def _set_settings_collapsed(self, collapsed: bool) -> None:
        if self._settings_collapsed == collapsed:
            return
        self._settings_collapsed = collapsed
        container = self._settings_container
        if container is not None:
            if collapsed:
                container.grid_remove()
            else:
                container.grid()
        self._update_settings_toggle_button()
        self.update_idletasks()

    def _update_settings_toggle_button(self) -> None:
        if self._settings_toggle_button is None:
            return
        text = "Ausklappen" if self._settings_collapsed else "Einklappen"
        self._settings_toggle_button.configure(text=text)

    def _create_loading_overlay(self, parent: tk.Widget) -> None:
        overlay = tk.Frame(parent, background=self._card_background, bd=0, highlightthickness=0)
        overlay.configure(padx=24, pady=16)
        message = ttk.Label(overlay, textvariable=self._loading_message_var, style="Body.TLabel")
        message.pack(anchor="center")
        spinner = ttk.Progressbar(overlay, mode="indeterminate", length=160)
        spinner.pack(anchor="center", pady=(12, 0))
        overlay.place_forget()
        self._loading_overlay = overlay
        self._loading_spinner = spinner

    def _show_loading_overlay(self, message: str) -> None:
        if self._loading_overlay is None or self._loading_spinner is None:
            return
        self._loading_message_var.set(message)
        self._loading_overlay.place(relx=0.5, rely=0.5, anchor="center")
        self._loading_overlay.lift()
        self._loading_spinner.start(12)

    def _hide_loading_overlay(self) -> None:
        if self._loading_overlay is None or self._loading_spinner is None:
            return
        self._loading_spinner.stop()
        self._loading_overlay.place_forget()
        self._loading_message_var.set("")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_legend(self, parent: tk.Widget) -> None:
        for child in parent.winfo_children():
            child.destroy()
        self._legend_items.clear()
        entries = [
            ("start", "1", "= Startposition"),
            ("end", "2", "= Endposition"),
        ]
        for index, (key, number, description) in enumerate(entries):
            padx = (0, 6) if index == 0 else (24, 6)
            number_label = tk.Label(
                parent,
                text=number,
                fg=self._legend_colors.get(key, "#ffffff"),
                bg=self._card_background,
                font=("Segoe UI", 11, "bold"),
                borderwidth=0,
                highlightthickness=2,
                highlightbackground=self._legend_colors.get(key, "#ffffff"),
                highlightcolor=self._legend_colors.get(key, "#ffffff"),
                padx=10,
                pady=4,
                cursor="hand2",
            )
            number_label.grid(row=0, column=index * 2, padx=padx)
            number_label.bind("<Button-1>", lambda _e, target=key: self._select_crop(target))
            description_label = ttk.Label(parent, text=description, cursor="hand2", style="Body.TLabel")
            description_label.grid(row=0, column=index * 2 + 1, sticky="w")
            description_label.bind("<Button-1>", lambda _e, target=key: self._select_crop(target))
            self._legend_items[key] = {
                "number": number_label,
                "text": description_label,
            }
        self._refresh_legend_state()

    def _refresh_legend_state(self) -> None:
        if not self._legend_items:
            return
        for key, widgets in self._legend_items.items():
            number_label: tk.Label = widgets["number"]  # type: ignore[assignment]
            text_label: ttk.Label = widgets["text"]  # type: ignore[assignment]
            color = self._legend_colors.get(key, "#ffffff")
            enabled = key == "end" or self.motion_enabled_var.get()
            is_active = enabled and self.active_crop_var.get() == key
            if enabled:
                number_label.configure(cursor="hand2")
                text_label.configure(cursor="hand2")
                if is_active:
                    number_label.configure(
                        bg=color,
                        fg="#02131f",
                        highlightbackground=color,
                        highlightcolor=color,
                    )
                else:
                    number_label.configure(
                        bg=self._card_background,
                        fg=color,
                        highlightbackground=color,
                        highlightcolor=color,
                    )
                text_label.configure(foreground="#e4ebff")
            else:
                number_label.configure(
                    bg=self._card_background,
                    fg="#3f4f78",
                    highlightbackground="#26324e",
                    highlightcolor="#26324e",
                    cursor="",
                )
                text_label.configure(foreground="#4c5c80", cursor="")

    def _update_control_mode(self, *_args: object) -> None:
        mode = self._control_mode_var.get()
        if self._regulator_frame is not None:
            if mode == "regulators":
                self._regulator_frame.grid(row=0, column=0, sticky="w")
            else:
                self._regulator_frame.grid_remove()
        if self._dpad_frame is not None:
            if mode == "dpad":
                self._dpad_frame.grid(row=0, column=0, sticky="w")
            else:
                self._dpad_frame.grid_remove()

    def _set_offset(self, x_value: float, y_value: float) -> None:
        if self._updating_controls:
            return
        self.offset_x.set(clamp(x_value, 0.0, 1.0))
        self.offset_y.set(clamp(y_value, 0.0, 1.0))
        self._on_slider_change(0.0)

    def _adjust_offset(self, delta_x: float, delta_y: float) -> None:
        if self.current_image is None:
            return
        self._set_offset(self.offset_x.get() + delta_x, self.offset_y.get() + delta_y)

    def _center_offset(self) -> None:
        if self.current_image is None:
            return
        self._set_offset(self.CENTER_VALUE, self.CENTER_VALUE)

    def _adjust_zoom(self, delta: float) -> None:
        if self._updating_controls or self.current_image is None:
            return
        new_ratio = clamp(
            self.size_ratio.get() + delta,
            self.MIN_ZOOM_RATIO,
            self.MAX_ZOOM_RATIO,
        )
        self.size_ratio.set(new_ratio)
        self._on_slider_change(0.0)

    def _refresh_crop_buttons(self) -> None:
        if not self._crop_buttons:
            return
        if not self._crop_buttons_enabled:
            for key, button in self._crop_buttons.items():
                base_style = "Start.TButton" if key == "start" else "End.TButton"
                button.state(["disabled"])
                button.configure(style=base_style)
            return
        for key, button in self._crop_buttons.items():
            base_style = "Start.TButton" if key == "start" else "End.TButton"
            active_style = "StartActive.TButton" if key == "start" else "EndActive.TButton"
            if key == "start" and not self.motion_enabled_var.get():
                button.state(["disabled"])
                button.configure(style=base_style)
                continue
            button.state(["!disabled"])
            style = active_style if self.active_crop_var.get() == key else base_style
            button.configure(style=style)

    def _select_crop(self, target: str) -> None:
        if target == "start" and not self.motion_enabled_var.get():
            return
        if self.active_crop_var.get() != target:
            self.active_crop_var.set(target)
        else:
            self._on_active_crop_change()
            self._refresh_legend_state()
        self._refresh_crop_buttons()

    def _tutorial_state_path(self, ensure: bool = False) -> Path:
        base = Path.home() / ".memoryball-studio"
        if ensure:
            ensure_dir(base)
        return base / "ui_state.json"

    def _load_tutorial_completed(self) -> bool:
        path = self._tutorial_state_path()
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(data.get("tutorial_completed"))

    def _save_tutorial_completed(self) -> None:
        path = self._tutorial_state_path(ensure=True)
        try:
            path.write_text(json.dumps({"tutorial_completed": True}, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _maybe_start_tutorial(self) -> None:
        if not self._tutorial_completed:
            self._start_tutorial()

    def _start_tutorial(self) -> None:
        if self._tutorial_running:
            return
        self.update_idletasks()
        steps = self._build_tutorial_steps()
        if not steps:
            return
        self._tutorial_steps = steps
        self._tutorial_index = 0
        self._tutorial_running = True
        self._show_tutorial_step(0)

    def _build_tutorial_steps(self) -> list[dict[str, object]]:
        steps = [
            {
                "widget": self.input_entry,
                "title": "Eingabe auswählen",
                "message": "👉 Wähle hier den Ordner mit deinen Bildern oder Videos aus.",
                "placement": "right",
            },
            {
                "widget": self.listbox,
                "title": "Dateien überblicken",
                "message": "👉 In dieser Liste erscheinen alle Medien aus dem Ordner. Ein Klick lädt die Vorschau.",
                "placement": "right",
            },
            {
                "widget": self.canvas,
                "title": "Ausschnitt anpassen",
                "message": "👉 Ziehe die farbigen Rahmen oder benutze die Regler, um den Bildausschnitt festzulegen.",
                "placement": "left",
            },
            {
                "widget": self.crop_button_frame if self.crop_button_frame is not None else self,
                "title": "Start- und Endpunkte wählen",
                "message": "👉 Mit diesen beiden Schaltflächen wechselst du zwischen Start (Rot) und Ende (Grün) der Bewegung.",
                "placement": "right",
            },
            {
                "widget": self.convert_button,
                "title": "Konvertierung starten",
                "message": "👉 Wenn alles passt, starte hier die Verarbeitung deiner Medien.",
                "placement": "above",
            },
        ]
        return steps

    def _show_tutorial_step(self, index: int) -> None:
        if index < 0 or index >= len(self._tutorial_steps):
            self._finish_tutorial()
            return
        self._destroy_tutorial_window()
        step = self._tutorial_steps[index]
        widget = step["widget"]
        if not isinstance(widget, tk.Misc):
            self._finish_tutorial()
            return
        widget.update_idletasks()
        self.update_idletasks()
        x = widget.winfo_rootx()
        y = widget.winfo_rooty()
        width = max(widget.winfo_width(), 1)
        height = max(widget.winfo_height(), 1)
        placement = step.get("placement", "right")
        offset = 24
        pos_x: float
        pos_y: float
        if placement == "left":
            pos_x = max(self.winfo_rootx() + 20, x - 320)
            pos_y = y
        elif placement == "below":
            pos_x = x
            pos_y = y + height + offset
        elif placement == "above":
            pos_x = x
            pos_y = max(self.winfo_rooty() + 20, y - 200)
        else:  # right
            pos_x = x + width + offset
            pos_y = y
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        pos_x = max(0.0, min(pos_x, screen_w - 320))
        pos_y = max(0.0, min(pos_y, screen_h - 200))
        window = tk.Toplevel(self)
        window.transient(self)
        window.attributes("-topmost", True)
        window.configure(background=self._card_background)
        window.title("Tutorial")
        window.geometry(f"+{int(pos_x)}+{int(pos_y)}")
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", lambda: self._stop_tutorial(record_completion=False))
        frame = ttk.Frame(window, style="Card.TFrame", padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        arrow = {"right": "⬅️", "left": "➡️", "below": "⬆️", "above": "⬇️"}.get(placement, "⬅️")
        ttk.Label(frame, text=f"{arrow} {step['title']}", style="TutorialHeading.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text=step["message"], wraplength=280, justify="left", style="Tutorial.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 12)
        )
        controls = ttk.Frame(frame, style="Card.TFrame")
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        ttk.Button(
            controls,
            text="Später",
            command=lambda: self._stop_tutorial(record_completion=False),
        ).grid(row=0, column=0, sticky="w")
        next_text = "Fertig" if index == len(self._tutorial_steps) - 1 else "Weiter"
        ttk.Button(
            controls,
            text=next_text,
            style="Accent.TButton",
            command=self._advance_tutorial,
        ).grid(row=0, column=1, sticky="e")
        try:
            window.grab_set()
        except tk.TclError:
            pass
        self._tutorial_window = window

    def _advance_tutorial(self) -> None:
        if not self._tutorial_running:
            return
        self._tutorial_index += 1
        if self._tutorial_index >= len(self._tutorial_steps):
            self._finish_tutorial()
        else:
            self._show_tutorial_step(self._tutorial_index)

    def _finish_tutorial(self) -> None:
        self._stop_tutorial(record_completion=True)

    def _stop_tutorial(self, record_completion: bool) -> None:
        self._destroy_tutorial_window()
        self._tutorial_running = False
        self._tutorial_steps = []
        self._tutorial_index = -1
        if record_completion:
            if not self._tutorial_completed:
                self._tutorial_completed = True
                self._save_tutorial_completed()

    def _destroy_tutorial_window(self) -> None:
        if self._tutorial_window is not None:
            try:
                self._tutorial_window.grab_release()
            except tk.TclError:
                pass
            self._tutorial_window.destroy()
            self._tutorial_window = None

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
        for button in self._compact_control_buttons:
            if enabled:
                button.state(["!disabled"])
            else:
                button.state(["disabled"])
        self.prev_button.state(["!disabled"] if enabled else ["disabled"])
        self.next_button.state(["!disabled"] if enabled else ["disabled"])
        self._crop_buttons_enabled = enabled
        if enabled:
            self._refresh_selected_button_state()
        else:
            self.convert_selected_button.state(["disabled"])
        self._refresh_crop_buttons()
        self._refresh_legend_state()
        self.listbox.state(["!disabled"])

    def _refresh_selected_button_state(self) -> None:
        if self._conversion_active:
            self.convert_selected_button.state(["disabled"])
            return
        if self.current_path is not None and self.current_path in self.image_files:
            self.convert_selected_button.state(["!disabled"])
        else:
            self.convert_selected_button.state(["disabled"])

    def _thumbnail_for(self, path: Path) -> ImageTk.PhotoImage:
        if is_image(path):
            thumbnail = self._thumbnail_cache.get(path)
            if thumbnail is None:
                thumbnail = self._create_image_thumbnail(path)
                self._thumbnail_cache[path] = thumbnail
            return thumbnail
        return self._get_video_thumbnail()

    def _create_image_thumbnail(self, path: Path, size: int = 48) -> ImageTk.PhotoImage:
        border_color = "#1b2032"
        background_color = "#0b0f1c"
        max_content = size - 8
        content_size = (max_content, max_content)
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception:
            image = Image.new("RGB", content_size, "#2a3149")
        else:
            if ImageOps is not None:
                image = ImageOps.contain(image, content_size, RESAMPLE_LANCZOS)
            else:  # pragma: no cover - fallback for older Pillow
                image.thumbnail(content_size, RESAMPLE_LANCZOS)
        thumb = Image.new("RGB", (size, size), background_color)
        draw = ImageDraw.Draw(thumb)
        draw.rectangle((0, 0, size - 1, size - 1), outline=border_color)
        offset = ((size - image.width) // 2, (size - image.height) // 2)
        thumb.paste(image, offset)
        return ImageTk.PhotoImage(thumb)

    def _get_video_thumbnail(self, size: int = 48) -> ImageTk.PhotoImage:
        if self._video_thumbnail is None:
            background_color = "#111624"
            accent = self._accent_color
            thumb = Image.new("RGB", (size, size), background_color)
            draw = ImageDraw.Draw(thumb)
            draw.rectangle((0, 0, size - 1, size - 1), outline=accent)
            triangle = [
                (size // 2 - 6, size // 2 - 9),
                (size // 2 - 6, size // 2 + 9),
                (size // 2 + 8, size // 2),
            ]
            draw.polygon(triangle, fill=accent)
            self._video_thumbnail = ImageTk.PhotoImage(thumb)
        return self._video_thumbnail

    def _select_list_index(self, index: int) -> None:
        if index < 0 or index >= len(self._list_iids):
            return
        iid = self._list_iids[index]
        self.listbox.selection_set(iid)
        self.listbox.focus(iid)
        self.listbox.see(iid)

    def _list_selection_indices(self) -> list[int]:
        indices: list[int] = []
        for iid in self.listbox.selection():
            try:
                position = self._list_iids.index(iid)
            except ValueError:
                continue
            indices.append(position)
        return indices

    def _default_output_for(self, path: Path) -> Path:
        if path.is_file():
            return path.parent / f"Converted {path.stem}"
        return path.parent / f"Converted {path.name}"

    def _resolve_output_dir(self) -> Optional[Path]:
        raw = self.output_var.get().strip()
        if not raw and self.input_path:
            return self._default_output_for(self.input_path)
        if not raw:
            return None
        return Path(raw).expanduser()

    def _get_preview_cropper(self) -> Optional[FaceCropper]:
        if self._preview_cropper is None:
            self._preview_cropper = FaceCropper()
        return self._preview_cropper

    def destroy(self) -> None:  # pragma: no cover - GUI shutdown
        self._destroy_tutorial_window()
        if self._preview_cropper is not None:
            self._preview_cropper.close()
            self._preview_cropper = None
        super().destroy()

    # ------------------------------------------------------------------
    # Folder handling
    # ------------------------------------------------------------------
    def _choose_input(self) -> None:
        choice = messagebox.askyesnocancel(
            "Eingabe wählen",
            "Möchtest du einen kompletten Ordner laden?\n"
            "Ja = Ordner, Nein = einzelne Datei.",
            parent=self,
        )
        if choice is None:
            return
        if choice:
            selection = filedialog.askdirectory(title="Eingabeordner wählen")
        else:
            selection = filedialog.askopenfilename(
                title="Eingabedatei wählen",
                filetypes=(
                    ("Unterstützte Bilder", "*.jpg *.jpeg *.png *.webp *.heic *.heif"),
                    ("Unterstützte Videos", "*.mp4 *.mov *.mkv *.avi"),
                    ("Alle Dateien", "*.*"),
                ),
            )
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
        self._auto_generated_paths.clear()
        self._load_media_files()
        self._refresh_output_list()

    def _load_media_files(self) -> None:
        import_message = "Importiere Eingabe…"
        self._show_loading_overlay(import_message)
        self.update_idletasks()
        try:
            self.media_files.clear()
            self.image_files.clear()
            for item in self.listbox.get_children():
                self.listbox.delete(item)
            self._list_paths.clear()
            self._list_iids.clear()
            self._thumbnail_cache.clear()
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
            base_for_display = (
                self.input_path if self.input_path.is_dir() else self.input_path.parent
            )
            for media in self.media_files:
                try:
                    display = media.relative_to(base_for_display)
                except ValueError:
                    display = Path(media.name)
                index = len(self._list_paths)
                iid = f"item-{index}"
                thumbnail = self._thumbnail_for(media)
                self.listbox.insert("", tk.END, iid=iid, text=str(display), image=thumbnail)
                self._list_paths.append(media)
                self._list_iids.append(iid)
                if is_image(media):
                    self.image_files.append(media)

            if self.image_files:
                first_image = self.image_files[0]
                index = self._list_paths.index(first_image)
                self._select_list_index(index)
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
        finally:
            if self._loading_message_var.get() == import_message:
                self._hide_loading_overlay()

    # ------------------------------------------------------------------
    # Preview & manual crop
    # ------------------------------------------------------------------
    def _on_listbox_select(self) -> None:
        if not self._list_paths:
            return
        selection = self._list_selection_indices()
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
            assert self.current_image is not None
            manual = self._default_manual_for_image(self.current_image)
            self.manual_crops[path] = manual
            self._auto_generated_paths.discard(path)
        else:
            manual = self._normalize_manual(manual)
        self.manual_crops[path] = manual
        self._apply_manual_to_controls(manual)
        self._set_controls_enabled(True)
        self._update_navigation_state()
        self._refresh_selected_button_state()
        self.progress_var.set("Bereit.")
        self._hide_loading_overlay()

    def _current_processing_options(self) -> ProcessingOptions:
        base_path: Path
        if self.input_path is not None:
            base_path = self.input_path
        elif self.current_path is not None:
            base_path = self.current_path
        else:
            base_path = Path.cwd()
        base_path = self._normalize_path(base_path)
        output_dir = self._resolve_output_dir()
        if output_dir is None:
            output_dir = self._default_output_for(base_path)
        return ProcessingOptions(
            input_path=base_path,
            output_dir=output_dir,
            size=self.size_var.get(),
            face_detection_enabled=True,
            detection_mode="auto",
            motion_enabled=self.motion_enabled_var.get(),
            motion_direction=self.motion_direction_var.get(),
        )

    def _normalize_manual_for_image(
        self,
        image: Image.Image,
        manual: ManualCrop,
        overflow: Optional[float] = None,
    ) -> ManualCrop:
        width, height = image.size
        start = self._normalize_crop_box(manual.start, width, height, overflow=overflow)
        end = self._normalize_crop_box(manual.end, width, height, overflow=overflow)
        return ManualCrop(start=start, end=end)

    def _default_manual_for_image(self, image: Image.Image) -> ManualCrop:
        width, height = image.size
        size = float(min(width, height))
        base = CropBox((width - size) / 2, (height - size) / 2, size)
        manual = ManualCrop(start=base, end=base)
        return self._normalize_manual_for_image(image, manual, overflow=0.0)

    def _compute_auto_manual_for_image(
        self,
        image: Image.Image,
        options: ProcessingOptions,
        cropper: Optional[FaceCropper],
    ) -> ManualCrop:
        if options.motion_enabled and cropper is not None:
            manual = determine_motion_manual(image, options, cropper)
        elif cropper is not None:
            crop = determine_crop_box(image, options, cropper)
            manual = ManualCrop(start=crop, end=crop)
        else:
            width, height = image.size
            size = float(min(width, height))
            base = CropBox((width - size) / 2, (height - size) / 2, size)
            manual = ManualCrop(start=base, end=base)
        return self._normalize_manual_for_image(image, manual, overflow=0.0)

    def _start_auto_detection(self, path: Path, *, message: str) -> None:
        if self.current_image is None:
            return
        image = self.current_image.copy()
        options = self._current_processing_options()
        cropper = self._get_preview_cropper()
        token = object()
        self._auto_task_token = token
        self._show_loading_overlay(message)
        self.progress_var.set(message)
        self._set_controls_enabled(False)

        def worker() -> None:
            try:
                manual = self._compute_auto_manual_for_image(image, options, cropper)
            except Exception as exc:
                result: ManualCrop | Exception = exc
            else:
                result = manual
            self.after(0, lambda: self._finish_auto_detection(token, path, result))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_auto_detection(
        self, token: object, path: Path, result: ManualCrop | Exception
    ) -> None:
        if token != self._auto_task_token:
            return
        self._auto_task_token = None
        if isinstance(result, Exception):
            self._hide_loading_overlay()
            self.progress_var.set("Analyse fehlgeschlagen.")
            self._set_controls_enabled(True)
            self._refresh_selected_button_state()
            self._update_navigation_state()
            messagebox.showerror(
                "Fehler",
                f"Die automatische Ausrichtung ist fehlgeschlagen.\n\n{result}",
            )
            return
        if path != self.current_path:
            self._hide_loading_overlay()
            self._set_controls_enabled(True)
            self._refresh_selected_button_state()
            self._update_navigation_state()
            return
        manual = self._normalize_manual(result)
        self.manual_crops[path] = manual
        self._auto_generated_paths.add(path)
        self._apply_manual_to_controls(manual, auto_generated=True)
        self._set_controls_enabled(True)
        self._refresh_selected_button_state()
        self._update_navigation_state()
        self.progress_var.set("Bereit.")
        self._hide_loading_overlay()

    def _auto_manual_current(self) -> ManualCrop:
        assert self.current_image is not None
        options = self._current_processing_options()
        cropper = self._get_preview_cropper()
        return self._compute_auto_manual_for_image(self.current_image, options, cropper)

    # ------------------------------------------------------------------
    # Memory-Minispiel für die Wartezeit
    # ------------------------------------------------------------------
    def _memory_grid_dimensions(
        self,
        total_cards: int,
        *,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
    ) -> tuple[int, int]:
        if total_cards <= 0:
            return (1, 1)
        if max_rows is None or max_rows <= 0:
            max_rows = total_cards
        if max_cols is None or max_cols <= 0:
            max_cols = total_cards
        best: Optional[tuple[int, int]] = None
        best_score: Optional[tuple[int, int, int]] = None
        for cols in range(1, max_cols + 1):
            rows = math.ceil(total_cards / cols)
            if rows > max_rows:
                continue
            overflow = rows * cols - total_cards
            diff = abs(rows - cols)
            score = (overflow, diff, cols)
            if best_score is None or score < best_score:
                best_score = score
                best = (rows, cols)
        if best is None:
            cols = min(max_cols, total_cards)
            cols = max(1, cols)
            rows = math.ceil(total_cards / cols)
            rows = min(max_rows, max(1, rows))
            best = (rows, cols)
        rows, cols = best
        if rows > cols:
            rows, cols = cols, rows
        return (max(1, rows), max(1, cols))

    def _create_memory_back_image(self, size: int) -> ImageTk.PhotoImage:
        size = max(32, size)
        base = Image.new("RGB", (size, size), "#0b1326")
        draw = ImageDraw.Draw(base)
        radius = max(8, size // 6)
        border_color = "#1e2f52"
        fill_color = "#101c36"
        draw.rounded_rectangle(
            (1, 1, size - 2, size - 2),
            radius=radius,
            fill=fill_color,
            outline=border_color,
            width=2,
        )
        pad = size // 4
        center = size // 2
        draw.line((pad, center, size - pad, center), fill=self._accent_color, width=3)
        draw.line((center, pad, center, size - pad), fill=self._success_color, width=3)
        return ImageTk.PhotoImage(base)

    def _create_memory_face_image(self, path: Path, size: int) -> ImageTk.PhotoImage:
        canvas = Image.new("RGB", (size, size), "#0b1326")
        border_color = "#2b3f66"
        max_content = max(1, size - 12)
        try:
            with Image.open(path) as img:
                picture = img.convert("RGB")
        except Exception:
            picture = Image.new("RGB", (max_content, max_content), "#24335a")
        picture.thumbnail((max_content, max_content), RESAMPLE_LANCZOS)
        offset = ((size - picture.width) // 2, (size - picture.height) // 2)
        canvas.paste(picture, offset)
        draw = ImageDraw.Draw(canvas)
        radius = max(6, size // 8)
        draw.rounded_rectangle(
            (1, 1, size - 2, size - 2),
            radius=radius,
            outline=border_color,
            width=2,
        )
        return ImageTk.PhotoImage(canvas)

    def _start_memory_game(self, source_images: list[Path], token: object) -> None:
        if not source_images:
            return
        total_images = len(source_images)
        game_images = list(source_images)
        self._close_memory_game()
        self._memory_flip_job = None
        self._hide_loading_overlay()
        self._set_settings_collapsed(True)
        if self.canvas.winfo_manager():
            self.canvas.grid_remove()
        self.update_idletasks()
        preview_parent = self.canvas.master
        preview_parent.update_idletasks()
        available_width = preview_parent.winfo_width()
        available_height = preview_parent.winfo_height()
        if available_width <= 1:
            available_width = max(self.CANVAS_SIZE, self.winfo_width() - 200)
        if available_height <= 1:
            available_height = max(self.CANVAS_SIZE, self.winfo_height() - 200)
        content_width = max(200, available_width - 40)
        content_height = max(200, available_height - 40)
        usable_width = max(160, content_width - 32)
        info_space = 200
        usable_height = max(self.MIN_MEMORY_CARD_SIZE + self.MEMORY_CARD_GAP, content_height - info_space)
        card_unit = self.MIN_MEMORY_CARD_SIZE + self.MEMORY_CARD_GAP
        max_cols = max(1, usable_width // card_unit)
        max_rows = max(1, usable_height // card_unit)
        max_cards = max_cols * max_rows
        if max_cards < 2:
            max_cards = 2
        if max_cards % 2 != 0:
            max_cards -= 1
        if max_cards < 2:
            max_cards = 2
        if len(game_images) * 2 > max_cards:
            max_pairs = max_cards // 2
            max_pairs = max(1, max_pairs)
            game_images = game_images[:max_pairs]
        total_cards = len(game_images) * 2
        rows, cols = self._memory_grid_dimensions(
            total_cards, max_rows=max_rows, max_cols=max_cols
        )
        horizontal_size = max(
            32, (usable_width // max(1, cols)) - self.MEMORY_CARD_GAP
        )
        vertical_size = max(
            32, (usable_height // max(1, rows)) - self.MEMORY_CARD_GAP
        )
        cell_size = max(
            self.MIN_MEMORY_CARD_SIZE,
            min(180, horizontal_size, vertical_size),
        )
        container = ttk.Frame(preview_parent, style="Card.TFrame", padding=16)
        container.grid(row=1, column=1, sticky="nsew", pady=(12, 12))
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)
        heading = ttk.Label(container, text="Memory", style="Heading.TLabel")
        heading.grid(row=0, column=0, sticky="w")
        sublabel = ttk.Label(
            container,
            text="Finde die Paare, während die Analyse läuft.",
            style="Body.TLabel",
            wraplength=usable_width,
        )
        sublabel.grid(row=1, column=0, sticky="w", pady=(4, 12))
        if len(game_images) < total_images:
            sublabel.configure(
                text=(
                    "Finde die Paare, während die Analyse läuft. "
                    f"{len(game_images)} von {total_images} Bildern passen in das Spielfeld."
                )
            )

        game_frame = ttk.Frame(container)
        game_frame.grid(row=2, column=0, sticky="n")

        back_image = self._create_memory_back_image(cell_size)
        cards: list[MemoryCard] = []
        card_paths = [path for path in game_images for _ in range(2)]
        random.shuffle(card_paths)
        for index, path in enumerate(card_paths):
            face_image = self._create_memory_face_image(path, cell_size)
            button = tk.Button(
                game_frame,
                image=back_image,
                command=lambda idx=index: self._on_memory_card_click(idx),
                bd=0,
                relief="flat",
                highlightthickness=0,
                background=self._card_background,
                activebackground=self._card_background,
                cursor="hand2",
                takefocus=0,
                padx=0,
                pady=0,
            )
            row = index // max(1, cols)
            column = index % max(1, cols)
            button.grid(row=row, column=column, padx=4, pady=4)
            button.image = back_image
            cards.append(MemoryCard(path=path, face_image=face_image, button=button))

        progress_var = tk.StringVar(value=f"Analysefortschritt: 0/{total_images}")
        progress_label = ttk.Label(container, textvariable=progress_var, style="Body.TLabel")
        progress_label.grid(row=3, column=0, sticky="w", pady=(16, 4))
        progress = ttk.Progressbar(
            container,
            mode="determinate",
            maximum=max(1, total_images),
            value=0,
        )
        progress.grid(row=4, column=0, sticky="ew")

        self._memory_container = container
        self._memory_game_state = MemoryGameState(
            frame=container,
            cards=cards,
            token=token,
            progressbar=progress,
            progress_var=progress_var,
            back_image=back_image,
            total=total_images,
        )

    def _update_memory_progress(self, token: object, processed: int, total: int) -> None:
        state = self._memory_game_state
        if state is None or state.token is not token:
            return
        state.total = total
        maximum = max(1, total)
        state.progressbar.configure(maximum=maximum)
        clamped = int(clamp(processed, 0, maximum))
        state.progressbar.configure(value=clamped)
        state.progress_var.set(f"Analysefortschritt: {clamped}/{maximum}")

    def _on_memory_card_click(self, index: int) -> None:
        state = self._memory_game_state
        if state is None or state.lock:
            return
        if index < 0 or index >= len(state.cards):
            return
        card = state.cards[index]
        if card.matched or card.revealed:
            return
        self._reveal_memory_card(state, index)
        if state.first_index is None:
            state.first_index = index
            return
        first_index = state.first_index
        first_card = state.cards[first_index]
        if first_card.path == card.path:
            card.matched = True
            first_card.matched = True
            card.button.configure(state="disabled", cursor="arrow")
            first_card.button.configure(state="disabled", cursor="arrow")
            accent = self._success_color
            for matched in (card, first_card):
                matched.button.configure(
                    highlightthickness=2,
                    highlightbackground=accent,
                    highlightcolor=accent,
                )
            state.first_index = None
            self._check_memory_game_completion()
            return

        state.lock = True
        if self._memory_flip_job is not None:
            try:
                self.after_cancel(self._memory_flip_job)
            except ValueError:
                pass
            self._memory_flip_job = None

        def hide_cards() -> None:
            self._hide_memory_card(state, first_index)
            self._hide_memory_card(state, index)
            state.first_index = None
            state.lock = False
            self._memory_flip_job = None

        self._memory_flip_job = self.after(800, hide_cards)

    def _reveal_memory_card(self, state: MemoryGameState, index: int) -> None:
        card = state.cards[index]
        card.button.configure(image=card.face_image)
        card.button.image = card.face_image
        accent = self._accent_color
        card.button.configure(
            highlightthickness=2,
            highlightbackground=accent,
            highlightcolor=accent,
        )
        card.revealed = True

    def _hide_memory_card(self, state: MemoryGameState, index: int) -> None:
        card = state.cards[index]
        if card.matched:
            return
        card.button.configure(image=state.back_image, state="normal", cursor="hand2")
        card.button.image = state.back_image
        card.button.configure(highlightthickness=0)
        card.revealed = False

    def _check_memory_game_completion(self) -> None:
        state = self._memory_game_state
        if state is None or state.completed:
            return
        if all(card.matched for card in state.cards):
            state.completed = True
            if not state.analysis_done:
                state.progress_var.set("Alle Paare gefunden! Die Analyse läuft noch…")
            if state.analysis_done:
                self.after(0, self._close_memory_game)

    def _on_memory_analysis_complete(self, token: object) -> None:
        state = self._memory_game_state
        if state is None or state.token is not token:
            return
        state.analysis_done = True
        maximum = max(1, state.total)
        state.progressbar.configure(maximum=maximum, value=maximum)
        state.progress_var.set(f"Analysefortschritt: {maximum}/{maximum}")
        if state.completed:
            self.after(0, self._close_memory_game)
            return

        def prompt_user() -> None:
            answer = messagebox.askyesno(
                "Analyse abgeschlossen",
                "Möchtest du zu Ende spielen? Deine Bilder sind fertig.",
            )
            if answer:
                return
            self._close_memory_game()

        self.after(0, prompt_user)

    def _close_memory_game(self) -> None:
        if self._memory_flip_job is not None:
            try:
                self.after_cancel(self._memory_flip_job)
            except ValueError:
                pass
            self._memory_flip_job = None
        self._memory_game_state = None
        if self._memory_container is not None:
            try:
                self._memory_container.destroy()
            except tk.TclError:
                pass
            self._memory_container = None
        self._set_settings_collapsed(False)
        if self.canvas.winfo_manager() == "":
            self.canvas.grid(row=1, column=1, sticky="n", pady=(12, 12))

    def _analyze_all_images(self) -> None:
        if not self.image_files:
            messagebox.showinfo("Analyse", "Keine Bilder zum Analysieren gefunden.")
            return
        if self.input_path is None:
            messagebox.showinfo("Analyse", "Bitte zuerst eine Eingabequelle wählen.")
            return
        if self._bulk_auto_token is not None:
            messagebox.showinfo("Analyse", "Eine Analyse läuft bereits.")
            return
        options = self._current_processing_options()
        cropper = self._get_preview_cropper()
        if cropper is None:
            messagebox.showerror(
                "Analyse",
                "Gesichtserkennung ist nicht verfügbar. Bitte installiere die erforderlichen Abhängigkeiten.",
            )
            return

        images = list(self.image_files)
        total = len(images)
        token = object()
        self._bulk_auto_token = token
        self._start_memory_game(images, token)
        self._set_controls_enabled(False)
        self._update_bulk_auto_progress(token, 0, total)

        def worker() -> None:
            results: dict[Path, ManualCrop] = {}
            errors: dict[Path, Exception] = {}
            for index, path in enumerate(images, start=1):
                try:
                    with Image.open(path) as img:
                        image = img.copy()
                except Exception as exc:
                    errors[path] = exc
                else:
                    try:
                        manual = self._compute_auto_manual_for_image(image, options, cropper)
                    except Exception as exc:  # pragma: no cover - GUI feedback
                        errors[path] = exc
                    else:
                        results[path] = manual
                finally:
                    self.after(
                        0,
                        lambda idx=index, total=total: self._update_bulk_auto_progress(
                            token, idx, total
                        ),
                    )
            self.after(
                0,
                lambda: self._finish_bulk_auto(token, results, errors, total),
            )

        thread = threading.Thread(target=worker, daemon=True)
        self._bulk_auto_thread = thread
        thread.start()

    def _update_bulk_auto_progress(self, token: object, processed: int, total: int) -> None:
        if token != self._bulk_auto_token:
            return
        message = f"Analysiere Bilder… {processed}/{total}"
        self.progress_var.set(message)
        self._loading_message_var.set(message)
        self._update_memory_progress(token, processed, total)

    def _finish_bulk_auto(
        self,
        token: object,
        results: dict[Path, ManualCrop],
        errors: dict[Path, Exception],
        total: int,
    ) -> None:
        if token != self._bulk_auto_token:
            return
        self._bulk_auto_token = None
        self._bulk_auto_thread = None
        self._hide_loading_overlay()
        self._set_controls_enabled(True)

        applied = 0
        retained = 0
        for path, manual in results.items():
            existing = self.manual_crops.get(path)
            if existing is not None and path not in self._auto_generated_paths:
                retained += 1
                continue
            self.manual_crops[path] = manual
            self._auto_generated_paths.add(path)
            if path == self.current_path and self.current_image is not None:
                self._apply_manual_to_controls(manual, auto_generated=True)
            applied += 1

        if errors:
            failed = len(errors)
            sample_path, sample_error = next(iter(errors.items()))
            message = (
                f"{applied} von {total} Bildern analysiert – {failed} fehlgeschlagen."
            )
            self.progress_var.set(message)
            messagebox.showwarning(
                "Analyse",
                "Nicht alle Bilder konnten automatisch analysiert werden. "
                f"Beispiel: {sample_path.name}\n\n{sample_error}",
            )
        else:
            message = f"{applied} von {total} Bildern analysiert."
            if retained:
                message += " Manuell angepasste Bilder wurden beibehalten."
            self.progress_var.set(message)

        self._refresh_selected_button_state()
        self._refresh_crop_buttons()
        self._refresh_legend_state()
        self._on_memory_analysis_complete(token)

    def _scale_crop(self, crop: CropBox, factor: float, width: int, height: int) -> CropBox:
        factor = clamp(factor, 0.01, 10.0)
        size = clamp(crop.size * factor, 1.0, max_crop_size(width, height))
        center_x = crop.x + crop.size / 2
        center_y = crop.y + crop.size / 2
        x = center_x - size / 2
        y = center_y - size / 2
        min_x, max_x = crop_position_bounds(
            size,
            width,
            overflow_ratio=self.PREVIEW_OVERFLOW_RATIO,
            axis="x",
        )
        min_y, max_y = crop_position_bounds(
            size,
            height,
            overflow_ratio=self.PREVIEW_OVERFLOW_RATIO,
            axis="y",
        )
        x = clamp(x, min_x, max_x)
        y = clamp(y, min_y, max_y)
        return self._normalize_crop_box(CropBox(x=x, y=y, size=size), width, height, overflow=0.0)

    def _normalize_crop_box(
        self,
        crop: CropBox,
        width: int,
        height: int,
        *,
        overflow: Optional[float] = None,
    ) -> CropBox:
        ratio = self.PREVIEW_OVERFLOW_RATIO if overflow is None else overflow
        return normalize_crop_with_overflow(width, height, crop, overflow_ratio=ratio)

    def _normalize_manual(self, manual: ManualCrop, overflow: Optional[float] = None) -> ManualCrop:
        assert self.current_image is not None
        return self._normalize_manual_for_image(self.current_image, manual, overflow=overflow)

    def _active_manual_crop(self, manual: ManualCrop) -> CropBox:
        if self.motion_enabled_var.get() and self.active_crop_var.get() == "start":
            return manual.start
        return manual.end

    def _sync_sliders_with_active(self, manual: ManualCrop) -> None:
        if self.current_image is None:
            return
        crop = self._active_manual_crop(manual)
        width, height = self.current_image.size
        max_side = max(1, max(width, height))
        size_ratio = clamp(crop.size / max_side, 0.01, 1.0)
        min_x, max_x = crop_position_bounds(
            crop.size,
            width,
            overflow_ratio=self.PREVIEW_OVERFLOW_RATIO,
            axis="x",
        )
        min_y, max_y = crop_position_bounds(
            crop.size,
            height,
            overflow_ratio=self.PREVIEW_OVERFLOW_RATIO,
            axis="y",
        )
        range_x = max_x - min_x
        range_y = max_y - min_y
        offset_x = clamp((crop.x - min_x) / range_x if range_x else 0.0, 0.0, 1.0)
        offset_y = clamp((crop.y - min_y) / range_y if range_y else 0.0, 0.0, 1.0)
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

    def _update_current_manual(
        self,
        manual: ManualCrop,
        *,
        sync_controls: bool = True,
        auto_generated: Optional[bool] = None,
    ) -> None:
        if self.current_image is None or self.current_path is None:
            return
        normalized = self._normalize_manual(manual)
        self.manual_crops[self.current_path] = normalized
        if auto_generated is True:
            self._auto_generated_paths.add(self.current_path)
        elif auto_generated is False:
            self._auto_generated_paths.discard(self.current_path)
        if sync_controls:
            self._sync_sliders_with_active(normalized)
        self._render_preview(normalized)
        self._update_position_label()

    def _apply_manual_to_controls(
        self, manual: ManualCrop, *, auto_generated: Optional[bool] = None
    ) -> None:
        self._update_current_manual(manual, sync_controls=True, auto_generated=auto_generated)

    def _update_motion_direction_state(self) -> None:
        if self.motion_enabled_var.get():
            state = "readonly"
        else:
            state = "disabled"
        self.motion_direction_combo.configure(state=state)

    def _on_motion_toggle(self) -> None:
        enabled = self.motion_enabled_var.get()
        self._update_motion_direction_state()
        if not enabled and self.active_crop_var.get() == "start":
            self.active_crop_var.set("end")
        self._refresh_crop_buttons()
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
                self._start_auto_detection(self.current_path, message="Analysiere Bewegung…")
                return
        self._apply_manual_to_controls(manual)
        self._refresh_selected_button_state()
        self._refresh_crop_buttons()
        self._refresh_legend_state()

    def _on_active_crop_change(self, *_args: object) -> None:
        if not self.motion_enabled_var.get():
            if self.active_crop_var.get() != "end":
                self.active_crop_var.set("end")
            self._refresh_crop_buttons()
            return
        if self.current_path is None or self.current_path not in self.manual_crops:
            self._refresh_crop_buttons()
            self._refresh_legend_state()
            return
        manual = self.manual_crops[self.current_path]
        self._sync_sliders_with_active(manual)
        self._render_preview(manual)
        self._refresh_crop_buttons()
        self._refresh_legend_state()

    def _on_motion_direction_change(self, *_args: object) -> None:
        label = self.motion_direction_label_var.get()
        value = self._motion_direction_value_by_label.get(label, "in")
        if value != self.motion_direction_var.get():
            self.motion_direction_var.set(value)
        if (
            self.motion_enabled_var.get()
            and self.current_path is not None
            and self.current_path in self.manual_crops
            and self.current_image is not None
        ):
            manual = self.manual_crops[self.current_path]
            if (
                abs(manual.start.x - manual.end.x) < 1e-3
                and abs(manual.start.y - manual.end.y) < 1e-3
                and abs(manual.start.size - manual.end.size) < 1e-3
            ):
                self._start_auto_detection(self.current_path, message="Analysiere Bewegung…")
            else:
                self._refresh_crop_buttons()
                self._refresh_legend_state()

    def _on_slider_change(self, _value: float | str) -> None:
        if self._updating_controls or self.current_image is None or self.current_path is None:
            return
        width, height = self.current_image.size
        max_side = max(1, max(width, height))
        ratio = clamp(self.size_ratio.get(), 0.01, 1.0)
        size = ratio * max_side
        min_x, max_x = crop_position_bounds(
            size,
            width,
            overflow_ratio=self.PREVIEW_OVERFLOW_RATIO,
            axis="x",
        )
        min_y, max_y = crop_position_bounds(
            size,
            height,
            overflow_ratio=self.PREVIEW_OVERFLOW_RATIO,
            axis="y",
        )
        norm_x = clamp(self.offset_x.get(), 0.0, 1.0)
        norm_y = clamp(self.offset_y.get(), 0.0, 1.0)
        range_x = max_x - min_x
        range_y = max_y - min_y
        x = min_x + norm_x * range_x if range_x else min_x
        y = min_y + norm_y * range_y if range_y else min_y
        new_crop = self._normalize_crop_box(CropBox(x=x, y=y, size=size), width, height, overflow=0.0)
        manual = self.manual_crops.get(self.current_path)
        if manual is None:
            return
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
        self._update_current_manual(
            ManualCrop(start=start, end=end),
            sync_controls=True,
            auto_generated=False,
        )

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

    def _draw_orientation_circle(
        self, rect: tuple[float, float, float, float], color: str, line_width: int
    ) -> None:
        x1, y1, x2, y2 = rect
        diameter = min(x2 - x1, y2 - y1)
        if diameter <= 0:
            return
        margin = diameter * self.CIRCLE_MARGIN
        top = y1
        bottom = y2 - 2 * margin
        if bottom <= top:
            top = y1 + margin
            bottom = y2 - margin
        self.canvas.create_oval(
            x1 + margin,
            top,
            x2 - margin,
            bottom,
            outline=color,
            width=line_width,
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
        self.canvas.config(cursor="")
        offset_x = (self.CANVAS_SIZE - display_width) / 2
        offset_y = (self.CANVAS_SIZE - display_height) / 2
        self.canvas.create_image(self.CANVAS_SIZE / 2, self.CANVAS_SIZE / 2, image=self._tk_image)

        self._canvas_scale = scale
        self._canvas_offset = (offset_x, offset_y)
        self._manual_display = {}

        active_target = "start" if self.motion_enabled_var.get() and self.active_crop_var.get() == "start" else "end"

        def draw_label(rect: tuple[float, float, float, float], target: str, text: str) -> None:
            cx = (rect[0] + rect[2]) / 2
            cy = (rect[1] + rect[3]) / 2
            tag = f"label_{target}"
            self.canvas.create_text(
                cx,
                cy,
                text=text,
                fill=self._legend_colors.get(target, "#ffffff"),
                font=("Segoe UI", 16, "bold"),
                tags=("crop_label", tag),
            )
            self.canvas.tag_bind(tag, "<Button-1>", lambda _e, t=target: self._select_crop(t))

        start_color = self._legend_colors.get("start", self._start_color)
        end_color = self._legend_colors.get("end", self._end_color)

        if self.motion_enabled_var.get():
            start_rect = self._canvas_rect(manual.start)
            end_rect = self._canvas_rect(manual.end)
            self._manual_display["start"] = start_rect
            self._manual_display["end"] = end_rect
            start_active = active_target == "start"
            end_active = active_target == "end"
            self.canvas.create_rectangle(
                *start_rect,
                outline=start_color,
                width=3 if start_active else 2,
            )
            self._draw_orientation_circle(start_rect, start_color, 3 if start_active else 2)
            self.canvas.create_rectangle(
                *end_rect,
                outline=end_color,
                width=3 if end_active else 2,
            )
            self._draw_orientation_circle(end_rect, end_color, 3 if end_active else 2)
            draw_label(start_rect, "start", "1")
            draw_label(end_rect, "end", "2")
            if start_active:
                self._draw_handles(start_rect, start_color)
            else:
                self._draw_handles(end_rect, end_color)
        else:
            end_rect = self._canvas_rect(manual.end)
            self._manual_display["end"] = end_rect
            self.canvas.create_rectangle(*end_rect, outline=end_color, width=3)
            self._draw_orientation_circle(end_rect, end_color, 3)
            self._draw_handles(end_rect, end_color)
            draw_label(end_rect, "end", "2")

        self.canvas.tag_bind("crop_label", "<Enter>", lambda _e: self.canvas.config(cursor="hand2"))
        self.canvas.tag_bind("crop_label", "<Leave>", lambda _e: self.canvas.config(cursor=""))

        self._update_crop_info(manual)
        has_prev, has_next = self._navigation_flags()
        self._update_canvas_navigation(has_prev, has_next)
        self._refresh_crop_buttons()
        self._refresh_legend_state()

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
        self._update_current_manual(
            ManualCrop(start=start, end=end),
            sync_controls=True,
            auto_generated=False,
        )

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
        self._select_list_index(list_index)
        self._load_preview(next_path)

    def _show_next_image(self) -> None:
        if self.current_path is None or self.current_path not in self.image_files:
            return
        index = self.image_files.index(self.current_path)
        if index >= len(self.image_files) - 1:
            return
        next_path = self.image_files[index + 1]
        list_index = self._list_paths.index(next_path)
        self._select_list_index(list_index)
        self._load_preview(next_path)

    def _reset_crop_to_auto(self) -> None:
        if self.current_image is None or self.current_path is None:
            return
        self._start_auto_detection(self.current_path, message="Automatische Ausrichtung…")

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
            messagebox.showerror("Fehler", "Bitte zuerst eine Eingabequelle wählen.")
            return
        output_dir = self._resolve_output_dir()
        if output_dir is None:
            messagebox.showerror("Fehler", "Bitte einen Ausgabeordner wählen oder eingeben.")
            return
        self._start_conversion(output_dir, self._manual_overrides_copy())

    def _on_convert_selected(self) -> None:
        if self.input_path is None:
            messagebox.showerror("Fehler", "Bitte zuerst eine Eingabequelle wählen.")
            return
        output_dir = self._resolve_output_dir()
        if output_dir is None:
            messagebox.showerror("Fehler", "Bitte einen Ausgabeordner wählen oder eingeben.")
            return
        selection = self._list_selection_indices()
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
        options = ProcessingOptions(
            input_path=self.input_path,
            output_dir=output_dir,
            size=self.size_var.get(),
            face_detection_enabled=True,
            detection_mode="auto",
            motion_enabled=self.motion_enabled_var.get(),
            motion_direction=self.motion_direction_var.get(),
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
