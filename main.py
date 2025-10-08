"""CLI entry point for memoryball-autocrop."""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

from src.face_cropper import FaceCropper
from src.image_pipeline import process_image
from src.video_pipeline import process_video
from src.utils import (
    ManualCrop,
    ProcessingOptions,
    iter_media_files,
    is_image,
    is_video,
    setup_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="memoryball-autocrop – Batch-Cropping für Fotos & Videos")
    parser.add_argument("--input", required=False, help="Eingabedatei oder -ordner")
    parser.add_argument("--output", required=False, help="Ausgabeordner")
    parser.add_argument("--mode", choices=["auto", "center", "manual"], default="auto")
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--fps", default="keep", help="Zielframerate oder 'keep'")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--min-face", type=float, default=0.1)
    parser.add_argument("--face-priority", choices=["largest", "center", "all"], default="largest")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pad", type=float, default=0.0)
    parser.add_argument("--image-format", choices=["jpg", "png", "webp"], default="jpg")
    parser.add_argument("--video-ext", choices=["mp4"], default="mp4")
    parser.add_argument("--keep-audio", choices=["on", "off"], default="on")
    parser.add_argument("--log-level", choices=["debug", "info"], default="info")
    parser.add_argument("--crop-x", type=int)
    parser.add_argument("--crop-y", type=int)
    parser.add_argument("--crop-w", type=int)
    parser.add_argument("--crop-h", type=int)
    parser.add_argument("--format", help="Optionales Format-Preset, z.B. jpg,mp4")
    parser.add_argument("--gui", action="store_true", help="Tkinter-GUI starten")
    parser.add_argument(
        "--motion-direction",
        choices=["in", "out"],
        default="in",
        help="Standardrichtung für automatische Bewegungen (rein- oder rauszoomen)",
    )
    return parser.parse_args()


def build_options(args: argparse.Namespace) -> ProcessingOptions:
    if args.format:
        parts = [p.strip().lower() for p in args.format.split(",") if p.strip()]
        for part in parts:
            if part in {"jpg", "png", "webp"}:
                args.image_format = part
            if part in {"mp4"}:
                args.video_ext = part
    input_path = Path(args.input).expanduser() if args.input else None
    output_dir = Path(args.output).expanduser() if args.output else None
    if input_path is None or output_dir is None:
        raise ValueError("--input und --output sind erforderlich, sofern nicht --gui verwendet wird")
    return ProcessingOptions(
        input_path=input_path,
        output_dir=output_dir,
        size=args.size,
        fps=args.fps,
        quality=args.quality,
        crf=args.crf,
        preset=args.preset,
        mode=args.mode,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        crop_w=args.crop_w,
        crop_h=args.crop_h,
        min_face=args.min_face,
        face_priority=args.face_priority,
        threads=args.threads,
        pad=args.pad,
        image_format=args.image_format,
        video_ext=args.video_ext,
        keep_audio=args.keep_audio == "on",
        log_level=args.log_level,
        face_detection_enabled=True,
        detection_mode="auto",
        motion_direction=args.motion_direction,
    )


def _process_images(
    images: List[Path],
    options: ProcessingOptions,
    logger: logging.Logger,
    manual_overrides: dict[Path, ManualCrop] | None = None,
) -> None:
    if not images:
        return
    thread_local = threading.local()
    detectors: List[FaceCropper] = []
    detectors_lock = threading.Lock()

    def _get_detector() -> FaceCropper | None:
        if not options.face_detection_enabled:
            return None
        detector = getattr(thread_local, "detector", None)
        if detector is None:
            detector = FaceCropper(
                min_face=options.min_face,
                face_priority=options.face_priority,
            )
            setattr(thread_local, "detector", detector)
            with detectors_lock:
                detectors.append(detector)
        return detector

    def _worker(path: Path) -> Path:
        detector = _get_detector()
        override = manual_overrides.get(path) if manual_overrides else None
        result = process_image(path, options, detector, manual_crop=override)
        return result.target

    with ThreadPoolExecutor(max_workers=options.threads or 1) as executor:
        futures = {executor.submit(_worker, image): image for image in images}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.exception("Fehler bei Bild %s: %s", futures[future], exc)
    for detector in detectors:
        detector.close()


def _process_videos(
    videos: Iterable[Path],
    options: ProcessingOptions,
    logger: logging.Logger,
) -> None:
    if not videos:
        return
    face_cropper = (
        FaceCropper(
            min_face=options.min_face,
            face_priority=options.face_priority,
        )
        if options.face_detection_enabled
        else None
    )
    for video in videos:
        try:
            process_video(video, options, face_cropper)
        except Exception as exc:
            logger.exception("Fehler bei Video %s: %s", video, exc)
    if face_cropper:
        face_cropper.close()


def run_cli(args: argparse.Namespace) -> None:
    if args.gui:
        from src.gui import launch_gui  # Lazy import, damit Tkinter optional bleibt
        launch_gui()
        return
    try:
        options = build_options(args)
    except ValueError as exc:
        raise SystemExit(str(exc))
    logger = setup_environment(options)
    media_files = list(iter_media_files(options.input_path))
    if not media_files:
        logger.warning("Keine unterstützten Dateien gefunden")
        return
    images = [f for f in media_files if is_image(f)]
    videos = [f for f in media_files if is_video(f)]
    logger.info("Gefundene Dateien: %d Bilder, %d Videos", len(images), len(videos))
    _process_images(images, options, logger)
    _process_videos(videos, options, logger)
    logger.info("Fertig.")


ERROR_LOG_PATH = Path(__file__).with_name("startup-errors.log")


def _pause_on_windows() -> None:
    """Prevent the console window from closing immediately on Windows."""

    if os.name != "nt" or "PYTEST_CURRENT_TEST" in os.environ:
        return
    try:
        input("\nDrücke Enter, um das Fenster zu schließen …")
    except EOFError:
        # Nothing we can do – on pythonw.exe there is no console to pause.
        pass


def _show_native_windows_message(title: str, message: str) -> bool:
    """Use the Win32 MessageBox API as a last resort."""

    if os.name != "nt":
        return False
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, title, 0x00000010)
    except Exception:
        return False
    return True


def _report_startup_error(exc: BaseException) -> None:
    """Write the traceback to a log file and present the error to the user."""

    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trace = "".join(traceback.format_exception(exc))
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}]\n{trace}\n")
    message = (
        "MemoryBall Studio konnte nicht gestartet werden.\n"
        f"Fehler: {exc}\n\n"
        f"Details wurden nach {ERROR_LOG_PATH.resolve()} geschrieben."
    )
    print(message, file=sys.stderr)
    print(trace, file=sys.stderr)

    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("MemoryBall Studio", message)
        root.destroy()
    except Exception:
        # Tkinter may not be available (e.g. pythonw without GUI libraries).
        _show_native_windows_message("MemoryBall Studio", message)

    _pause_on_windows()


def _launch_gui_with_feedback() -> None:
    """Start the Tkinter GUI and convert missing dependency errors into hints."""

    def _dependency_hint(exc: ModuleNotFoundError) -> RuntimeError:
        missing = getattr(exc, "name", None) or str(exc) or "ein unbekanntes Paket"
        if missing.lower() in {"tkinter", "_tkinter", "tk"}:
            hint = (
                "Tkinter ist nicht installiert. Installiere Python erneut über python.org "
                "und aktiviere die Option 'tcl/tk and IDLE', oder verwende 'pip install tk' "
                "in einer Python-Installation, die Tkinter bereitstellt."
            )
        elif missing.lower() in {"pil", "pillow"}:
            hint = "Installiere Pillow mit 'python -m pip install Pillow'."
        else:
            hint = "Installiere alle Abhängigkeiten mit 'python -m pip install -r requirements.txt'."
        return RuntimeError(
            "MemoryBall Studio konnte die grafische Oberfläche nicht starten.\n"
            f"Fehlendes Paket: {missing}\n{hint}"
        )

    try:
        from src.gui import launch_gui
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on user setup
        raise _dependency_hint(exc) from exc

    try:
        launch_gui()
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on user setup
        raise _dependency_hint(exc) from exc
    except RuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - GUI specific failures
        try:
            import tkinter as tk

            if isinstance(exc, tk.TclError):
                raise RuntimeError(
                    "Tkinter konnte nicht initialisiert werden. Bitte stelle sicher, dass eine Desktop-Umgebung "
                    "verfügbar ist und Tk/Tcl installiert wurde (unter Windows im Python-Installer die Option "
                    "'tcl/tk and IDLE' aktivieren).\n"
                    f"Originalfehler: {exc}"
                ) from exc
        except Exception:
            pass
        raise


def main() -> None:
    """Entry point that routes between GUI and CLI modes."""

    if len(sys.argv) == 1:
        _launch_gui_with_feedback()
        return
    run_cli(parse_args())


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # pragma: no cover - defensive for user experience
        _report_startup_error(exc)
        raise SystemExit(1)
