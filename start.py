#!/usr/bin/env python3
"""Bootstrapper for MemoryBall Studio.

This script is intended to be the single entry point a user double-clicks.
It tries to launch :mod:`main` directly and, if that fails, automatically
creates a virtual environment, installs the project's requirements and retries
once more.  On subsequent starts the already-prepared environment is reused so
no additional installation work happens as long as the application starts
successfully.
"""
from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List
from urllib.request import urlopen

BASE_DIR = Path(__file__).resolve().parent
MAIN_FILE = BASE_DIR / "main.py"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
VENV_DIR = BASE_DIR / "venv"
VENV_BIN = VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
VENV_PYTHON = VENV_BIN / ("python.exe" if os.name == "nt" else "python")
FFMPEG_BINARIES = ["ffmpeg.exe", "ffprobe.exe"] if os.name == "nt" else ["ffmpeg", "ffprobe"]
FFMPEG_FALLBACK_DIR = BASE_DIR / "ffmpeg-bin"

_PATH_PREFIXES: list[str] = []


def _build_env(extra_paths: Iterable[str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    paths: list[str] = []
    if extra_paths:
        paths.extend(extra_paths)
    paths.extend(_PATH_PREFIXES)
    if paths:
        original = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([*(p for p in paths if p), original]) if original else os.pathsep.join(
            p for p in paths if p
        )
    return env


@contextmanager
def _activity_indicator(message: str) -> Iterator[None]:
    stop_event = threading.Event()

    def _worker() -> None:
        spinner = itertools.cycle("|/-\\")
        while not stop_event.wait(0.1):
            sys.stderr.write(f"\r{message} {next(spinner)}")
            sys.stderr.flush()
        sys.stderr.write("\r" + " " * (len(message) + 2) + "\r")
        sys.stderr.flush()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()


def _run_process(cmd: List[str], indicator: str | None = None) -> int:
    """Run ``cmd`` inheriting stdio and return the exit code."""

    env = _build_env()
    if indicator:
        with _activity_indicator(indicator):
            completed = subprocess.run(cmd, cwd=BASE_DIR, env=env)
    else:
        completed = subprocess.run(cmd, cwd=BASE_DIR, env=env)
    return int(completed.returncode or 0)


def _launch_application(python_executable: Path) -> int:
    """Launch the main application with the provided Python interpreter."""

    cmd = [str(python_executable), str(MAIN_FILE), *sys.argv[1:]]
    return _run_process(cmd)


def _ensure_venv() -> Path:
    """Make sure a virtual environment exists and return its Python path."""

    if not VENV_PYTHON.exists():
        print("[Installer] Erstelle virtuelle Umgebung …")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True, cwd=BASE_DIR)
    return VENV_PYTHON


def _install_requirements(python_executable: Path) -> None:
    """Install or update the required dependencies using ``python_executable``."""

    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError("requirements.txt nicht gefunden")

    print("[Installer] Aktualisiere pip …")
    with _activity_indicator("[Installer] Bereite pip vor …"):
        subprocess.run(
            [str(python_executable), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            cwd=BASE_DIR,
            env=_build_env(),
        )

    print("[Installer] Installiere Projekt-Abhängigkeiten …")
    with _activity_indicator("[Installer] Warte auf pip …"):
        subprocess.run(
            [
                str(python_executable),
                "-m",
                "pip",
                "install",
                "-r",
                str(REQUIREMENTS_FILE),
            ],
            check=True,
            cwd=BASE_DIR,
            env=_build_env(),
        )


def _ffmpeg_in_path() -> bool:
    for binary in FFMPEG_BINARIES:
        candidate = shutil.which(binary, path=os.pathsep.join(_PATH_PREFIXES + [os.environ.get("PATH", "")]))
        if candidate:
            continue
        candidate = shutil.which(binary)
        if candidate:
            continue
        return False
    return True


def _local_ffmpeg_dirs() -> list[Path]:
    candidates = []
    for directory in (VENV_BIN, FFMPEG_FALLBACK_DIR):
        if directory.exists() and all((directory / name).exists() for name in FFMPEG_BINARIES):
            candidates.append(directory)
    return candidates


def _ffmpeg_exists_locally() -> bool:
    return bool(_local_ffmpeg_dirs())


def _download_ffmpeg_windows(target_dir: Path) -> None:
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    print("[Installer] Lade FFmpeg (Windows Essentials) herunter …")
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "ffmpeg.zip"
        with urlopen(url) as response, archive_path.open("wb") as fh:
            chunk = 1024 * 256
            with _activity_indicator("[Installer] Download läuft …"):
                while True:
                    data = response.read(chunk)
                    if not data:
                        break
                    fh.write(data)
        with zipfile.ZipFile(archive_path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(tuple(name.lower() for name in FFMPEG_BINARIES))]
            if not members:
                raise RuntimeError("FFmpeg-Archiv enthält keine ausführbaren Dateien")
            for member in members:
                zf.extract(member, tmpdir)
                extracted = Path(tmpdir) / member
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(extracted, target_dir / extracted.name)
    print("[Installer] FFmpeg wurde installiert.")


def _ensure_ffmpeg() -> None:
    local_dirs = _local_ffmpeg_dirs()
    if _ffmpeg_in_path() or local_dirs:
        for directory in local_dirs:
            _prepend_path(directory)
        return

    if os.name == "nt":
        target_dir = VENV_BIN if VENV_BIN.exists() else FFMPEG_FALLBACK_DIR
        _download_ffmpeg_windows(target_dir)
        _prepend_path(target_dir)
        return

    raise RuntimeError(
        "FFmpeg (ffmpeg/ffprobe) wurde nicht gefunden und kann automatisch nur unter Windows installiert werden. "
        "Bitte installiere FFmpeg manuell und stelle sicher, dass es im PATH verfügbar ist."
    )


def _prepend_path(path: Path) -> None:
    resolved = str(path)
    if resolved and resolved not in _PATH_PREFIXES:
        _PATH_PREFIXES.insert(0, resolved)


def main() -> int:
    os.chdir(BASE_DIR)

    python_to_use = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
    if VENV_BIN.exists():
        _prepend_path(VENV_BIN)
    try:
        _ensure_ffmpeg()
    except RuntimeError as exc:
        print(f"[Starter] Hinweis: {exc}")
    except Exception as exc:  # pragma: no cover - Netzwerk-/Dateifehler
        print(f"[Starter] Hinweis: FFmpeg konnte nicht automatisch installiert werden ({exc}).")

    print("[Starter] Starte MemoryBall Studio …")
    exit_code = _launch_application(python_to_use)

    if exit_code == 0:
        return 0

    print("[Starter] Start fehlgeschlagen (Exit-Code {}), führe Installer aus …".format(exit_code))

    try:
        python_to_use = _ensure_venv()
        if VENV_BIN.exists():
            _prepend_path(VENV_BIN)
        _install_requirements(python_to_use)
        _ensure_ffmpeg()
    except subprocess.CalledProcessError as exc:
        print("[Installer] Der Installationsschritt ist fehlgeschlagen.")
        return int(exc.returncode or 1)
    except Exception as exc:  # pragma: no cover - defensive for user setups
        print(f"[Installer] Fehler: {exc}")
        return 1

    print("[Starter] Neuer Versuch nach Installation …")
    return _launch_application(python_to_use)


if __name__ == "__main__":
    sys.exit(main())
