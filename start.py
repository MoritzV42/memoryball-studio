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

import os
import subprocess
import sys
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent
MAIN_FILE = BASE_DIR / "main.py"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
VENV_DIR = BASE_DIR / "venv"
VENV_BIN = VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
VENV_PYTHON = VENV_BIN / ("python.exe" if os.name == "nt" else "python")


def _run_process(cmd: List[str]) -> int:
    """Run ``cmd`` inheriting stdio and return the exit code."""

    completed = subprocess.run(cmd, cwd=BASE_DIR)
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
    subprocess.run([str(python_executable), "-m", "pip", "install", "--upgrade", "pip"], check=True, cwd=BASE_DIR)

    print("[Installer] Installiere Projekt-Abhängigkeiten …")
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
    )


def main() -> int:
    os.chdir(BASE_DIR)

    python_to_use = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)

    print("[Starter] Starte MemoryBall Studio …")
    exit_code = _launch_application(python_to_use)

    if exit_code == 0:
        return 0

    print("[Starter] Start fehlgeschlagen (Exit-Code {}), führe Installer aus …".format(exit_code))

    try:
        python_to_use = _ensure_venv()
        _install_requirements(python_to_use)
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
