"""Hilfsskript zum Bauen der Windows-Setup-EXE via PyInstaller."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER_DIR = Path(__file__).resolve().parent
INSTALLER_SCRIPT = INSTALLER_DIR / "installer.py"
ICON = ROOT / "assets" / "memoryball.ico"
DIST_DIR = ROOT / "dist"


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # type: ignore # noqa: F401
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def build() -> None:
    ensure_pyinstaller()
    import PyInstaller.__main__

    args = [
        str(INSTALLER_SCRIPT),
        "--onefile",
        "--name",
        "MemoryBallStudioSetup",
        "--noconsole",
    ]
    if ICON.exists():
        args.extend(["--icon", str(ICON)])
    PyInstaller.__main__.run(args)

    installer_dist = INSTALLER_DIR / "dist"
    output = installer_dist / "MemoryBallStudioSetup.exe"
    if output.exists():
        DIST_DIR.mkdir(exist_ok=True)
        shutil.move(str(output), str(DIST_DIR / output.name))
    generated = INSTALLER_SCRIPT.with_suffix(".spec")
    if generated.exists():
        generated.unlink()
    build_dir = INSTALLER_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if installer_dist.exists():
        shutil.rmtree(installer_dist)


if __name__ == "__main__":
    build()
