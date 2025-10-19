"""Windows installer bootstrap for Memory Ball Studio."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

try:
    from urllib import request
except Exception as exc:  # pragma: no cover - unlikely import issue
    raise SystemExit(f"Konnte urllib.request nicht importieren: {exc}")

GITHUB_ZIP_URL = "https://github.com/MemoryBall/memoryball-studio/archive/refs/heads/main.zip"
INSTALL_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "MemoryBallStudio"
VENV_DIR = INSTALL_ROOT / ".venv"
START_SCRIPT = INSTALL_ROOT / "start.py"
REQUIREMENTS_FILE = INSTALL_ROOT / "requirements.txt"
ICON_PATH = INSTALL_ROOT / "assets" / "memoryball.ico"


class InstallerError(RuntimeError):
    """Custom error type for installer failures."""


def log(message: str) -> None:
    print(f"[MemoryBallStudio Installer] {message}")


def download_repo(destination: Path) -> None:
    zip_path = destination / "repo.zip"
    log(f"Lade Repository aus {GITHUB_ZIP_URL} ...")
    with request.urlopen(GITHUB_ZIP_URL) as response, open(zip_path, "wb") as target:
        shutil.copyfileobj(response, target)
    log("Entpacke Download ...")
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.namelist() if not m.endswith("/")]
        if not members:
            raise InstallerError("Leeres Archiv erhalten.")
        top_level = members[0].split("/")[0]
        archive.extractall(destination)
    extracted = destination / top_level
    if not extracted.exists():
        raise InstallerError("Entpacktes Repository konnte nicht gefunden werden.")
    log(f"Kopiere Dateien nach {INSTALL_ROOT} ...")
    INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
    for child in extracted.iterdir():
        target = INSTALL_ROOT / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def run_command(command: Iterable[str], **kwargs) -> None:
    log("Starte: " + " ".join(map(str, command)))
    result = subprocess.run(command, check=False, **kwargs)
    if result.returncode != 0:
        raise InstallerError(
            f"Kommando {' '.join(map(str, command))} schlug fehl (Code {result.returncode})."
        )


def ensure_venv() -> Path:
    if not VENV_DIR.exists():
        log("Erstelle virtuelle Umgebung ...")
        run_command([sys.executable, "-m", "venv", str(VENV_DIR)])
    scripts_dir = VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
    python_exe = scripts_dir / ("pythonw.exe" if os.name == "nt" else "python")
    if not python_exe.exists():
        python_exe = scripts_dir / "python.exe"
    if not python_exe.exists():
        raise InstallerError("Python-Interpreter in der virtuellen Umgebung nicht gefunden.")
    run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    if REQUIREMENTS_FILE.exists():
        run_command([str(python_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
    else:
        log("Warnung: requirements.txt nicht gefunden – überspringe Dependency-Installation.")
    return python_exe


def create_shortcut(python_exe: Path) -> None:
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    shortcut_path = desktop / "Memory Ball Studio.lnk"
    if not desktop.exists():
        log(f"Desktop-Ordner {desktop} nicht gefunden – überspringe Shortcuts.")
        return
    if not START_SCRIPT.exists():
        raise InstallerError(f"Start-Script {START_SCRIPT} fehlt.")
    icon = ICON_PATH if ICON_PATH.exists() else START_SCRIPT
    ps_script = (
        "$Shell = New-Object -ComObject WScript.Shell;\n"
        "$Shortcut = $Shell.CreateShortcut('{shortcut}');\n"
        "$Shortcut.TargetPath = '{target}';\n"
        "$Shortcut.Arguments = '\"{argument}\"';\n"
        "$Shortcut.WorkingDirectory = '{working}';\n"
        "$Shortcut.IconLocation = '{icon},0';\n"
        "$Shortcut.Save();\n"
    ).format(
        shortcut=str(shortcut_path).replace("'", "''"),
        target=str(python_exe).replace("'", "''"),
        argument=str(START_SCRIPT).replace("'", "''"),
        working=str(INSTALL_ROOT).replace("'", "''"),
        icon=str(icon).replace("'", "''"),
    )
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False) as script_file:
        script_file.write(ps_script)
        script_path = Path(script_file.name)
    try:
        run_command(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ]
        )
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except AttributeError:
            # Python < 3.8 compatibility
            if script_path.exists():
                script_path.unlink()


def main() -> None:
    log("Starte Memory Ball Studio Installation ...")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            download_repo(Path(tmp_dir))
        python_exe = ensure_venv()
        create_shortcut(python_exe)
        log("Installation abgeschlossen! Ein Desktop-Icon wurde erstellt.")
    except InstallerError as error:
        log(f"Fehler: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
