# memoryball-autocrop

memoryball-autocrop ist ein CLI- und GUI-Tool zum automatischen Zuschneiden von Fotos und Videos auf quadratische 480×480 Pixel. Gesichter werden mit MediaPipe erkannt, der Crop folgt dem Gesicht dank Glättung. Videos werden über ffmpeg kodiert, Audio bleibt optional erhalten.

## Features

* Stapelverarbeitung für Bilder (JPG/PNG/HEIC/WebP) und Videos (MP4/MOV/MKV/AVI)
* Automatische Gesichtserkennung mit MediaPipe Face Detection
* Glättung der Bounding-Box via exponentiellem gleitenden Mittel
* Fallback: sicherer Center-Crop, optional mit Padding
* Video-Export über ffmpeg inkl. Audio-Kontrolle
* Multithreading für Bilder, komfortable Tkinter-Oberfläche mit Vorschau
* Manuelle Anpassung des Zuschnitts pro Bild inkl. Vorschau

## Installation

Voraussetzung: Python 3.10 oder neuer sowie ffmpeg/ffprobe auf dem System.

```bash
ffmpeg -version
```

Falls ffmpeg fehlt:

* **Windows**: `choco install ffmpeg` oder ZIP von [ffmpeg.org](https://ffmpeg.org) entpacken und PATH setzen
* **macOS**: `brew install ffmpeg`
* **Linux**: `sudo apt install ffmpeg`

Projekt installieren:

```bash
git clone <repo>
cd memoryball-autocrop
pip install -r requirements.txt
```

## CLI-Nutzung

```bash
python main.py --input "C:\in" --output "C:\out" --mode auto --size 480 --min-face 0.12 \
  --quality 90 --threads 4 --fps keep --image-format jpg --video-ext mp4 --face-priority largest
```

Wichtige Parameter:

| Parameter | Beschreibung |
|-----------|--------------|
| `--input` | Datei oder Ordner (rekursiv) |
| `--output` | Zielordner (wird angelegt) |
| `--mode` | `auto`, `center`, `manual` (manuell nutzt `--crop-*` als Startwerte) |
| `--size` | Ziel-Kantenlänge (Standard 480) |
| `--fps` | Zahl oder `keep` |
| `--quality` | Bildqualität (1–100) |
| `--crf` | Video-CRF (Standard 20) |
| `--preset` | ffmpeg-Preset (Standard `medium`) |
| `--min-face` | Minimale Gesichtsfläche relativ zu kleinster Bildkante |
| `--face-priority` | Auswahl bei mehreren Gesichtern (`largest`/`center`/`all`) |
| `--threads` | Threads für Bilder |
| `--pad` | Optionales Padding (z. B. `0.05` für 5 %) |
| `--image-format` | `jpg`, `png` oder `webp` |
| `--video-ext` | Aktuell `mp4` |
| `--keep-audio` | `on`/`off` |
| `--log-level` | `info` oder `debug` |

### Beispiele

Nur Bilder verarbeiten:

```bash
python main.py --input ./bilder --output ./export --image-format jpg --no-face
```

Nur Videos mit Audio behalten:

```bash
python main.py --input ./videos --output ./export --fps keep --keep-audio on --threads 2
```

Gemischter Ordner mit Padding und reduzierter FPS:

```bash
python main.py --input ./medien --output ./export --pad 0.05 --fps 30 --quality 95
```

## GUI

Die GUI startet automatisch, sobald `main.py` ohne Parameter geöffnet wird (z. B. per Doppelklick). Alternativ kann sie auch explizit über die Konsole gestartet werden:

```bash
python main.py --gui
```

**Workflow:**

1. Eingabeordner wählen – der Ausgabeordner wird automatisch als `Converted <Ordnername>` vorgeschlagen.
2. Bilder in der Liste auswählen, automatische Erkennung prüfen und bei Bedarf mit Zoom- und Positions-Slidern anpassen.
3. Videos werden automatisch mitbearbeitet und nutzen die gleichen Einstellungen.
4. Mit „Konvertieren“ die Ausgabe erstellen; der Fortschritt wird angezeigt.

## Performance-Tipps

* Erhöhe `--threads` für viele Bilder (CPU-Kerne berücksichtigen)
* Nutze schnellere ffmpeg-Presets (`--preset fast`) für schnelleren Videoexport
* Bei quadratischen Quellen greift ein schneller Pfad: lediglich Resize statt Crop

## Troubleshooting

* **HEIC wird nicht gelesen** – Stelle sicher, dass `pillow-heif` installiert ist und die Datei nicht DRM-geschützt ist.
* **Beschädigte Metadaten** – ffmpeg/ffprobe können bei fehlerhaften Dateien abbrechen. Die Anwendung loggt Warnungen und verarbeitet den Rest weiter.
* **Gesicht wird nicht erkannt** – Erhöhe `--min-face` nicht zu stark, nutze `--face-priority center` oder deaktiviere die Erkennung (`--no-face`).

## Tests

Einfacher Testlauf (erstellt Dummy-Bilder/-Videos und prüft die Ausgabegrößen):

```bash
pytest
```

## Schritt-für-Schritt lokal

1. ffmpeg installieren (siehe oben)
2. Repository klonen
3. `pip install -r requirements.txt`
4. Beispiel: `python main.py --input "D:\Rohmaterial" --output "D:\MemoryBall" --size 480 --fps keep --threads 6 --min-face 0.1`
