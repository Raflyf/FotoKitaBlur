# Foto Kita Blur - Peace Sign Privacy Shield

Real-time privacy shield: the webcam feed is automatically blurred the moment
you flash a **peace sign**. Detection runs fully in the browser via MediaPipe
(no frame ever leaves the machine), with an optional Python/OpenCV CLI fallback
for standalone use.

## Features

- Peace sign detection with 5-frame temporal hysteresis (no flickering)
- Gesture **parity** between frontend (JavaScript) and backend (Python) logic
- Extra fun gestures: middle finger crown, finger heart, particle effects
- Hand skeleton overlay, face tracking boxes, diagnostics HUD
- Music playback synced from a local audio file (0:23 - 0:52)
- Responsive dark UI, mobile-optimized detection loop
- Strict Content-Security-Policy, no inline JS handlers

## Architecture

| Layer | Technology |
| --- | --- |
| Web frontend | Vanilla JS + MediaPipe Tasks Vision (`hand_landmarker.task`) |
| Backend | Flask (serves static assets + audio, adds security headers) |
| CLI fallback | Python + OpenCV + MediaPipe (`blur.py`) |
| Gesture logic | Mirror implementations in `main.js` and `blur.py` |

The web app needs the Flask server only to serve assets and audio; all
computer vision runs client-side.

## Requirements

Python 3.10+ (backend / CLI only). Frontend needs no build step.

```bash
pip install -r requirements.txt
```

Pinned versions: Flask 3.1.3, mediapipe 0.10.35, numpy 2.4.6,
opencv-python 4.13.0.92.

## Run

Web app (serves on http://127.0.0.1:5000 by default):

```bash
python app.py
```

CLI camera fallback:

```bash
python blur.py
```

Environment variables for `app.py`:

| Variable | Default | Description |
| --- | --- | --- |
| `FOTO_BLUR_HOST` | `127.0.0.1` | Bind host |
| `FOTO_BLUR_PORT` | `5000` | Bind port |
| `FOTO_BLUR_DEBUG` | `0` | Set to `1` for debug mode |

CLI flags for `blur.py` (`python blur.py --help`):

| Flag | Default | Description |
| --- | --- | --- |
| `--camera` | `0` | Camera device index |
| `--min-confidence` | `0.5` | Detection confidence threshold |
| `--kernel` | `61` | Gaussian blur kernel size (odd, min 3) |
| `--no-enhance` | off | Disable CLAHE low-light enhancement |

Press `ESC` to exit the CLI. Music requires the local `media/music/` files,
which are intentionally **not** tracked in git (copyrighted audio, ~14 MB);
drop `foto-kita-blur.mp3` and `kicau-mania.mp3` in that folder to enable it.

## Gesture Specification

The peace sign is strict: index and middle fingers extended, ring and pinky
folded, a visible V-spread, and thumb tucked. `blur.py` mirrors the JS
heuristics from `main.js` (distance ratios 1.15/0.85, spread > 0.32 palm),
guaranteed by `tests/test_blur.py`.

## Tests

```bash
python -m unittest tests.test_blur -v
```

Covers kernel normalization, peace-sign detection/rejection, middle finger,
and finger heart logic.

## Project Structure

```
app.py                 Flask server + security headers + audio routes
blur.py                OpenCV/MediaPipe CLI detector (gesture parity)
static/main.js         Frontend detection, effects, music, UI
static/models/         MediaPipe hand + face model files
templates/index.html   Single-page UI
tests/test_blur.py     Gesture logic unit tests
media/music/           Local audio (gitignored)
```

## Privacy & Security

- Frames are processed locally; no video is uploaded anywhere
- CSP: `default-src 'self'`, scripts only from self + jsdelivr (with
  `'wasm-unsafe-eval'` required by the MediaPipe WASM runtime), `nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`
- Audio routes use fixed whitelist keys; arbitrary paths return 404
