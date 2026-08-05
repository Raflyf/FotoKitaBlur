# Foto Kita Blur — Peace Sign Privacy Shield

Your webcam is always watching. In the middle of a video call, a live stream,
or a class, the one thing you cannot afford is being caught off-guard — a
private moment, a sensitive screen, a face you did not intend to show. Most
privacy tools make you fumble for a button. Foto Kita Blur takes a different
approach: **your hand is the button.**

Flash a **peace sign** (✌️) at the camera and every face in the frame is
instantly blurred, until you lower your hand. No keyboard, no menu, no delay.
It is a privacy shield you can operate with one gesture, in half a second,
without breaking eye contact with the conversation.

## Why this exists

The idea came from a simple observation: people who blur their faces on
camera — streamers, students, privacy-conscious workers — all have the same
problem. Toggling the blur is a multi-step chore (alt-tab, find the button,
click, go back), so they end up leaving the blur on permanently or forgetting
it entirely. A gesture-based switch removes the cost of using privacy
protection, which means it actually gets used.

The project is deliberately playful as well: the peace sign blurs faces, but
other gestures trigger lighter effects — hearts, a crown, even a scuba-diving
cat. Privacy does not have to feel like a panic button.

## How it works

Everything runs on your device. The app opens your webcam and feeds each frame
through two MediaPipe models loaded in the browser:

1. **Hand landmarker** — tracks 21 landmarks per hand, which lets the app tell
   *which fingers are up* rather than just "there is a hand".
2. **Face detector** — locates every face in the frame so effects (blur,
   crowns) can be applied precisely to the right person.

A detection loop runs ~15 times per second. Each gesture has its own logic
with **temporal hysteresis** — a gesture must be detected consistently for a
few consecutive frames before it triggers, and it stays active for a short
window after you lower your hand, so a single missed frame never makes the
effect flicker. When the peace sign is active, a CSS blur filter is applied
over the canvas; the raw frames themselves are never stored or transmitted.

If you do not want to run a browser at all, `blur.py` is a standalone
Python/OpenCV CLI that implements the exact same gesture logic (the heuristics
are mirrored between the two, and locked in by unit tests).

## Gestures

| Gesture | How to do it | Effect |
| --- | --- | --- |
| ✌️ Peace (V) | Index and middle finger up, ring and pinky tucked | Faces get blurred |
| 🫰 Love, one hand | Thumb tip and index tip pinched into a small heart | Heart emoji floats up |
| ❤️ Love, two hands | Both hands form one big heart (index tips touching) | Two heart emojis, one per hand |
| 🖕 Middle finger | Middle finger up, the rest tucked | Cheeky crown + emoji |
| 🤿 Scubacat | One hand covers the nose, the other waves fast | Scuba-diving cat + birdsong audio |

The in-app **Kamus Gestur** panel (a toggleable dictionary in Bahasa
Indonesia, remember your choice between visits) explains each gesture on
screen, so new users never need the docs.

Detection is strict on purpose: the peace sign requires fully extended index
and middle fingers, folded ring and pinky, a visible V-spread, and a tucked
thumb — so a relaxed hand or a wave never accidentally blurs your screen.

## Running the web app

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000, allow camera access, click **Start Camera**,
and try the gestures. No build step; the frontend is vanilla JavaScript.

Environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `FOTO_BLUR_HOST` | `127.0.0.1` | Bind host |
| `FOTO_BLUR_PORT` | `5000` | Bind port |
| `FOTO_BLUR_DEBUG` | `0` | Set to `1` for debug mode |

## Running the CLI fallback

```bash
python blur.py
```

| Flag | Default | Description |
| --- | --- | --- |
| `--camera` | `0` | Camera device index |
| `--min-confidence` | `0.5` | Detection confidence threshold |
| `--kernel` | `61` | Gaussian blur kernel size (odd, min 3) |
| `--no-enhance` | off | Disable CLAHE low-light enhancement |

Press `ESC` to exit. Requires Python 3.10+ and a working camera.

> Music: the web app can play a soundtrack (0:23–0:52) and a birdsong clip
> for the scuba cat, but the audio files are copyrighted and intentionally
> not tracked in git. Drop `foto-kita-blur.mp3` and `kicau-mania.mp3` into
> `media/music/` to enable them.

## Architecture

| Layer | Technology |
| --- | --- |
| Web frontend | Vanilla JS + MediaPipe Tasks Vision (`hand_landmarker.task`) |
| Backend | Flask — serves assets + audio, adds security headers |
| CLI fallback | Python + OpenCV + MediaPipe (`blur.py`) |
| Gesture logic | Mirrored implementations in `main.js` and `blur.py` |

Flask exists only to serve the static app and audio over HTTP; all computer
vision runs client-side, and the CLI is a self-contained alternative for
people who prefer the terminal.

## Project structure

```
app.py                 Flask server + security headers + audio routes
blur.py                OpenCV/MediaPipe CLI detector (gesture parity)
static/main.js         Frontend detection, effects, music, UI
static/models/         MediaPipe hand + face model files
templates/index.html   Single-page UI
tests/test_blur.py     Gesture logic unit tests
media/music/           Local audio (gitignored)
```

## Testing

```bash
python -m unittest tests.test_blur -v
```

The tests cover kernel normalization, peace-sign acceptance and rejection
(including hands that should *not* trigger it), the middle-finger and finger
heart heuristics, and guard the parity between the Python and JavaScript
logic.

## Privacy & security

- All frames are processed locally in the browser; nothing is uploaded
- Strict Content-Security-Policy: scripts only from self + jsdelivr, with
  `'wasm-unsafe-eval'` required by the MediaPipe WASM runtime
- `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`
- Audio routes use a fixed whitelist of keys; arbitrary paths return 404

## Built with

MediaPipe Tasks Vision, vanilla JavaScript, Flask, OpenCV, NumPy — pinned
versions in `requirements.txt` (Flask 3.1.3, mediapipe 0.10.35, numpy 2.4.6,
opencv-python 4.13.0.92).
