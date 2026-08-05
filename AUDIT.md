# Project Audit & Changes

Audit run 2026-08-05 against the `foto_kita_blur` codebase. Restore point
before this audit: commit `8d21dd3`.

## Findings (before)

- **app.py** terminated the process via `os._exit(0)`/`sys.exit` workaround
  instead of a clean shutdown; no environment config, no security headers,
  no error handlers, no path-confined asset serving.
- **blur.py** created a fresh `cv2.createCLAHE` object every frame (stateful,
  wasteful); `is_peace()` used lax thresholds that did not match the frontend
  logic (frontend could engage blur while the CLI could not).
- **main.js** carried dead state (`faceDetected`, `lastPeaceState`); queried
  `#finger-* .finger-status` five times per frame; re-created `isMobile`
  three times; exposed controls on `window`; mutated checkbox DOM refs per
  event.
- **index.html** used inline `onclick`/`oninput`/`onchange` handlers and
  inline styles (blocks a strict CSP).
- **style.css** contained an unused `.placeholder-icon` rule.
- Two copyrighted MP3 files (~14 MB) were tracked in git; two dead assets
  (`cat-scuba-scuba.gif`, `static/absolute-cinema.webp`) unreferenced.
- `requirements.txt` pinned nothing; `.gitignore` did not cover media.

## Changes applied

### app.py (rewritten)
- Config through `FOTO_BLUR_HOST` / `FOTO_BLUR_PORT` / `FOTO_BLUR_DEBUG`.
- Security headers on every response: CSP (self + jsdelivr, fonts, etc.),
  `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`.
- Audio served from `media/music/` via a fixed-key whitelist
  (`AUDIO_FILES`), `send_from_directory(..., conditional=True)`.
- JSON error handlers for 404/500 with request logging.
- CSP correction (found during browser runtime test): MediaPipe Tasks Web is
  a WASM binary, so `script-src` must allow `'wasm-unsafe-eval'`; without it
  the model fails to instantiate and the UI stays disabled forever.

### blur.py (rewritten)
- `LANDMARKS` id map; CLAHE allocated once in `__init__`.
- `is_peace()` now matches frontend heuristics (1.15 / 0.85 distance ratios,
  spread > 0.32 palm, thumb tucked, degenerate-palm guard).
- Added `is_middle_finger()` and `is_finger_heart()`.
- Temporal hysteresis via `PEACE_HOLD_FRAMES = 5`.
- `_normalize_kernel()` clamps to min 3 and enforces odd size.
- `process_frame()` returns `(frame, blur_active, hand_count)`; HUD with fps.
- `main()` with `argparse` (camera / confidence / kernel / no-enhance) and
  `try/finally` camera + detector release.

### static/main.js
- Removed dead `faceDetected` and `lastPeaceState`.
- Single `IS_MOBILE` constant; face detection throttled every 2nd frame on
  mobile only.
- Cached `fingerStatusEls`, checkbox and button refs (no per-frame/rerendering
  DOM lookups).
- `runDetection()` guards models not-ready without swallowing the
  `isDetecting` reset; `lastFaceResults` reset on stop.
- Removed `window.*` exposure; inline handlers moved to `addEventListener`
  (CSP-compatible).
- FIX-22: `isFingerHeart()` thresholds relaxed (folded 0.9 -> 1.05, thumb-index
  0.70 -> 0.85 palm) so the Korean heart triggers reliably with partial curls;
  mirrored in `blur.py` for parity. Two-hand classic heart now emits one emoji
  per hand (at each index tip, deduped against one-hand hearts) and the
  per-face cap is 2 hearts (left + right hand) with global gesture consumption.
- FIX-23: responsive gesture release + multi-hand parity.
  - Hold/latch durations shortened so effects vanish promptly: peace 10 -> 4,
    scubacat 15 -> 8, heart/cheeky 8 -> 3, face tracks 5 -> 4.
  - Two-hand classic heart gained its own 3-frame latch (hand pairs have no
    stable ID) with a snapshot of index-tip positions, and its threshold was
    widened 0.5 -> 0.6 * avgPalm.
  - Cheeky (middle finger) now allows 2 emojis per face (one per hand), same
    global-consumption scheme as hearts.
  - A peace/V sign kills the heart latch immediately and wins over the heart,
    so forming a peace sign can no longer false-trigger the love emoji;
    `isFingerHeart()` fold threshold raised 1.05 -> 1.12 (below the 1.15x
    peace extension) to close the false-trigger zone during peace transitions,
    mirrored in `blur.py`.
- FIX-24: added "Kamus Gestur" — a Bahasa Indonesia hand-gesture glossary
  panel (5 cards: Peace, Love Satu/Dua Tangan, Jari Tengah, Scubacat) with an
  on/off toggle button. CSP-strict (no inline handlers); open state persists
  via localStorage, closes with the X button or Escape, respects
  `prefers-reduced-motion`, and the panel is aria-labeled/expandable.
- FIX-25: scubacat was nearly impossible to trigger. The nose-hand gate only
  checked fingertips (thumb/index), but the pose is a FIST covering the nose —
  now the palm center (landmark 9) counts too and the gate widened
  (0.12/0.6 -> 0.14/0.7 palm). The waving-hand proximity gate widened
  (1.2x/1.5x -> 1.6x/2.0x face bbox) so a horizontal wrist swipe swinging past
  the face still belongs to that person. Waving detection now accepts a single
  fast one-way swipe via a total-travel metric (`WAVING_TRAVEL_MIN` 0.30
  palm-normalized), not just 2+ reversals; `WAVING_RANGE_MIN` lowered 0.2 ->
  0.15 and charge/trigger rebalanced (18/25 -> 25/20) so one confirmed frame
  fires the gesture.
- FIX-26: scubacat triggered but the birdsong (kicau) never played. The audio
  file and `/kicau` route were fine; the cause was Chrome's autoplay policy —
  `kicauAudio.play()` is called from inside the detection loop, not a direct
  user gesture, so playback was silently blocked. Fixed by unlocking the
  element on the first real interaction (pointerdown/touchstart/keydown:
  play-then-pause once).

### templates/index.html
- Removed all inline event handlers and inline layout styles.
- Added meta description, theme-color, and an SVG data-URI favicon.
- Buttons/switch rows use new classes (`.btn-flex`, `.btn-music`, `.switch-row`).
- FIX-24: added "Kamus Gestur" toggle button and glossary panel markup.

### static/style.css
- Added `.btn-flex`, `.btn-music`, `.switch-row`; removed dead
  `.placeholder-icon`.
- FIX-24: added glossary styles (`.btn-glossary.active`, `.glossary-panel`,
  cards, grid, entry animation) with `prefers-reduced-motion` guard.

### Media / git
- MP3s moved to `media/music/` (renamed to machine-safe slugs), staged
  `git rm --cached`; `.gitignore` now covers `media/` and audio extensions.
- Dead assets `cat-scuba-scuba.gif` and `static/absolute-cinema.webp` deleted.
- `requirements.txt` pinned to installed versions.

### Docs & tests
- `README.md` rewritten (setup, config, gestures, structure, privacy).
- `tests/test_blur.py` (unittest) covers kernel normalization, peace
  detection/rejection, middle finger, finger heart; all pass.

## Verification

- `python -m py_compile app.py blur.py` - ok
- `node --check static/main.js` - ok
- `python -m unittest tests.test_blur -v` - 11/11 ok
- Flask smoke test - `/` 200 + CSP header; `/music` and `/kicau` stream the
  audio with correct media type; path traversal `/etc/passwd` -> 404.
- Browser runtime (Chrome DevTools MCP): models instantiate and both MediaPipe
  graphs start; camera stream 1920x1080 (readyState 4); no console errors.

## Remaining debt / notes

- Frontend detection thresholds live in `main.js` and are mirrored manually in
  `blur.py`; keep them in sync (tests guard the Python side only).
- Music files are gitignored by design (licensing); the app needs a local
  `media/music/` folder to play audio.
- No CI yet; suggest running the unittest step on push.