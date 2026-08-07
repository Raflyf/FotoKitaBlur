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
## Fix run 2026-08-06 (FIX-27)

- **gesture predicates extracted** to `static/gestures.js` (single source for
  browser + tests); `main.js` imports them. Removes the unguarded JS duplicate
  that mirrored `blur.py` with zero automated coverage.
- **tests/gestures.test.mjs** (node:test) mirrors `tests/test_blur.py`
  fixtures -> the real frontend product logic now has coverage. Run:
  `node --test tests/gestures.test.mjs`.
- **interpolateSnapshot** pairs spawns by nearest position (not array index)
  so hearts no longer lerp across the screen when spawn counts change.
- **getUserMedia** capped at 1280x720 (canvas is clamped to 800px wide).
- Verified: py_compile ok, `python -m unittest tests.test_blur` 11/11,
  `node --test tests/gestures.test.mjs` 8/8, `node --check` both JS files.
- Pushed to `raflyf` remote (origin returned 403; no access to `Raflyf02`).

## Notes (not fixed, by design)

- Whole-canvas CSS blur vs "memblur wajah" claim: peace blurs the entire
  frame, not per-face regions. A product decision, not a logic bug.
- Audio loop times (133/146/52) are hard-coded offsets into the MP3s; brittle
  if the files are re-encoded, harmless as-is.

## Fix run 2026-08-06 (FIX-28, scubacat false-trigger)

- Opening both palms no longer triggers scubacat. Detection now requires the
  intended gesture: nose hand is a FIST (all four fingers folded), the other
  hand is an OPEN palm, and the wave is HORIZONTAL (x-dominant, >=2 reversals).
- Removed the WAVING_TRAVEL_MIN single-swipe branch (one-way raises triggered
  the cat with zero reversals) and its now-dead constant.

## FIX-36 — Refine isFingerHeart thresholds (false positive + false negative)

**Date:** 2026-08-05
**Commit:** `238a45b`
**Files:** `static/gestures.js`, `blur.py`, `static/main.js`, `tests/gestures.test.mjs`, `templates/index.html`

### Root cause

`isFingerHeart` had three weak gates that let non-love poses (pistol /
thumb-up) through while rejecting real finger hearts:

1. **index/middle ratio 1.20** — a pointing hand easily exceeds this.
2. **thumb-index distance 0.40 × palm** — far too generous; pistol hands
   project thumb and index tip close together in 2D.
3. **fold threshold 1.30** — middle finger could be nearly straight and
   still pass.

### Changes

| Gate | Before | After | Direction |
|------|--------|-------|-----------|
| index/middle ratio | 1.20 | 1.15 | slightly easier for true hearts |
| fold threshold | 1.30 | 1.45 | more tolerant of real hands |
| thumb-index distance | 0.40 × palm | 0.25 × palm | much tighter — rejects non-pinches |
| thumb Y tolerance (JS) | 0.15 × palm | kept | — |
| thumb Y tolerance (Python) | 0.05 × palm | 0.15 × palm | loosened for tilted hands |
| cache-bust version | v=7 | v=8 | force browser reload |

Python `is_finger_heart` synced to match JS thresholds exactly.

### Tests

- Added `pistolHand()` fixture + assertion in `tests/gestures.test.mjs`.
- All 9 JS + 11 Python tests pass.

### Also in this commit

- Removed orphan scubacat glossary card from `templates/index.html`
  (leftover from FIX-34).

## FIX-37 — Require crossed thumb/index for finger heart

**Date:** 2026-08-07
**Commit:** `aaf5d8f`
**Files:** `static/gestures.js`, `blur.py`, `static/main.js`, `tests/gestures.test.mjs`, `tests/test_blur.py`

### Root cause

After FIX-36, the `isFingerHeart` predicate successfully rejected "pistol"
hands but still false-triggered on a "pinch" (thumb tip and index tip touching
but not crossing). A true Korean finger heart crosses the thumb tip over the
index finger.

### Changes

- Added crossing check: `Math.sign(landmarks[4].x - landmarks[6].x) !== Math.sign(landmarks[8].x - landmarks[6].x)`. This forces the thumb tip and index tip to sit on opposite sides of the index PIP joint.
- Synced identical check to `blur.py`.
- Bumped `gestures.js?v=9` in `main.js`.
- Added `pinchHand` (rejected) and `crossedFingerHeartHand` (accepted) tests.

## FIX-38 — Replace pinch distance with 2D line-segment intersection

**Date:** 2026-08-07
**Commit:** `35ff02e`
**Files:** `static/gestures.js`, `blur.py`, `static/main.js`, `tests/gestures.test.mjs`, `tests/test_blur.py`

### Root cause

Previous thresholds required thumb tip and index tip to be close in 2D space
(`< 0.25 × palm`). In real finger hearts (as shown in user photos), the index
finger extends across/past the thumb so their tips do not touch; while non-love
pinches (where tips touch on the same side) were falsely passing.

### Changes

- Removed `distThumbIndex` distance gate completely.
- Added 2D line segment intersection gate using CCW (counter-clockwise)
  determinant test: checks whether the thumb shaft (landmarks 3→4) visually
  crosses the index shaft (landmarks 6→8).
- Fold thresholds relaxed to `1.65×` so hands tilted toward/away from the camera
  do not fail fold checks.
- Synced identical CCW logic to Python `blur.py`.
- Bumped `gestures.js?v=10` in `main.js`.

### Tests

- All 9 JS tests (`tests/gestures.test.mjs`) and 11 Python tests
  (`tests/test_blur.py`) pass.

## FIX-39 — Gate heart crown & ambient particles on actual heart gesture

**Date:** 2026-08-07
**Commit:** `d72105b`
**Files:** `static/main.js`, `tests/gestures.test.mjs`

### Root cause

The **Show Heart Crown** checkbox (`showCrown`) was gating *ambient heart particles*
and *crown emojis* solely on whether a face was visible. As long as the checkbox
was on and any face existed, `lastHeartState` became `true`, causing continuous
heart-particle spawning and crown rendering even when the user was **not** doing
a finger-heart gesture. This made it *look* like the heart gesture was triggering
from a light touch / non-gesture.

### Changes

- Added `isFaceHeart` flag during face-gesture assignment; set when a heart
  gesture is matched to a face within `GESTURE_FACE_GATE`.
- Propagate `isHeart` to the crown data (`newCrowns`).
- Crown rendering now emits heart emojis **only** for faces with `isHeart === true`
  when the "Show Heart Crown" checkbox is on (and cheeky emojis only for
  `isCheeky` faces when the "Show Middle Finger Crown" checkbox is on).
- `lastHeartState` now derived from `newCrowns.some(c => c.isHeart)` instead of
  `newCrowns.length > 0`, so ambient particles spawn **only** during/after real
  heart gestures.
- Added `nearTouchHand()` regression test to ensure fingertips merely touching
  does not trigger heart.
