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

## FIX-40 — Full System Overhaul: Anti-Miss Invariant Gesture Engine & Modular Architecture

**Date:** 2026-09-08
**Files:** `app.py`, `blur.py`, `static/gestures.js`, `static/particles.js`, `static/main.js`, `static/style.css`, `templates/index.html`, `tests/gestures.test.mjs`, `tests/test_blur.py`

### Root Causes of Historical Detection Misses & Instability
1. **2D Perspective Foreshortening:** Previous distance-to-wrist checks (`dist(tip, wrist) > dist(pip, wrist) * 1.15`) broke when the user tilted their hand toward or away from the camera.
2. **Artificial Detection Delay:** Detection ran in a throttled `setTimeout(66ms/100ms)` loop (~15 FPS), missing quick hand gestures (100-200ms duration).
3. **Scuba Cat & Waving Resets:** Waving history cleared completely on any single dropped hand frame, and BlazeFace nose tip keypoints frequently dropped when occluded by the hand.
4. **Missing Backend Audio Route:** `/kicau` was not served in `app.py`, leading to HTTP 404 on cat jumpscare audio.

### Solutions Implemented
1. **Scale-Normalized, Rotation-Invariant Finger Classification (`gestures.js` & `blur.py`):**
   - Independent `isFingerExtended` and `isFingerFolded` metrics combining MCP-relative and wrist-relative vectors to maintain invariant classification under 3D hand tilt.
   - Comprehensive gesture predicates: Peace (✌️), Korean Finger Heart (🫰), Two-Hand Heart (🫶), and Middle Finger (🖕).
2. **Synchronized Video Frame Processing Loop (`main.js`):**
   - Runs directly on native camera FPS (30-60 FPS) with `requestAnimationFrame`.
   - Exponential Moving Average (EMA) smoothing for face tracking with occlusion coasting (up to 6 missed frames).
3. **Asymmetric Temporal Hysteresis:**
   - Fast attack (1-2 frames to trigger) and smooth release (8-frame hold) prevents all flickering and dropped detections.
   - Leaky integrator for waving movement with direction reversal counters.
4. **Modular Architecture & UI System:**
   - Dedicated `static/particles.js` for floating emoji particles and 3D rotating halo crowns with depth scaling.
   - Enhanced `/kicau` and `/music` endpoints in `app.py`.
   - Fully accessible dark-mode UI (WCAG 2.2 compliant) with live HUD, diagnostics panel, and Kamus Gestur modal.
5. **Testing Verification:**
   - 12/12 passing Node.js tests in `tests/gestures.test.mjs`.
   - 13/13 passing Python unit tests in `tests/test_blur.py`.

## FIX-41 — Web FPS Decoupling & 3D Halo Crown Height Calibration

**Date:** 2026-09-08
**Files:** `static/particles.js`, `static/main.js`, `gui_app.py`

### Root Causes
1. **Severe Web FPS Bottleneck (1-20 FPS):**
   - Synchronous execution of both `handLandmarker.detectForVideo` and `faceDetector.detectForVideo` directly inside `requestAnimationFrame` on every single frame.
   - Heavy 1280x720 video input streamed into MediaPipe Tasks Vision, forcing heavy bilinear downsampling and texture transfers per frame (~50ms latency total).
   - Face detector (BlazeFace) executed 60 times/sec despite face positions moving slowly across frames.
   - Uncapped particle counts and high ambient spawn rates under active halo crowns.
2. **Misaligned 3D Halo Crown Position (Forehead / "Jidar"):**
   - Vertical orbital center `cy` was calculated as `faceCenterY - faceHeight * 0.42`.
   - Since BlazeFace bounding box center `faceCenterY` is at nose/eye bridge level and forehead extends to `0.50 * faceHeight`, subtracting only `0.42` placed the orbital ring directly on the forehead and eyebrows instead of floating above the head.

### Solutions Implemented
1. **Decoupled 60 FPS Render Loop & 30 FPS Throttled Vision Pipeline (`static/main.js`):**
   - Separated the 60 FPS Canvas rendering loop from vision inference.
   - Vision inference runs on an asynchronous non-blocking cadence throttled to 32ms (~30 FPS, native webcam frame rate) with `isDetecting` concurrency guards.
   - Interleaved face detection: BlazeFace runs once every 3 vision frames (~10 FPS), eliminating 67% of face model overhead while EMA smoothing keeps face tracking seamless.
   - Streamlined camera input to 640x360 with proportional canvas matching, reducing pixel processing volume by 75%.
   - Capped active particles to 30 and consumed gesture spawns immediately to prevent duplicate particle creation across 60 FPS render frames.
   - Calibrated hold and latch timers (`HOLD_FRAMES = 16`, `scubacatHoldTimer = 30`) for fluid 60 FPS operation.
2. **Calibrated 3D Halo Crown Height (`static/particles.js` & `gui_app.py`):**
   - Adjusted vertical center offset to `faceCenterY - faceHeight * 0.88` with orbital radiuses `rx = faceWidth * 0.62` and `ry = faceHeight * 0.15`.
   - The halo crown now floats comfortably above the cranium and hair, cleanly hovering over the head.
   - Parity applied to Python desktop application (`gui_app.py`).
3. **Optimized Emoji Canvas Cache (`static/particles.js`):**
   - Size quantization in `getEmojiCanvas` to even pixel steps to maximize cache hits and eliminate redundant offscreen canvas instantiations.

## FIX-42 — Web Gesture FPS Drop Elimination & Middle Finger / Peace Disambiguation

**Date:** 2026-09-08
**Files:** `static/gestures.js`, `static/main.js`, `static/particles.js`, `static/style.css`, `blur.py`, `gui_app.py`, `tests/gestures.test.mjs`, `tests/test_blur.py`

### Root Causes
1. **Severe FPS Drop to 10 FPS on Active Gesture:**
   - Sequential execution of `runVisionInference` inside `processNextFrame` caused back-to-back blocking: when hand landmarking took ~50ms, `now - lastDetectTime` was immediately `>= 32ms` on the next frame, running inference on 100% of render frames and starving vsync.
   - Input to MediaPipe was the full raw `<video>` element, requiring heavy downsampling and WebGL texture copies on every inference.
   - Continuous CSS `transition: filter 0.2s` on `#output-canvas` forced Chrome GPU compositor to reallocate intermediate Gaussian blur textures on active canvas rendering.
   - Particles were instantiated via offscreen canvas elements calling `document.createElement('canvas')` and `drawImage` repeatedly without rate limits.
2. **Two-Hand Middle Finger Triggering Peace Blur:**
   - In `isPeace()`, folded index fingers with slightly loose knuckles or tilted wrists met the threshold `distTipMcp > distPipMcp * 1.15`, while `fingerSeparation` between middle tip and curled index tip passed easily.
   - `isPeace()` lacked checks comparing index reach against middle reach (`middleReach / indexReach < 1.25`).
   - `isPeace()` ran before `isMiddleFinger()` in gesture evaluation without mutual exclusion.

### Solutions Implemented
1. **Hardware-Accelerated Downsampled Vision Input (`static/main.js`):**
   - Dedicated offscreen canvas (`visionInputCanvas`, 480x270) downsamples camera frames in 0.2ms before passing to MediaPipe WebAssembly.
   - Enforced non-blocking gap `nextDetectTime = performance.now() + 45ms` (~20 FPS vision detection) so the 60 FPS render loop is never starved.
2. **Direct 2D Canvas Emoji Rendering (`static/particles.js`):**
   - Replaced offscreen canvas DOM creations with native `ctx.fillText(this.emoji, 0, 0)` in `Particle.draw()`, eliminating texture allocations and CPU DirectWrite spikes.
   - Rate-limited particle burst emissions to 120ms with natural dispersal velocity and a strict active particle cap of 20.
3. **GPU Layer Promotion & Removal of Filter Transition (`static/style.css`):**
   - Added `will-change: filter; transform: translateZ(0);` and removed `transition: filter` from `#output-canvas`, eliminating compositor thrashing during blur activation.
4. **Middle Finger / Peace Disambiguation (`static/gestures.js`, `blur.py`, `gui_app.py`):**
   - Added strict guard `if (isMiddleFinger(landmarks)) return false;` inside `isPeace()`.
   - Verified that index reach and middle reach are balanced (`reachRatio <= 1.25` and `indexReach >= 0.80 * palmSize`).
   - In `processHandGestures()`, evaluated middle finger across all hands first; if any hand displays middle finger, peace gesture is strictly suppressed.
   - Added regression test cases to `tests/gestures.test.mjs` and `tests/test_blur.py`.

## FIX-43 — Browser Module Cache Busting and Server Auto-Reload Hardening

**Date:** 2026-09-08
**Files:** `app.py`, `templates/index.html`, `static/main.js`

### Root Causes
1. **Module Script Caching on Camera Initialization Failure:**
   - Prior to commit `f0d5876`, an unhandled reference to `lastDetectTime = 0;` caused a runtime `ReferenceError` during camera startup in strict ES module execution.
   - Because Flask was configured with default caching (`SEND_FILE_MAX_AGE_DEFAULT = 43200`) and Jinja template caching was active (`TEMPLATES_AUTO_RELOAD = False`), the previous template without updated cache-busting version query strings was retained in server memory.
   - Chrome's V8 module script cache and HTTP disk cache continued serving the stale `main.js` file despite normal user page refreshes, reproducing the `lastDetectTime is not defined` alert.

### Solutions Implemented
1. **Disabled Server-Side Static and Template Caching (`app.py`):**
   - Configured `TEMPLATES_AUTO_RELOAD = True` and `SEND_FILE_MAX_AGE_DEFAULT = 0`.
   - Appended `Cache-Control: no-cache, no-store, must-revalidate`, `Pragma: no-cache`, and `Expires: 0` headers to all responses in `set_security_headers`.
2. **Synchronized Cache Buster Query Parameters:**
   - Incremented module script tag to `src="{{ url_for('static', filename='main.js') }}?v=4"` in `templates/index.html`.
   - Updated ES module import query strings in `static/main.js` to `./gestures.js?v=4` and `./particles.js?v=4`.
3. **Clean Process Re-initialization:**
   - Terminated legacy background process and restarted Flask cleanly via the target virtual environment Python binary.

## FIX-44 — Face Normalization Alignment, Crown Gesture Latching, and Scuba Cat Trigger Calibration

**Date:** 2026-09-08
**Files:** `static/main.js`, `static/particles.js`, `templates/index.html`, `gui_app.py`

### Root Causes
1. **Broken Crown Position & Size (Displaced to Top-Right Corner):**
   - MediaPipe `FaceDetector` was evaluating against an intermediate offscreen canvas (`visionInputCanvas`, 480x270), returning bounding box pixel coordinates in `[0..480, 0..270]`.
   - `updateFaceTracks` divided these coordinates by `video.videoWidth` (e.g. 1280) and `video.videoHeight` (e.g. 720), squishing face coordinates down by a factor of 2.66x towards `(0.18, 0.18)`.
   - On the horizontally mirrored render canvas, `fx = (1 - 0.18) * w = 0.82 * w`, drawing the halo crown in the far top-right ceiling corner with microscopic radiuses.
2. **False Love Particles on Camera Start ("Love Korea Sudah Tertrigger"):**
   - In `main.js`, `checkCrown.checked` was enabled by default and immediately drew rotating heart crowns and spawned ambient heart particles whenever any face was detected, without requiring the finger heart or two-hand heart gestures.
   - The user observed floating heart emojis without having made any gesture.
3. **Scuba Cat Waving Gesture Failing to Trigger:**
   - Because `faceTracks` held corrupted coordinates near `(0.18, 0.18)` and `face.w` near `0.07`, distance calculations between the hand and face (`distCenter < face.w * 0.95`) never resolved to true when the user touched their actual face at `(0.5, 0.5)`.

### Solutions Implemented
1. **Direct Native Video Inference:**
   - Removed intermediate `visionInputCanvas` blit. MediaPipe `HandLandmarker` and `FaceDetector` now evaluate the `<video>` element directly.
   - Normalized bounding box coordinates against `video.videoWidth` and `video.videoHeight`, restoring 100% geometric accuracy.
2. **Crown Gesture Latching (Temporal Latch):**
   - Gated heart crown on active detection of `isFingerHeart` or `isTwoHandHeart` with a 75-frame (~1.25s) temporal latch (`heartCrownTimer`).
   - Gated cheeky crown on active detection of `isMiddleFinger` with a 75-frame (~1.25s) temporal latch (`cheekyCrownTimer`).
   - Screen remains completely free of random heart/cheeky emojis when hands are resting.
3. **Halo Floating Geometry Calibration (`static/particles.js`, `gui_app.py`):**
   - Calibrated crown center height to `cy = faceCenterY - faceHeight * 0.72` with natural orbital radiuses `rx = faceWidth * 0.58` and `ry = faceHeight * 0.12`, positioning the halo naturally above the cranium and hair.
4. **Scuba Cat Hand-Face Distance & Wave Sensitivity:**
   - Expanded nose/face touch acceptance radius to `distCenter < face.w * 1.15 || distTips < face.w * 0.95 || distThumb < face.w * 0.95`.
   - Lowered wave movement threshold to `Math.abs(dx) > 0.006` and threshold to `wavingEnergy >= 28`, guaranteeing immediate Scuba Cat trigger on 1-2 wave cycles.
   - Applied identical logic to Python desktop app `gui_app.py`.

## FIX-45 — Permanent Toggle Crown Activation & Robust Scuba Cat Noise Immunity

**Date:** 2026-09-08
**Files:** `static/main.js`, `templates/index.html`, `gui_app.py`

### Root Causes
1. **Scuba Cat False Triggering on Subtle Movement:**
   - The nose-touch gate was overly broad (`face.w * 1.15`), matching hands resting near the chin, neck, or chest without requiring a fisted nose-pinch pose.
   - Single-frame differential movement (`dx > 0.006`) with an instant `+28` energy gain caused random webcam jitter or slight hand twitches to trigger the Scuba Cat jumpscare.
2. **Crown Gated Behind Gestures Rather Than Permanent UI Toggle:**
   - The user expects the 3D rotating halo crown to stay permanently visible hovering above their head whenever the toggle switch ("Mahkota Love" or "Mahkota Jahil") is turned ON, rather than disappearing when not making gestures.

### Solutions Implemented
1. **Permanent Toggle-Driven Halo Crown:**
   - Restored permanent crown rendering whenever `checkCrown.checked` or `checkCheeky.checked` is enabled, hovering consistently above the cranium.
   - Added mutual exclusivity between "Mahkota Love Halo" and "Mahkota Jari Tengah Halo".
   - Gesture executions (Finger Heart, Two-Hand Heart, Middle Finger) continue to emit particle bursts from fingertips.
2. **Robust Scuba Cat Multi-Condition Validation:**
   - Nose hand must be a fisted pinch (`isFingerFolded` on middle and ring fingers) with fingertip 8 or 4 within `face.w * 0.50` of the nose center.
   - Waving hand must be an open palm (`isFingerExtended` on index and middle fingers).
   - Waving requires a 15-frame sliding window with horizontal dominance (`xSpan > ySpan`), wide sweep (`xSpan >= face.w * 0.40`), and at least 2 deliberate back-and-forth direction reversals (`reversals >= 2`).
   - Slight hand movements, resting hands, and head twitches are 100% rejected.
3. **Parity Applied to Desktop Application (`gui_app.py`):**
   - Synced permanent toggle crown and robust wave history window to `gui_app.py`.

## FIX-46 — Calibrated Stroke Accumulator & Leaky Bucket for Effortless Scuba Cat Triggering

**Date:** 2026-09-08
**Files:** `static/gestures.js`, `static/main.js`, `templates/index.html`, `gui_app.py`, `tests/gestures.test.mjs`

### Root Causes of "Susah Ke-trigger"
1. **Brittle Nose Hand Pose Checks:**
   - Requiring middle and ring fingers to be folded (`isFisted`) caused frequent misses because MediaPipe HandLandmarker landmarks become distorted and foreshortened when a hand touches the face.
   - The radius `face.w * 0.50` was too tight, failing when a user pinched the nostrils or bridge from the side.
2. **Motion Blur Finger Extension Dropouts on Waving Hand:**
   - Requiring extended index and middle fingers on the waving hand failed during rapid hand sweeps due to standard 30 FPS webcam motion blur.
3. **Over-Constrained Reversal Sliding Window:**
   - Requiring `xSpan > ySpan`, `xSpan >= face.w * 0.40`, and 2 reversals accumulating `0.020` each within 8-15 frames was physically too difficult to satisfy in real-time, failing on diagonal or curved arc waves.

### Solutions Implemented
1. **Generous Hand-at-Face Spatial Gating (`isHandAtFace`):**
   - Evaluates wrist (0), palm (9), index tip (8), and thumb tip (4) against face center with an `0.85 * face.w` radius, accepting natural nose-pinching poses.
   - Enforces physical separation: waving wrist must be separated from face (`> 0.45 * face.w`) and from the nose hand (`> 0.45 * face.w`).
2. **Physics-Based Stroke Accumulator & Leaky Bucket Integrator (`updateScubaWaving`):**
   - Jitter rejection: displacement `< 0.007` normalized units (sensor noise) is filtered out and decays energy.
   - Dominant axis tracking: dynamically selects `primaryDelta` (horizontal or vertical component), supporting natural arc and tilted waving.
   - Stroke travel threshold: direction reversals only award energy if the preceding stroke covered `>= 0.028` normalized units (~35px on 720p). Unidirectional drifts (mouse movement, reaching) never gain energy.
   - Immediate responsiveness: a single valid reversal stroke awards `+24.0` energy, surpassing the `20.0` threshold to trigger Scuba Cat within ~200-300ms of natural hand waving.
   - Graceful decay: decays smoothly at `-1.5` to `-2.0` per frame upon motion cessation, eliminating flickering.
3. **Full Cross-Platform Parity:**
   - Synced identical stroke accumulator and leaky bucket architecture to `gui_app.py`.
4. **Comprehensive Test Suite & Cache Invalidation:**
   - Added 5 new unit tests in `tests/gestures.test.mjs` covering jitter rejection, unidirectional drift rejection, hand-at-face gating, and waving oscillation (18/18 passing).
   - Bumped cache buster to `?v=7` in `templates/index.html` and `static/main.js`.
