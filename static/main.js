import { FilesetResolver, HandLandmarker, FaceDetector } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { getDistance, isPeace, isMiddleFinger, isFingerHeart } from "./gestures.js";

let handLandmarker;
let faceDetector;
let isModelLoaded = false;
let isCameraActive = false;
let localStream = null;
let animationFrameId = null;

// Face & Crown tracking state
let crownAngle = 0;
let lastHeartState = false;
let lastCheekyState = false;

let lastFaceResults = null;
let faceDetectCounter = 0;
let detectionTimeoutId = null;

// FIX-04: per-face tracking with stable IDs and EMA smoothing
let faceTracks = [];        // [{ id, headX, headY, headWidth, headHeight, isCheeky, missedFrames }]
let nextFaceId = 0;
const TRACK_KEEP_FRAMES = 4;   // keep a track alive for this many missed detections before retiring
const FACE_EMA_ALPHA = 0.4;    // new-observation weight: pos = pos + (new - pos) * ALPHA

// FIX-06 & FIX-12: cache last two detection snapshots to interpolate between detections
let prevSnapshot = null;
let currSnapshot = null;       // { time, crowns, spawns, cheekySpawns }

// FIX-02: temporal hysteresis (latch) for the peace/blur gesture
const PEACE_HOLD_DURATION = 4; // hold blur for N extra detection frames after last positive detection
let peaceHoldTimer = 0;

// FIX-16: framerate-independent crown rotation
let lastCrownTime = performance.now();
const CROWN_ANGULAR_SPEED = 2.4; // radians per second (== 0.04/frame at 60fps)

// Skeleton drawing & scubacat gesture states
let showSkeleton = false;
let lastHandLandmarks = [];
let scubacatHoldTimer = 0;
const SCUBACAT_HOLD_DURATION = 8;
let wavingActivity = 0;

// FIX-05: lightweight hand tracker (nearest-wrist match) for stable hand identity
let prevHands = [];          // [{ id, x, y }]
let nextHandId = 0;
const HAND_MATCH_GATE = 0.3; // normalized max distance to re-identify a hand between frames

// FIX-08: per-hand waving history for oscillation (fanning/flapping) detection.
// Uses window RANGE (total side-to-side travel) + hysteresis-based reversals, which
// robustly captures a fast fanning motion while rejecting MediaPipe wrist jitter.
// The old instantaneous-displacement approach failed at wave turning points where the
// hand momentarily slows, so the gesture never triggered.
let wavingHistories = new Map(); // handId -> { xs: [wristX,...], ys: [wristY,...] }
const WAVING_WINDOW = 12;        // frames of wrist history to keep (~0.8s at 15fps)
const WAVING_RANGE_MIN = 0.15;   // min normalized (by palmSize) side-to-side range over the window
const WAVING_STEP_MIN = 0.02;    // min normalized step to confirm a direction change (rejects jitter)
const WAVING_REVERSALS_MIN = 2;  // min robust direction reversals (1 full back-and-forth cycle)
const WAVING_CHARGE = 25;        // charge per frame of confirmed waving
const WAVING_DECAY = 5;          // decay per frame of no waving
const WAVING_TRIGGER = 20;       // trigger threshold (a single confirmed frame now triggers)

// FIX-04: face tracking gates
const FACE_MATCH_GATE = 0.4;     // normalized centroid distance to re-identify a face track

// FIX-11: gesture-to-face assignment gate. Nearest-face assignment + one-gesture-per-face
// exclusivity (below) fix the duplicate-spawn issue; the gate just needs to be wide enough
// for the middle-finger/cheeky gesture (hand often away from the face) while not so wide that
// a gesture matches an unrelated face. 0.45 is a middle ground (old value was 0.6).
const GESTURE_FACE_GATE = 0.45;

// FIX-02: whole-frame blur state (hysteresis-latched peace/blur)
let effectivePeace = false;      // hysteresis-latched peace/blur state

const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],         // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],         // Index
    [0, 9], [9, 10], [10, 11], [11, 12],     // Middle
    [0, 13], [13, 14], [14, 15], [15, 16],   // Ring
    [0, 17], [17, 18], [18, 19], [19, 20],   // Pinky
    [5, 9], [9, 13], [13, 17]               // Palm base
];

const video = document.getElementById('webcam');
const canvas = document.getElementById('output-canvas');

// Request high-performance (discrete) GPU for Canvas2D rendering.
// This signals the browser/OS to prefer the dedicated GPU (e.g. RTX 3050) over integrated graphics.
const ctx = canvas.getContext('2d', {
    powerPreference: 'high-performance',
    alpha: false
});

const btnToggle = document.getElementById('btn-toggle');
const statusBadge = document.getElementById('status-badge');
const statusText = document.getElementById('status-text');
const loadingPlaceholder = document.getElementById('loading-placeholder');
const loadingText = document.getElementById('loading-text');

const IS_MOBILE = /Mobi|Android/i.test(navigator.userAgent);

// FIX-17: cache control DOM references once instead of querying every frame
const blurInput = document.getElementById('input-blur');
const blurLabel = document.getElementById('label-blur');
const skeletonCheckbox = document.getElementById('check-skeleton');
const crownCheckbox = document.getElementById('check-crown');
const cheekyCheckbox = document.getElementById('check-cheeky');
const btnGlossary = document.getElementById('btn-glossary');
const glossaryPanel = document.getElementById('glossary-panel');
const btnGlossaryClose = document.getElementById('btn-glossary-close');
const confInput = document.getElementById('input-conf');
const confLabel = document.getElementById('label-conf');
const catVideoEl = document.getElementById('cat-video');
const fingerPanel = document.getElementById('finger-panel');
const btnMusic = document.getElementById('btn-music');

// Finger diagnostics elements are cached once at startup (they are mutated at
// detection rate, so per-call querySelector was a measurable hot spot).
const fingerStatusEls = {};
(function cacheFingerUI() {
    for (const id of ['thumb', 'index', 'middle', 'ring', 'pinky']) {
        fingerStatusEls[id] = document.querySelector(`#finger-${id} .finger-status`);
    }
})();

let lastAppliedPeace = null; // FIX-17: avoid redundant canvas.style.filter writes

// Initialize MediaPipe models
async function initializeModel() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        
        // Initialize Hand Landmarker
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "/static/models/hand_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "video",
            numHands: 6 // Set to 6 hands to support multiple people simultaneously
        });

        // Initialize Face Detector — configured to detect multiple faces simultaneously
        faceDetector = await FaceDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "/static/models/blaze_face_short_range.tflite",
                delegate: "GPU"
            },
            runningMode: "video",
            minDetectionConfidence: 0.4,    // Lower threshold to catch partially visible faces
            minSuppressionThreshold: 0.5,   // FIX-03: higher NMS threshold retains overlapping faces (lower values suppress them)
            numFaces: 6                     // Allow up to 6 simultaneous faces
        });
        
        isModelLoaded = true;
        loadingPlaceholder.style.display = 'none';
        btnToggle.disabled = false;
        btnToggle.textContent = 'Start Camera';
        statusText.textContent = 'Ready';
    } catch (err) {
        console.error("Failed to load MediaPipe models:", err);
        loadingText.textContent = "Failed to load models. Check internet connection.";
    }
}

// Run model initialization
initializeModel();

// GPU Acceleration: Offscreen Canvas Cache for emojis (prevents slow CPU text rasterization drop-frames)
// FIX-22: cache is bounded with an LRU eviction policy and coarser size quantization to limit memory.
const EMOJI_CACHE_MAX = 64;        // maximum number of cached emoji canvases
const EMOJI_SIZE_STEP = 8;         // quantize sizes to nearest 8px to collapse near-duplicate entries
const emojiCache = new Map();      // key -> canvas; insertion order used as LRU (oldest first)
function getEmojiCanvas(emoji, size) {
    // FIX-22: quantize to a coarser step so visually-identical sizes share one canvas
    const roundedSize = Math.max(1, Math.round(size / EMOJI_SIZE_STEP) * EMOJI_SIZE_STEP);
    const key = `${emoji}-${roundedSize}`;
    if (emojiCache.has(key)) {
        // FIX-22: move-to-end to mark as most-recently-used
        const cached = emojiCache.get(key);
        emojiCache.delete(key);
        emojiCache.set(key, cached);
        return cached;
    }
    const offscreen = document.createElement('canvas');
    const pad = Math.ceil(roundedSize * 0.4);
    const width = roundedSize + pad * 2;
    const height = roundedSize + pad * 2;
    offscreen.width = width;
    offscreen.height = height;
    
    const oCtx = offscreen.getContext('2d');
    oCtx.font = `${roundedSize}px sans-serif`;
    oCtx.textBaseline = 'middle';
    oCtx.textAlign = 'center';
    
    // Add subtle drop-shadow glow to hearts
    oCtx.shadowColor = 'rgba(255, 105, 180, 0.8)';
    oCtx.shadowBlur = Math.round(roundedSize * 0.2);
    
    oCtx.fillText(emoji, width / 2, height / 2);
    // FIX-22: enforce LRU cap by evicting the oldest entry (first key in insertion order)
    if (emojiCache.size >= EMOJI_CACHE_MAX) {
        const oldestKey = emojiCache.keys().next().value;
        emojiCache.delete(oldestKey);
    }
    emojiCache.set(key, offscreen);
    return offscreen;
}

class HeartParticle {
    constructor(x, y, emojis = ['❤️', '💖', '💝', '💕', '💗', '💓', '💘']) {
        this.x = x;
        this.y = y;
        this.size = Math.random() * 24 + 20; // Size (20px to 44px)
        this.opacity = 1.0;
        this.vx = (Math.random() - 0.5) * 3; // Float side-to-side
        this.vy = -(Math.random() * 3 + 3);  // Float upwards
        this.emoji = emojis[Math.floor(Math.random() * emojis.length)];
        this.rotation = (Math.random() - 0.5) * 0.4;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.opacity -= 0.015;
    }

    draw(ctx) {
        ctx.save();
        ctx.globalAlpha = this.opacity;
        
        // Draw pre-rendered emoji canvas using GPU-accelerated drawImage
        const img = getEmojiCanvas(this.emoji, this.size);
        ctx.translate(this.x, this.y);
        ctx.rotate(this.rotation);
        ctx.drawImage(img, -img.width / 2, -img.height / 2);
        ctx.restore();
    }
}

let particles = [];
let heartSpawnCooldown = 0;
let cheekySpawnCooldown = 0;
let ambientSpawnCooldown = 0;

// Hysteresis: once triggered, stay active for N extra detection frames to bridge brief drops
const HEART_HOLD_DURATION = 3; // number of frames to hold after last positive detection
// FIX-05: keyed by stable hand-tracking ID (not array index) so timers survive reordering
const heartHoldTimers = new Map(); // key: hand track id, value: frames remaining

// FIX-23: latch for the two-hand classic heart. Hand pairs have no stable ID, so use one
// shared timer + a snapshot of the last positive index-tip positions.
const TWO_HAND_HEART_HOLD = 3;
let twoHandHeartTimer = 0;
let twoHandHeartPos = null; // [{x, y}, {x, y}] index tips at last positive detection

const CHEEKY_HOLD_DURATION = 3;
// FIX-05: keyed by stable hand-tracking ID (not array index)
const cheekyHoldTimers = new Map(); // key: hand track id, value: frames remaining

// FIX-05: lightweight hand tracker. Matches each detected hand to the nearest previous wrist
// (normalized coords) within HAND_MATCH_GATE, reusing its ID; unmatched hands get a new ID.
// Returns an array of { id, landmarks } aligned 1:1 with the input landmarks array.
function trackHands(landmarksList) {
    const used = new Array(prevHands.length).fill(false);
    const assignments = new Array(landmarksList.length).fill(null);

    for (let i = 0; i < landmarksList.length; i++) {
        const wrist = landmarksList[i][0];
        let bestIdx = -1;
        let bestDist = Infinity;
        for (let j = 0; j < prevHands.length; j++) {
            if (used[j]) continue;
            const d = getDistance(wrist, prevHands[j]);
            if (d < bestDist) {
                bestDist = d;
                bestIdx = j;
            }
        }
        if (bestIdx !== -1 && bestDist < HAND_MATCH_GATE) {
            used[bestIdx] = true;
            assignments[i] = prevHands[bestIdx].id;
        }
    }

    const nextPrev = [];
    for (let i = 0; i < landmarksList.length; i++) {
        const wrist = landmarksList[i][0];
        let id = assignments[i];
        if (id === null) {
            id = nextHandId++;
        }
        nextPrev.push({ id, x: wrist.x, y: wrist.y });
        assignments[i] = { id, landmarks: landmarksList[i] };
    }
    prevHands = nextPrev;
    return assignments; // [{ id, landmarks }, ...]
}


// Gesture predicates (isPeace / isMiddleFinger / isFingerHeart / getDistance)
// live in ./gestures.js so the browser and the Node test suite share one source.

// Toggle client-side camera capture
async function toggleCamera() {
    if (!isModelLoaded) return;

    if (isCameraActive) {
        // Stop camera
        stopCamera();
    } else {
        // Start camera
        try {
            btnToggle.disabled = true;
            btnToggle.textContent = 'Initializing Camera...';

            // Show placeholder with starting message
            loadingPlaceholder.style.display = 'flex';
            loadingText.textContent = 'Starting camera...';
            const spinner = loadingPlaceholder.querySelector('.spinner');
            if (spinner) spinner.style.display = 'block';

            // FIX-27: cap native resolution. The canvas is clamped to 800px wide anyway, so
            // requesting 1080p just wastes decode/GPU on the convolutions and MediaPipe runs.
            localStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: "user"
                },
                audio: false
            });

            video.srcObject = localStream;
            video.removeEventListener('loadeddata', startDetectionLoop); // Remove pending listener if user spammed stop/start
            video.addEventListener('loadeddata', startDetectionLoop, { once: true });
            isCameraActive = true;

            // FIX-17: use cached finger-panel reference
            if (fingerPanel) fingerPanel.style.display = 'flex';

            btnToggle.disabled = false;
            btnToggle.textContent = 'Stop Camera';
            btnToggle.className = 'btn btn-danger';
            statusBadge.className = 'status-badge active';
            statusText.textContent = 'Active';
        } catch (err) {
            console.error("Camera access failed:", err);
            alert("Could not access camera. Please allow camera permissions.");
            btnToggle.disabled = false;
            btnToggle.textContent = 'Start Camera';
            loadingPlaceholder.style.display = 'none';
        }
    }
}

function stopCamera() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (detectionTimeoutId) {
        clearTimeout(detectionTimeoutId);
        detectionTimeoutId = null;
    }
    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        localStream = null;
    }
    video.srcObject = null;
    isCameraActive = false;
    isDetecting = false; // Reset lock guard
    video.removeEventListener('loadeddata', startDetectionLoop);

    if (fingerPanel) fingerPanel.style.display = 'none';

    // Reset landmarks, waving activity, and cat-video states
    lastHandLandmarks = [];
    wavingActivity = 0;
    scubacatHoldTimer = 0;

    // FIX-21: reset face detection counter so mobile throttling restarts cleanly
    faceDetectCounter = 0;
    lastFaceResults = null;
    // FIX-05 / FIX-08: reset hand tracker and per-hand waving histories
    prevHands = [];
    wavingHistories.clear();
    // FIX-04: retire all face tracks so stale crowns don't linger after restart
    faceTracks = [];
    // FIX-06 & FIX-12: clear interpolation snapshots
    prevSnapshot = null;
    currSnapshot = null;
    // FIX-02: reset peace hysteresis latch
    peaceHoldTimer = 0;
    // FIX-05: clear gesture hysteresis timers
    heartHoldTimers.clear();
    cheekyHoldTimers.clear();
    // FIX-23: reset two-hand heart latch state
    twoHandHeartTimer = 0;
    twoHandHeartPos = null;
    // FIX-17: reset redundant-write guard
    lastAppliedPeace = null;

    if (catVideoEl) {
        catVideoEl.style.display = 'none';
    }
    if (kicauAudio) {
        // FIX-21: pause without re-seeking to 133 so a later restart resumes from the loop point naturally
        kicauAudio.pause();
    }

    // Clear canvas with black fill and remove CSS blur filter
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    canvas.style.filter = 'none';

    // Restore stopped camera placeholder message and hide spinner
    loadingPlaceholder.style.display = 'flex';
    loadingText.textContent = 'Camera is stopped. Click Start Camera to begin.';
    const spinner = loadingPlaceholder.querySelector('.spinner');
    if (spinner) spinner.style.display = 'none';

    btnToggle.textContent = 'Start Camera';
    btnToggle.className = 'btn';
    statusBadge.className = 'status-badge';
    statusText.textContent = 'Ready';
}

// Update confidence text label smoothly (triggered during drag)
function updateConfidenceLabel() {
    // FIX-17: use cached DOM reference
    const val = confInput.value;
    confLabel.textContent = val + '%';
}

// Apply configuration change to MediaPipe (triggered when slider is released)
function applyConfidenceSetting() {
    // FIX-17: use cached DOM reference
    const val = confInput.value;
    const conf = parseFloat(val) / 100;
    if (handLandmarker) {
        // FIX-14: update ALL hand confidence thresholds, not just detection.
        // Tracking confidence is set slightly lower than detection so a hand already being
        // tracked is kept alive across brief detection dips.
        handLandmarker.setOptions({
            minHandDetectionConfidence: conf,
            minHandPresenceConfidence: conf,
            minTrackingConfidence: Math.max(0, conf - 0.1)
        });
    }
}

// Main frame processing loop
let lastVideoTime = -1;
let isDetecting = false;

// FIX-04: update face tracks from a fresh set of face detections.
// Each detection is matched to the nearest existing track (by centroid distance, gated),
// reusing its stable ID and EMA-smoothing its box. Unmatched detections spawn new tracks.
// Tracks that miss detection for TRACK_KEEP_FRAMES are retired.
function updateFaceTracks(detections, scaleX, scaleY) {
    const used = new Array(faceTracks.length).fill(false);
    const matched = new Array(detections.length).fill(null);

    for (let i = 0; i < detections.length; i++) {
        const bbox = detections[i].boundingBox;
        const cx = (bbox.originX + bbox.width / 2) / video.videoWidth;
        const cy = (bbox.originY + bbox.height / 2) / video.videoHeight;
        let bestIdx = -1;
        let bestDist = Infinity;
        for (let j = 0; j < faceTracks.length; j++) {
            if (used[j]) continue;
            const t = faceTracks[j];
            const d = Math.hypot(t.normX - cx, t.normY - cy);
            if (d < bestDist) {
                bestDist = d;
                bestIdx = j;
            }
        }
        if (bestIdx !== -1 && bestDist < FACE_MATCH_GATE) {
            used[bestIdx] = true;
            matched[i] = bestIdx;
        }
    }

    const nextTracks = [];
    // Carry forward matched tracks with EMA smoothing
    for (let i = 0; i < detections.length; i++) {
        const bbox = detections[i].boundingBox;
        const nx = (bbox.originX + bbox.width / 2) * scaleX;
        const ny = bbox.originY * scaleY;
        const nw = bbox.width * scaleX;
        const nh = bbox.height * scaleY;
        const normX = (bbox.originX + bbox.width / 2) / video.videoWidth;
        const normY = (bbox.originY + bbox.height / 2) / video.videoHeight;
        if (matched[i] !== null) {
            const t = faceTracks[matched[i]];
            // FIX-04: EMA smoothing — pos = pos + (new - pos) * ALPHA
            t.headX = t.headX + (nx - t.headX) * FACE_EMA_ALPHA;
            t.headY = t.headY + (ny - t.headY) * FACE_EMA_ALPHA;
            t.headWidth = t.headWidth + (nw - t.headWidth) * FACE_EMA_ALPHA;
            t.headHeight = t.headHeight + (nh - t.headHeight) * FACE_EMA_ALPHA;
            t.normX = normX;
            t.normY = normY;
            t.missedFrames = 0;
            nextTracks.push(t);
        } else {
            nextTracks.push({
                id: nextFaceId++,
                headX: nx, headY: ny, headWidth: nw, headHeight: nh,
                normX, normY, isCheeky: false, missedFrames: 0
            });
        }
    }
    // Keep unmatched-but-still-alive tracks for a few frames (no EMA update; freeze last pos)
    for (let j = 0; j < faceTracks.length; j++) {
        if (!used[j]) {
            const t = faceTracks[j];
            t.missedFrames++;
            if (t.missedFrames <= TRACK_KEEP_FRAMES) {
                nextTracks.push(t);
            }
        }
    }
    faceTracks = nextTracks;
}

// FIX-06 & FIX-12: linear interpolation between the two most recent detection snapshots.
// Returns interpolated { crowns, spawns, cheekySpawns } for the current render time.
function interpolateSnapshot() {
    if (!currSnapshot) return { crowns: [], spawns: [], cheekySpawns: [] };
    if (!prevSnapshot || prevSnapshot.time === currSnapshot.time) {
        return {
            crowns: currSnapshot.crowns,
            spawns: currSnapshot.spawns,
            cheekySpawns: currSnapshot.cheekySpawns
        };
    }
    const now = performance.now();
    let frac = (now - prevSnapshot.time) / (currSnapshot.time - prevSnapshot.time);
    if (!isFinite(frac) || frac < 0) frac = 0;
    if (frac > 1) frac = 1; // hold the latest position once we've caught up to it

    const lerp = (a, b) => a + (b - a) * frac;
    const lerpCrown = (a, b) => ({
        headX: lerp(a.headX, b.headX),
        headY: lerp(a.headY, b.headY),
        headWidth: lerp(a.headWidth, b.headWidth),
        headHeight: lerp(a.headHeight, b.headHeight),
        isCheeky: b.isCheeky
    });
    const lerpSpawn = (a, b) => ({ x: lerp(a.x, b.x), y: lerp(a.y, b.y) });

    // Match by index (snapshots are built from tracked faces, so order is stable per track id set).
    const crowns = currSnapshot.crowns.map((c, i) => {
        const p = prevSnapshot.crowns[i];
        return p ? lerpCrown(p, c) : c;
    });
    // FIX-27: pair spawns by nearest position instead of array index. Spawn counts change
    // between snapshots (hearts spawn/expire), so index pairing lerped unrelated hearts
    // across the screen. Nearest-match with one-shot use keeps each particle's motion smooth.
    const matchLerp = (currList, prevList) => {
        const used = new Array(prevList.length).fill(false);
        return currList.map(s => {
            let best = -1;
            let bestD = Infinity;
            for (let i = 0; i < prevList.length; i++) {
                if (used[i]) continue;
                const d = Math.hypot(prevList[i].x - s.x, prevList[i].y - s.y);
                if (d < bestD) {
                    bestD = d;
                    best = i;
                }
            }
            if (best !== -1) {
                used[best] = true;
                return lerpSpawn(prevList[best], s);
            }
            return s;
        });
    };
    const spawns = matchLerp(currSnapshot.spawns, prevSnapshot.spawns);
    const cheekySpawns = matchLerp(currSnapshot.cheekySpawns, prevSnapshot.cheekySpawns);
    return { crowns, spawns, cheekySpawns };
}

// 60FPS Continuous Render Loop
function renderLoop() {
    if (!isCameraActive) return;

    // FIX-17: read cached control values once per frame
    const blurVal = blurInput ? blurInput.value : 0;
    if (blurLabel) blurLabel.textContent = blurVal + 'px';
    const showCrown = crownCheckbox ? crownCheckbox.checked : false;
    const showCheeky = cheekyCheckbox ? cheekyCheckbox.checked : false;
    showSkeleton = skeletonCheckbox ? skeletonCheckbox.checked : false;

    // FIX-06 & FIX-12: interpolate overlay positions between detections for smooth motion
    const interp = interpolateSnapshot();
    const interpCrowns = interp.crowns;
    const interpSpawns = interp.spawns;
    const interpCheekySpawns = interp.cheekySpawns;

    // Render video frame
    ctx.save();

    // Flip horizontally for selfie mirroring
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    // FIX-02: whole-canvas CSS blur driven by the hysteresis-latched peace state.
    // FIX-17: only touch canvas.style.filter when the effective state actually changes.
    if (effectivePeace) {
        if (lastAppliedPeace !== true) {
            canvas.style.filter = `blur(${blurVal}px)`;
            lastAppliedPeace = true;
        }
    } else {
        if (lastAppliedPeace !== false) {
            canvas.style.filter = 'none';
            lastAppliedPeace = false;
        }
    }

    // Draw video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Draw hand skeleton if enabled
    if (showSkeleton && lastHandLandmarks.length > 0) {
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';

        for (const hand of lastHandLandmarks) {
            // Draw connections (white lines)
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
            for (const conn of HAND_CONNECTIONS) {
                const p1 = hand[conn[0]];
                const p2 = hand[conn[1]];
                ctx.beginPath();
                ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
                ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
                ctx.stroke();
            }

            // Draw joints (red dots)
            ctx.fillStyle = 'rgba(255, 0, 0, 0.9)';
            for (const lm of hand) {
                ctx.beginPath();
                ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
    }

    ctx.restore();

    // Draw floating heart particles (drawn after restore so they aren't mirrored!)
    particles = particles.filter(p => p.opacity > 0);
    for (const p of particles) {
        p.update();
        p.draw(ctx);
    }

    // Draw rotating crown above each crowned face if either toggle is active
    if ((showCrown || showCheeky) && interpCrowns.length > 0) {
        // FIX-16: framerate-independent rotation using a time delta
        const now = performance.now();
        crownAngle += CROWN_ANGULAR_SPEED * (now - lastCrownTime) / 1000;
        lastCrownTime = now;
        const numHearts = 6;

        for (const crown of interpCrowns) {
            const rx = crown.headWidth * 0.55; // Horizontal radius of the crown
            const ry = crown.headHeight * 0.10; // Vertical radius (flattened for 3D look)
            const cx = canvas.width - crown.headX; // Canvas is mirrored horizontally
            const cy = crown.headY - crown.headHeight * 0.38; // Hover above the head like an angel halo

            // Choose emojis based purely on which toggle switch is active
            let emojis = null;
            if (showCheeky) {
                emojis = ['🖕', '😜', '🤪', '🖕', '😝', '👅'];
            } else if (showCrown) {
                emojis = ['💖', '❤️', '💕', '💗', '💓', '💝'];
            }

            // Skip drawing this face's crown if no active configuration matches
            if (!emojis) continue;

            for (let i = 0; i < numHearts; i++) {
                const angle = crownAngle + (i * 2 * Math.PI / numHearts);
                const hx = cx + Math.cos(angle) * rx;
                const hy = cy + Math.sin(angle) * ry;

                // 3D scaling: Front-facing hearts (sin > 0) are larger, back-facing are smaller
                const depthScale = 0.85 + Math.sin(angle) * 0.25;
                const size = (crown.headWidth * 0.18) * depthScale;

                ctx.save();
                // Draw pre-rendered emoji canvas using GPU-accelerated drawImage
                const img = getEmojiCanvas(emojis[i % emojis.length], size);
                ctx.drawImage(img, hx - img.width / 2, hy - img.height / 2);
                ctx.restore();
            }
        }
    } else {
        // FIX-16: keep the timestamp fresh so the first frame after enabling isn't a huge jump
        lastCrownTime = performance.now();
    }

    // Spawn hearts at hand positions (FIX-06/FIX-12: use interpolated spawn positions)
    if (interpSpawns.length > 0) {
        heartSpawnCooldown--;
        if (heartSpawnCooldown <= 0) {
            for (const spawn of interpSpawns) {
                const cx = (1.0 - spawn.x) * canvas.width;
                const cy = spawn.y * canvas.height;
                particles.push(new HeartParticle(cx, cy));
            }
            heartSpawnCooldown = 5; // Spawn rate at hands
        }
    } else {
        heartSpawnCooldown = 0;
    }

    // Spawn cheeky tongue-out particles at hand positions (FIX-06/FIX-12: interpolated)
    if (interpCheekySpawns.length > 0) {
        cheekySpawnCooldown--;
        if (cheekySpawnCooldown <= 0) {
            for (const spawn of interpCheekySpawns) {
                const cx = (1.0 - spawn.x) * canvas.width;
                const cy = spawn.y * canvas.height;
                particles.push(new HeartParticle(cx, cy, ['🖕', '😜', '🤪', '😝', '👅']));
            }
            cheekySpawnCooldown = 5;
        }
    } else {
        cheekySpawnCooldown = 0;
    }

    // Spawn ambient particles scattered randomly across the entire preview (fewer, slower)
    if (lastCheekyState) {
        ambientSpawnCooldown--;
        if (ambientSpawnCooldown <= 0) {
            const numAmbient = Math.random() < 0.4 ? 2 : 1;
            for (let i = 0; i < numAmbient; i++) {
                const rx = Math.random() * canvas.width;
                const ry = Math.random() * canvas.height * 0.85;
                const p = new HeartParticle(rx, ry, ['🖕', '😜', '🤪', '😝', '👅']);
                p.size = p.size * 0.55;
                p.vy = -(Math.random() * 1.5 + 0.8);
                p.vx = (Math.random() - 0.5) * 1.2;
                p.opacity = 0.7;
                particles.push(p);
            }
            ambientSpawnCooldown = 18;
        }
    } else if (lastHeartState) {
        ambientSpawnCooldown--;
        if (ambientSpawnCooldown <= 0) {
            const numAmbient = Math.random() < 0.4 ? 2 : 1;
            for (let i = 0; i < numAmbient; i++) {
                const rx = Math.random() * canvas.width;
                const ry = Math.random() * canvas.height * 0.85; // Avoid very bottom
                const p = new HeartParticle(rx, ry);
                p.size = p.size * 0.55;  // Smaller than hand hearts
                p.vy = -(Math.random() * 1.5 + 0.8); // Slower float upward
                p.vx = (Math.random() - 0.5) * 1.2;  // Gentle side drift
                p.opacity = 0.7;         // Slightly more transparent from start
                particles.push(p);
            }
            ambientSpawnCooldown = 18;
        }
    } else {
        ambientSpawnCooldown = 0;
    }

    animationFrameId = requestAnimationFrame(renderLoop);
}

// Throttled ML Detection Loop (Runs ~15 times per second to prevent CPU/GPU overload)
async function runDetection() {
    if (!isCameraActive) return;
    if (isDetecting) return; // Skip if previous run is still processing
    if (!handLandmarker || !faceDetector) return; // Models not ready yet

    isDetecting = true;
    try {
        const startTimeMs = performance.now();
        
        // Only run landmarker and update state if we have a new camera frame
        if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
            lastVideoTime = video.currentTime;
            
            const results = handLandmarker.detectForVideo(video, startTimeMs);
            
            // Mobile Optimization: run face detector only every 2nd frame (since faces move slower than hands)
            let faceResults = { detections: [] };
            faceDetectCounter++;
            if (!IS_MOBILE || faceDetectCounter % 2 === 0) {
                faceResults = faceDetector.detectForVideo(video, startTimeMs);
                lastFaceResults = faceResults;
            } else {
                faceResults = lastFaceResults || { detections: [] };
            }

            // Store raw landmarks for real-time skeleton drawing
            lastHandLandmarks = results.landmarks || [];

            let peaceDetected = false;
            const heartGestures = [];   // each entry: { x, y, handId }
            const cheekyGestures = [];  // each entry: { x, y, handId }

            // 1. Diagnose hands and check for gestures
            if (results.landmarks && results.landmarks.length > 0) {
                // FIX-05: assign stable hand IDs via the lightweight wrist tracker so hysteresis
                // timers survive hand reordering between frames.
                const trackedHands = trackHands(results.landmarks);

                // Update diagnostics UI with status of the first hand
                const firstHand = results.landmarks[0];
                const wrist = firstHand[0];
                const palmSize = getDistance(firstHand[0], firstHand[9]); // Wrist to middle MCP joint
                const thumbUp = getDistance(firstHand[4], firstHand[5]) > (palmSize * 0.65);
                const indexUp = getDistance(firstHand[8], wrist) > getDistance(firstHand[6], wrist) * 1.15;
                const middleUp = getDistance(firstHand[12], wrist) > getDistance(firstHand[10], wrist) * 1.1;
                const ringUp = getDistance(firstHand[16], wrist) > getDistance(firstHand[14], wrist) * 1.1;
                const pinkyUp = getDistance(firstHand[20], wrist) > getDistance(firstHand[18], wrist) * 1.1;

                updateFingerUI('thumb', thumbUp);
                updateFingerUI('index', indexUp);
                updateFingerUI('middle', middleUp);
                updateFingerUI('ring', ringUp);
                updateFingerUI('pinky', pinkyUp);

                // Check peace gesture (causes blur)
                for (const landmarks of results.landmarks) {
                    if (isPeace(landmarks)) {
                        peaceDetected = true;
                    }
                }

                // Check one-hand Korean finger heart for all hands — with hysteresis latch
                // FIX-05: keyed by stable hand track id instead of array index
                // FIX-22: collect ids of hands firing a real heart THIS frame so the
                // two-hand classic-heart detector below can avoid double-firing.
                // FIX-23: a peace/V sign must never emit a heart — kill the latch immediately
                // so the emoji disappears the instant the user switches to peace.
                const heartHandIds = new Set();
                for (const th of trackedHands) {
                    const landmarks = th.landmarks;
                    const handId = th.id;
                    if (isPeace(landmarks)) {
                        heartHoldTimers.set(handId, 0);
                        continue; // peace wins over heart for this hand
                    }
                    if (isFingerHeart(landmarks)) {
                        heartHandIds.add(handId);
                        heartHoldTimers.set(handId, HEART_HOLD_DURATION); // Reset latch on positive detection
                    }
                    // Use latch: emit gesture if timer still active
                    const remaining = heartHoldTimers.get(handId) || 0;
                    if (remaining > 0) {
                        heartHoldTimers.set(handId, remaining - 1);
                        const gx = (landmarks[8].x + landmarks[4].x) / 2;
                        const gy = (landmarks[8].y + landmarks[4].y) / 2;
                        heartGestures.push({ x: gx, y: gy, handId });
                    }
                }

                // Check middle finger gesture for all hands — with hysteresis latch
                // FIX-05: keyed by stable hand track id instead of array index
                for (const th of trackedHands) {
                    const landmarks = th.landmarks;
                    const handId = th.id;
                    if (isMiddleFinger(landmarks)) {
                        cheekyHoldTimers.set(handId, CHEEKY_HOLD_DURATION); // Reset latch
                    }
                    const remaining = cheekyHoldTimers.get(handId) || 0;
                    if (remaining > 0) {
                        cheekyHoldTimers.set(handId, remaining - 1);
                        const gx = landmarks[12].x; // Middle finger tip
                        const gy = landmarks[12].y;
                        cheekyGestures.push({ x: gx, y: gy, handId });
                    }
                }

                // Check two-hand classic heart for any pairs of hands
                // FIX-13: normalize the index/thumb tip distances by the average palmSize of the
                // two hands, consistent with the one-hand isFingerHeart approach (raw 0.12 was
                // distance-dependent and failed for hands close to / far from the camera).
                // FIX-22: emit ONE heart per hand (at each index tip) so a two-hand heart
                // produces two emojis; skip pairs where both hands already fired a
                // one-hand finger heart this frame to avoid quadruple-firing.
                // FIX-23: pairs have no stable ID, so keep a shared latch timer; the pair
                // snapshot keeps emitting during brief detection dips. Threshold widened to
                // 0.6 * avgPalm so a slightly-loose heart still registers.
                if (results.landmarks.length >= 2) {
                    pairLoop:
                    for (let i = 0; i < results.landmarks.length; i++) {
                        for (let j = i + 1; j < results.landmarks.length; j++) {
                            const l1 = results.landmarks[i];
                            const l2 = results.landmarks[j];
                            const id1 = trackedHands[i] ? trackedHands[i].id : null;
                            const id2 = trackedHands[j] ? trackedHands[j].id : null;
                            if (id1 !== null && id2 !== null && heartHandIds.has(id1) && heartHandIds.has(id2)) {
                                continue; // both hands already emitted a one-hand heart
                            }
                            const palm1 = getDistance(l1[0], l1[9]);
                            const palm2 = getDistance(l2[0], l2[9]);
                            const avgPalm = (palm1 + palm2) / 2;
                            if (avgPalm < 0.01) continue;
                            const distIndex = getDistance(l1[8], l2[8]);
                            const distThumb = getDistance(l1[4], l2[4]);
                            // FIX-13: 0.6 * avgPalm replaces the old raw 0.12 threshold
                            if (distIndex < 0.6 * avgPalm && distThumb < 0.6 * avgPalm) {
                                twoHandHeartPos = [{ x: l1[8].x, y: l1[8].y }, { x: l2[8].x, y: l2[8].y }];
                                twoHandHeartTimer = TWO_HAND_HEART_HOLD;
                                break pairLoop; // emit from the closest-matching pair only
                            }
                        }
                    }
                }
                // FIX-23: emit during the latch window, but never while one-hand finger hearts
                // are firing this frame (those already produced their own emojis).
                if (twoHandHeartTimer > 0 && heartHandIds.size === 0 && twoHandHeartPos) {
                    heartGestures.push({ x: twoHandHeartPos[0].x, y: twoHandHeartPos[0].y, handId: -1 });
                    heartGestures.push({ x: twoHandHeartPos[1].x, y: twoHandHeartPos[1].y, handId: -2 });
                    twoHandHeartTimer--;
                }
            } else {
                updateFingerUI('thumb', false);
                updateFingerUI('index', false);
                updateFingerUI('middle', false);
                updateFingerUI('ring', false);
                updateFingerUI('pinky', false);
                // FIX-05: no hands this frame — clear the tracker so IDs restart fresh next time
                prevHands = [];
            }

            // Check scubacat gesture: one hand holding the nose and the other hand waving.
            // FIX-07: iterate ALL hands/faces (not just index 0/1) to support multiple people.
            // FIX-08: waving is detected via palmSize-normalized displacement + oscillation
            // (direction reversals) rather than a tiny raw-displacement threshold.
            let scubacatDetected = false;
            let currentFrameWaving = false;

            if (faceResults.detections && faceResults.detections.length > 0 && results.landmarks && results.landmarks.length >= 2) {
                // FIX-07: for each face, find the nose-touching hand; then among the REMAINING
                // hands find a waving one that is close to the SAME face bbox (same person).
                for (const detection of faceResults.detections) {
                    const bbox = detection.boundingBox;
                    const faceCenterX = (bbox.originX + bbox.width / 2) / video.videoWidth;
                    const faceCenterY = (bbox.originY + bbox.height / 2) / video.videoHeight;
                    const faceW = bbox.width / video.videoWidth;
                    const faceH = bbox.height / video.videoHeight;

                    let noseX = null;
                    let noseY = null;
                    if (detection.keypoints && detection.keypoints.length > 2) {
                        noseX = detection.keypoints[2].x;
                        noseY = detection.keypoints[2].y;
                    }
                    if (noseX === null || noseY === null) continue;

                    // 1. Find the hand touching this face's nose (search ALL hands)
                    // FIX-25: the nose-hand is usually a CLOSED FIST covering the nose, so check
                    // the palm center (landmark 9) as well as the fingertips; widen the gate.
                    let noseHandIndex = -1;
                    for (let i = 0; i < results.landmarks.length; i++) {
                        const landmarks = results.landmarks[i];
                        const wrist = landmarks[0];
                        const palmSize = getDistance(landmarks[0], landmarks[9]);
                        if (palmSize < 0.01) continue;
                        const gate = Math.max(0.14, palmSize * 0.7); // scale the touch gate by hand size
                        const indexDist = Math.hypot(landmarks[8].x - noseX, landmarks[8].y - noseY);
                        const thumbDist = Math.hypot(landmarks[4].x - noseX, landmarks[4].y - noseY);
                        const palmDist = Math.hypot(landmarks[9].x - noseX, landmarks[9].y - noseY);
                        const touchesNose = indexDist < gate || thumbDist < gate || palmDist < gate;
                        // FIX-28: the nose hand must be a FIST/pinch (all four fingers folded).
                        // Merely opening both palms near the face must not count as scubacat.
                        // FIX-29: 1.10 tolerance — when a fist covers the nose, MediaPipe joint
                        // positions become unreliable; strict 1.05 would reject genuine fists.
                        const fisted =
                            getDistance(landmarks[8], wrist) < getDistance(landmarks[6], wrist) * 1.10 &&
                            getDistance(landmarks[12], wrist) < getDistance(landmarks[10], wrist) * 1.10 &&
                            getDistance(landmarks[16], wrist) < getDistance(landmarks[14], wrist) * 1.10 &&
                            getDistance(landmarks[20], wrist) < getDistance(landmarks[18], wrist) * 1.10;
                        if (touchesNose && fisted) {
                            noseHandIndex = i;
                            break;
                        }
                    }
                    if (noseHandIndex === -1) continue;

                    // 2. Among the REMAINING hands, find a waving one belonging to the same person
                    //    (wrist within the face bbox, expanded slightly).
                    for (let i = 0; i < results.landmarks.length; i++) {
                        if (i === noseHandIndex) continue;
                        const wavingHand = results.landmarks[i];
                        const wrist = wavingHand[0];
                        const palmSize = getDistance(wavingHand[0], wavingHand[9]);
                        if (palmSize < 0.01) continue;

                        // FIX-28: the waving hand must be an OPEN PALM facing the camera
                        // (index and middle extended, fingers straight). A fist or a loose
                        // bent hand must not drive the wave.
                        const palmOpen =
                            getDistance(wavingHand[8], wrist) > getDistance(wavingHand[6], wrist) * 1.15 &&
                            getDistance(wavingHand[12], wrist) > getDistance(wavingHand[10], wrist) * 1.15;
                        if (!palmOpen) continue;

                        // FIX-07: gate the waving hand to the same face via proximity
                        // FIX-25: widen the gate — a horizontal wrist swipe swings further than
                        // the face bbox while staying on the same person.
                        const inFaceX = Math.abs(wrist.x - faceCenterX) < faceW * 1.6;
                        const inFaceY = Math.abs(wrist.y - faceCenterY) < faceH * 2.0;
                        if (!inFaceX || !inFaceY) continue;

                        // FIX-08: maintain a per-hand wrist history and require oscillation.
                        // Use a stable key from the hand tracker if available, else fall back to index.
                        const waveKey = (prevHands[i] && prevHands[i].id !== undefined) ? prevHands[i].id : i;
                        let hist = wavingHistories.get(waveKey);
                        if (!hist) {
                            hist = { xs: [], ys: [] };
                            wavingHistories.set(waveKey, hist);
                        }

                        // FIX-08: record the wrist into the rolling history window.
                        hist.xs.push(wrist.x);
                        hist.ys.push(wrist.y);
                        if (hist.xs.length > WAVING_WINDOW) {
                            hist.xs.shift();
                            hist.ys.shift();
                        }

                        // Need a full window before evaluating.
                        if (hist.xs.length < WAVING_WINDOW) continue;

                        // FIX-08: window RANGE — total side-to-side travel, normalized by palmSize.
                        // A fanning/waving hand sweeps a wide horizontal arc; a static or jittery
                        // hand stays within a tiny range. This is robust at wave turning points
                        // (where instantaneous velocity is ~0) because it measures total span, not
                        // per-frame speed.
                        let minX = 1, maxX = 0, minY = 1, maxY = 0;
                        for (let k = 0; k < hist.xs.length; k++) {
                            if (hist.xs[k] < minX) minX = hist.xs[k];
                            if (hist.xs[k] > maxX) maxX = hist.xs[k];
                            if (hist.ys[k] < minY) minY = hist.ys[k];
                            if (hist.ys[k] > maxY) maxY = hist.ys[k];
                        }
                        const yRange = (maxY - minY) / palmSize;
                        const range = Math.max(xRange, yRange);

                        // FIX-28: count direction reversals (hysteresis-gated) on the dominant
                        // motion axis. We require >=2 reversals (a back-and-forth turn) so a
                        // single one-way raise does not trigger. The fisted nose-hand + open-palm
                        // wave-hand requirements above are what stop two open palms from firing.
                        let reversals = 0;
                        let confirmedDir = 0;       // last confirmed direction (+1/-1)
                        let sinceChange = 0;        // accumulated travel since last confirmed dir
                        for (let k = 1; k < hist.xs.length; k++) {
                            const dx = hist.xs[k] - hist.xs[k - 1];
                            const dy = hist.ys[k] - hist.ys[k - 1];
                            const step = (xRange >= yRange) ? dx : dy;
                            const dir = step > 0 ? 1 : (step < 0 ? -1 : 0);
                            if (dir === 0) continue;
                            if (confirmedDir === 0) {
                                confirmedDir = dir;
                                sinceChange = Math.abs(step) / palmSize;
                            } else if (dir !== confirmedDir) {
                                sinceChange += Math.abs(step) / palmSize;
                                if (sinceChange >= WAVING_STEP_MIN) {
                                    reversals++;
                                    confirmedDir = dir;
                                    sinceChange = 0;
                                }
                            } else {
                                sinceChange += Math.abs(step) / palmSize;
                            }
                        }

                        // FIX-28: trigger a quick oscillation (>=2 reversals) over a wide span.
                        // Accepts horizontal/vertical/diagonal so the wave is not overly strict;
                        // the fisted nose hand is the anti-false-positive guard.
                        if (range >= WAVING_RANGE_MIN && reversals >= WAVING_REVERSALS_MIN) {
                            currentFrameWaving = true;
                            break; // found a waving hand for this face
                        }
                    }

                    if (currentFrameWaving) break; // a face already triggered waving
                }
            } else {
                // FIX-08: clear histories when there are no faces/hands to wave
                wavingHistories.clear();
            }

            // FIX-08: charge/decay with slower charge and faster decay, raised trigger threshold
            if (currentFrameWaving) {
                wavingActivity = Math.min(100, wavingActivity + WAVING_CHARGE);
            } else {
                wavingActivity = Math.max(0, wavingActivity - WAVING_DECAY);
            }

            // FIX-08: raised trigger threshold
            scubacatDetected = wavingActivity > WAVING_TRIGGER;

            if (scubacatDetected) {
                scubacatHoldTimer = SCUBACAT_HOLD_DURATION;
            } else if (scubacatHoldTimer > 0) {
                scubacatHoldTimer--;
            }

            // Show or hide the picture-in-picture cat video overlay and play/pause kicauAudio (2:13 to 2:26)
            // FIX-17: use cached catVideoEl reference
            if (catVideoEl) {
                if (scubacatHoldTimer > 0) {
                    if (catVideoEl.style.display === 'none') {
                        catVideoEl.style.display = 'block';
                    }
                    if (kicauAudio.paused) {
                        kicauAudio.currentTime = 133; // Seek to 2:13
                        kicauAudio.play().catch(e => console.log("Kicau audio play blocked:", e));
                    }
                } else {
                    if (catVideoEl.style.display !== 'none') {
                        catVideoEl.style.display = 'none';
                    }
                    if (!kicauAudio.paused) {
                        // FIX-21: pause without re-seeking so the loop resumes cleanly on re-trigger
                        kicauAudio.pause();
                    }
                }
            }

            // 2. Process detected faces and match them to gestures
            const scaleX = canvas.width / video.videoWidth;
            const scaleY = canvas.height / video.videoHeight;

            // FIX-04: update persistent face tracks (with stable IDs + EMA smoothing) from the
            // fresh detections. Tracks survive brief detection gaps (TRACK_KEEP_FRAMES).
            if (faceResults.detections && faceResults.detections.length > 0) {
                updateFaceTracks(faceResults.detections, scaleX, scaleY);
            } else {
                // FIX-04: still age existing tracks so they retire after the keep-alive window
                for (const t of faceTracks) t.missedFrames++;
                faceTracks = faceTracks.filter(t => t.missedFrames <= TRACK_KEEP_FRAMES);
            }

            // FIX-11: assign each gesture to its SINGLE nearest face (within a tight gate) instead
            // of spawning particles for every face within 0.6. One-gesture-per-face is enforced by
            // tracking which faces have already consumed a gesture this frame.
            // FIX-22: heart gestures are consumed globally (each heart feeds one face), and each
            // face may take up to 2 hearts — one per hand — so a two-hand classic heart emits
            // two emojis instead of one.
            // FIX-23: same 2-per-face rule for cheeky gestures, so two middle fingers raised
            // by one person emit two emojis (one per hand).
            const newSpawns = [];
            const newCheekySpawns = [];
            const heartConsumed = new Set();      // indices into heartGestures already used
            const cheekyConsumed = new Set();     // indices into cheekyGestures already used

            for (const track of faceTracks) {
                let isFaceCheeky = false;

                // FIX-11 + FIX-23: nearest-face assignment for cheeky gestures
                let cheekyCountForFace = 0;
                for (let gi = 0; gi < cheekyGestures.length && cheekyCountForFace < 2; gi++) {
                    if (cheekyConsumed.has(gi)) continue;
                    const g = cheekyGestures[gi];
                    const dist = Math.hypot(track.normX - g.x, track.normY - g.y);
                    if (dist < GESTURE_FACE_GATE) {
                        isFaceCheeky = true;
                        newCheekySpawns.push({ x: g.x, y: g.y });
                        cheekyConsumed.add(gi);
                        cheekyCountForFace++;
                    }
                }

                // FIX-11 + FIX-22: nearest-face assignment for heart gestures
                let heartCountForFace = 0;
                for (let gi = 0; gi < heartGestures.length && heartCountForFace < 2; gi++) {
                    if (heartConsumed.has(gi)) continue;
                    const g = heartGestures[gi];
                    const dist = Math.hypot(track.normX - g.x, track.normY - g.y);
                    if (dist < GESTURE_FACE_GATE) {
                        newSpawns.push({ x: g.x, y: g.y });
                        heartConsumed.add(gi);
                        heartCountForFace++;
                    }
                }

                // Mark this track's crown type (cheeky if a cheeky gesture matched, else heart)
                track.isCheeky = isFaceCheeky;
            }

            // FIX-04: build the crown list from the (smoothed, tracked) face tracks
            const newCrowns = faceTracks.map(t => ({
                headX: t.headX,
                headY: t.headY,
                headWidth: t.headWidth,
                headHeight: t.headHeight,
                isCheeky: t.isCheeky
            }));

            // FIX-02: temporal hysteresis (latch) for the peace/blur gesture.
            // On positive detection, charge the hold timer; otherwise decay it. The effective
            // blur state stays active while the timer > 0, bridging brief detection drops.
            if (peaceDetected) {
                peaceHoldTimer = PEACE_HOLD_DURATION;
            } else if (peaceHoldTimer > 0) {
                peaceHoldTimer--;
            }
            effectivePeace = peaceHoldTimer > 0;

            // FIX-06 & FIX-12: publish a new interpolation snapshot for this detection.
            // The render loop linearly interpolates between prevSnapshot and currSnapshot.
            const snapshot = {
                time: performance.now(),
                crowns: newCrowns,
                spawns: newSpawns,
                cheekySpawns: newCheekySpawns
            };
            prevSnapshot = currSnapshot;
            currSnapshot = snapshot;

            // Update ambient-particle states (read cached toggles)
            const showCrown = crownCheckbox ? crownCheckbox.checked : false;
            const showCheeky = cheekyCheckbox ? cheekyCheckbox.checked : false;
            lastHeartState = showCrown && (newCrowns.length > 0 && !newCrowns.some(c => c.isCheeky));
            lastCheekyState = showCheeky && newCrowns.some(c => c.isCheeky);
        }
    } catch (err) {
        console.error("ML detection execution failed:", err);
    }
    
    isDetecting = false;
    
    // Schedule next detection (100ms on mobile for lower CPU/heat, 66ms on desktop)
    const delay = IS_MOBILE ? 100 : 66;
    detectionTimeoutId = setTimeout(runDetection, delay);
}

function startDetectionLoop() {
    // Adjust canvas size to match video resolution (capped at max width of 800 for high performance)
    const maxCanvasWidth = 800;
    let targetWidth = video.videoWidth;
    let targetHeight = video.videoHeight;
    if (targetWidth > maxCanvasWidth) {
        const scale = maxCanvasWidth / targetWidth;
        targetWidth = maxCanvasWidth;
        targetHeight = Math.round(targetHeight * scale);
    }

    if (canvas.width !== targetWidth) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        
        // Dynamically adjust viewer-box aspect ratio to match camera feed, preventing mobile landscape cropping!
        const viewerBox = document.getElementById('viewer-box');
        if (viewerBox) {
            viewerBox.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
        }
    }

    // Hide starting placeholder when the camera feed begins rendering
    if (loadingPlaceholder) {
        loadingPlaceholder.style.display = 'none';
    }

    // Start 60fps rendering
    renderLoop();

    // Start throttled AI model runs
    runDetection();
}

// Helper to update diagnostics UI indicators (DOM refs are pre-cached)
function updateFingerUI(id, extended) {
    const el = fingerStatusEls[id];
    if (!el) return;
    if (extended) {
        el.textContent = "Extended";
        el.className = "finger-status extended";
    } else {
        el.textContent = "Folded";
        el.className = "finger-status folded";
    }
}

// Kicau Mania Audio (plays 2:12 to 2:26)
const kicauAudio = new Audio('/kicau');
kicauAudio.preload = 'auto';
kicauAudio.addEventListener('timeupdate', () => {
    if (kicauAudio.currentTime >= 146) {
        kicauAudio.currentTime = 133;
    }
});

// FIX-26: unlock kicau autoplay on the first real user gesture. The scubacat
// audio is played from inside the detection loop (not a direct user gesture),
// so Chrome's autoplay policy blocks it unless the element was already
// "opened" by a gesture. Play-then-pause once on the first interaction.
function unlockKicauAudio() {
    if (!kicauAudio) return;
    const p = kicauAudio.play();
    if (p && p.then) p.then(() => kicauAudio.pause()).catch(() => {});
}
window.addEventListener('pointerdown', unlockKicauAudio, { once: true });
window.addEventListener('touchstart', unlockKicauAudio, { once: true });
window.addEventListener('keydown', unlockKicauAudio, { once: true });

// Native HTML5 Audio Player integration
const audio = document.getElementById('audio-player');
let isMusicPlaying = false;

function toggleMusic() {
    if (!audio) return;
    if (isMusicPlaying) {
        stopMusic();
    } else {
        startMusic();
    }
}

// Monitor time update to stop at exactly 52 seconds (0:52)
audio.addEventListener('timeupdate', () => {
    if (audio.currentTime >= 52) {
        stopMusic();
    }
});

function startMusic() {
    audio.currentTime = 23; // Seek to 23s (0:23)
    audio.play().then(() => {
        isMusicPlaying = true;

        btnMusic.innerHTML = `
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect></svg>
            Stop Music
        `;
        btnMusic.style.background = 'var(--danger-bg)';
        btnMusic.style.color = 'var(--danger)';
        btnMusic.style.borderColor = 'rgba(243, 18, 96, 0.2)';
    }).catch(err => {
        console.error("Audio playback blocked or failed:", err);
        alert("Playback failed. Please click on the page to interact first.");
    });
}

function stopMusic() {
    if (audio) {
        audio.pause();
    }
    isMusicPlaying = false;

    btnMusic.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>
        Play Music (0:23 - 0:52)
    `;
    btnMusic.style.background = 'rgba(255,255,255,0.03)';
    btnMusic.style.color = 'var(--text-primary)';
    btnMusic.style.borderColor = 'var(--border)';
}

// Mutual exclusivity logic for crown toggles (reuses the cached checkbox refs)
if (crownCheckbox && cheekyCheckbox) {
    crownCheckbox.addEventListener('change', () => {
        if (crownCheckbox.checked) {
            cheekyCheckbox.checked = false;
        }
    });

    cheekyCheckbox.addEventListener('change', () => {
        if (cheekyCheckbox.checked) {
            crownCheckbox.checked = false;
        }
    });
}

// Hand Gesture Glossary (Kamus Gestur) toggle. CSP-strict, no inline handlers.
// FIX-24: persistent dictionary of recognized hand gestures in Bahasa Indonesia
// with an on/off toggle; open state survives page reloads via localStorage.
function setGlossaryOpen(open) {
    if (!glossaryPanel || !btnGlossary) return;
    glossaryPanel.hidden = !open;
    btnGlossary.setAttribute('aria-expanded', String(open));
    btnGlossary.classList.toggle('active', open);
    try {
        localStorage.setItem('foto-kita-blur-glossary-open', open ? '1' : '0');
    } catch (e) { /* storage unavailable — non-fatal */ }
}

if (btnGlossary) {
    btnGlossary.addEventListener('click', () => setGlossaryOpen(glossaryPanel.hidden));
}
if (btnGlossaryClose) {
    btnGlossaryClose.addEventListener('click', () => setGlossaryOpen(false));
}
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && glossaryPanel && !glossaryPanel.hidden) {
        setGlossaryOpen(false);
    }
});
// Restore last open state (defensive try/catch for restricted storage contexts)
try {
    if (localStorage.getItem('foto-kita-blur-glossary-open') === '1') {
        setGlossaryOpen(true);
    }
} catch (e) { /* ignore */ }

// Event binding (replaces inline onclick/oninput/onchange handlers so the app
// can ship a strict CSP without 'unsafe-inline' for scripts).
btnToggle.addEventListener('click', toggleCamera);
confInput.addEventListener('input', updateConfidenceLabel);
confInput.addEventListener('change', applyConfidenceSetting);
if (btnMusic) btnMusic.addEventListener('click', toggleMusic);
