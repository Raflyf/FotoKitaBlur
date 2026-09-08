// Foto Kita Blur - Real-time AI Vision & Gesture Processing Engine
import { FilesetResolver, HandLandmarker, FaceDetector } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { getDistance, isPeace, isMiddleFinger, isFingerHeart, isTwoHandHeart, isFingerExtended } from "./gestures.js?v=3";
import { Particle, draw3DCrown } from "./particles.js?v=3";

// Global Vision Models & State
let handLandmarker = null;
let faceDetector = null;
let isModelReady = false;
let isCameraActive = false;
let mediaStream = null;
let animFrameId = null;

// Audio Controllers
const musicAudio = document.getElementById('audio-player');
const kicauAudio = document.getElementById('kicau-player');
let isMusicPlaying = false;

// DOM Elements Cache
const video = document.getElementById('webcam');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d', { powerPreference: 'high-performance', alpha: false });

const btnToggle = document.getElementById('btn-toggle');
const btnMusic = document.getElementById('btn-music');
const btnGlossary = document.getElementById('btn-glossary');
const btnGlossaryClose = document.getElementById('btn-glossary-close');
const glossaryPanel = document.getElementById('glossary-panel');

const statusBadge = document.getElementById('status-badge');
const statusText = document.getElementById('status-text');
const loadingPlaceholder = document.getElementById('loading-placeholder');
const loadingText = document.getElementById('loading-text');

const hudStatus = document.getElementById('hud-status');
const hudFps = document.getElementById('hud-fps');
const catVideoEl = document.getElementById('cat-video');

const blurInput = document.getElementById('input-blur');
const blurLabel = document.getElementById('label-blur');
const confInput = document.getElementById('input-conf');
const confLabel = document.getElementById('label-conf');

const checkSkeleton = document.getElementById('check-skeleton');
const checkCrown = document.getElementById('check-crown');
const checkCheeky = document.getElementById('check-cheeky');

const fingerEls = {
    thumb: document.getElementById('finger-thumb')?.querySelector('.finger-status'),
    index: document.getElementById('finger-index')?.querySelector('.finger-status'),
    middle: document.getElementById('finger-middle')?.querySelector('.finger-status'),
    ring: document.getElementById('finger-ring')?.querySelector('.finger-status'),
    pinky: document.getElementById('finger-pinky')?.querySelector('.finger-status'),
};

// Hand Connections for Skeleton Rendering
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],         // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],         // Index
    [0, 9], [9, 10], [10, 11], [11, 12],     // Middle
    [0, 13], [13, 14], [14, 15], [15, 16],   // Ring
    [0, 17], [17, 18], [18, 19], [19, 20],   // Pinky
    [5, 9], [9, 13], [13, 17]               // Palm base
];

// Persistent Face Tracking
let faceTracks = []; // [{ id, x, y, w, h, missedFrames }]
let nextFaceId = 0;
const FACE_EMA = 0.35; // Smoothing factor

// Particles & Animations
let activeParticles = [];
let crownAngle = 0;
let lastCrownTime = performance.now();
const CROWN_ANGULAR_SPEED = 2.4; // radians per sec

// Gesture Hold & Hysteresis Timers (Anti-Flicker / Anti-Miss at 60 FPS)
const HOLD_FRAMES = 16; // ~270ms hold on drop at 60 FPS
let peaceHoldTimer = 0;
let scubacatHoldTimer = 0;
let lastAppliedBlur = false;

// Scuba Cat / Waving Integrator
let wavingEnergy = 0;
let handMovementHistories = new Map(); // handId -> { lastX, lastDir, reversals, sinceChange }

// Real-time 60 FPS Render & Throttled Vision State
let fpsCounter = 0;
let lastFpsTime = performance.now();
let displayFps = 0;

// High-Speed Downsampled Offscreen Vision Canvas
const visionInputCanvas = document.createElement('canvas');
visionInputCanvas.width = 480;
visionInputCanvas.height = 270;
const visionInputCtx = visionInputCanvas.getContext('2d', { willReadFrequently: true });

let isDetecting = false;
let nextDetectTime = 0;
let faceInferenceCounter = 0;
const VISION_MIN_GAP_MS = 45; // Enforce minimum 45ms gap between vision runs (~20 FPS detection rate)
const FACE_STRIDE = 3;         // BlazeFace runs every 3rd vision tick (~7-10 FPS)

let lastHeartParticleTime = 0;
let lastCheekyParticleTime = 0;

let latestHandResults = null;
let latestFaceResults = null;
let latestGestures = {
    peace: false,
    fingerHeart: false,
    twoHandHeart: false,
    middleFinger: false,
    scubacat: false,
    heartSpawns: [],
    cheekySpawns: []
};

// ==========================================
// 1. INITIALIZE AI MODELS
// ==========================================
async function initVisionModels() {
    try {
        if (loadingText) loadingText.textContent = "Mengunduh MediaPipe Tasks Vision...";
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");

        if (loadingText) loadingText.textContent = "Memuat Model Hand Landmarker & Face Detector...";
        
        const confVal = (parseInt(confInput.value) || 50) / 100;

        // Load Hand Landmarker with GPU acceleration
        try {
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "/static/models/hand_landmarker.task",
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                numHands: 2,
                minHandDetectionConfidence: confVal,
                minTrackingConfidence: confVal
            });
        } catch (gpuErr) {
            console.warn("GPU delegate unavailable for HandLandmarker, falling back to CPU:", gpuErr);
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "/static/models/hand_landmarker.task",
                    delegate: "CPU"
                },
                runningMode: "VIDEO",
                numHands: 2,
                minHandDetectionConfidence: confVal,
                minTrackingConfidence: confVal
            });
        }

        // Load Face Detector with GPU acceleration
        try {
            faceDetector = await FaceDetector.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "/static/models/blaze_face_short_range.tflite",
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                minDetectionConfidence: 0.45
            });
        } catch (gpuErr) {
            console.warn("GPU delegate unavailable for FaceDetector, falling back to CPU:", gpuErr);
            faceDetector = await FaceDetector.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "/static/models/blaze_face_short_range.tflite",
                    delegate: "CPU"
                },
                runningMode: "VIDEO",
                minDetectionConfidence: 0.45
            });
        }

        isModelReady = true;
        if (loadingPlaceholder) loadingPlaceholder.style.display = 'none';

        if (statusBadge && statusText) {
            statusBadge.className = 'status-badge ready';
            statusText.textContent = 'Model Siap';
        }

        btnToggle.disabled = false;
        btnToggle.textContent = 'Nyalakan Kamera';
        btnToggle.className = 'btn btn-flex btn-primary';

    } catch (err) {
        console.error("Gagal memuat model vision:", err);
        if (loadingText) loadingText.textContent = "Gagal memuat model AI. Periksa koneksi internet.";
        if (statusText) statusText.textContent = "Error AI";
    }
}

// ==========================================
// 2. CAMERA MANAGEMENT
// ==========================================
async function toggleCamera() {
    if (isCameraActive) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    if (!isModelReady) return;

    btnToggle.disabled = true;
    btnToggle.textContent = 'Menghubungkan Kamera...';

    try {
        let stream = null;
        let lastErr = null;

        // Multi-tier constraint fallback: 720p ideal -> 480p ideal -> raw video:true
        const constraintTiers = [
            { video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
            { video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
            { video: true, audio: false }
        ];

        for (const constraints of constraintTiers) {
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                if (stream) break;
            } catch (tierErr) {
                lastErr = tierErr;
                console.warn("Retrying with next camera constraint tier:", tierErr);
            }
        }

        if (!stream) {
            throw lastErr || new Error("Tidak dapat mengakses aliran kamera.");
        }

        mediaStream = stream;
        video.srcObject = stream;

        // Safely wait for video metadata with timeout guard
        await new Promise((resolve) => {
            if (video.videoWidth > 0 && video.videoHeight > 0) {
                resolve();
                return;
            }
            const onLoaded = () => {
                video.removeEventListener('loadedmetadata', onLoaded);
                resolve();
            };
            video.addEventListener('loadedmetadata', onLoaded);
            setTimeout(resolve, 1000);
        });

        await video.play().catch(e => console.warn("video.play warning:", e));

        // Set matching lightweight resolution for high-performance canvas
        let vw = video.videoWidth || 640;
        let vh = video.videoHeight || 480;
        const targetWidth = 640;
        const scale = Math.min(1.0, targetWidth / vw);
        canvas.width = Math.round(vw * scale);
        canvas.height = Math.round(vh * scale);

        isCameraActive = true;
        btnToggle.disabled = false;
        btnToggle.textContent = 'Matikan Kamera';
        btnToggle.className = 'btn btn-flex btn-primary btn-active-danger';

        if (statusBadge && statusText) {
            statusBadge.className = 'status-badge active';
            statusText.textContent = 'Kamera Aktif';
        }
        if (hudStatus) hudStatus.textContent = 'Memindai Gestur...';

        // Start High-FPS Synchronized Render Loop
        lastCrownTime = performance.now();
        lastFpsTime = performance.now();
        nextDetectTime = 0;
        faceInferenceCounter = 0;
        fpsCounter = 0;
        animFrameId = requestAnimationFrame(processNextFrame);

    } catch (err) {
        console.error("Gagal membuka kamera:", err);

        let userMsg = "Tidak dapat mengakses kamera: " + (err.message || err.name);
        if (err.name === "NotReadableError" || err.name === "TrackStartError") {
            userMsg = "Kamera sedang digunakan oleh proses lain (misal aplikasi GUI / Zoom / Discord / tab lain). Mohon tutup aplikasi tersebut lalu coba lagi.";
        } else if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            userMsg = "Izin kamera ditolak di browser. Pastikan izin kamera telah diberikan di ikon gembok pada URL bar.";
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
            userMsg = "Perangkat kamera tidak ditemukan. Pastikan webcam terhubung.";
        }

        alert(userMsg);
        btnToggle.disabled = false;
        btnToggle.textContent = 'Nyalakan Kamera';
        btnToggle.className = 'btn btn-flex btn-primary';
    }
}

function stopCamera() {
    if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    video.srcObject = null;
    isCameraActive = false;
    isDetecting = false;
    latestHandResults = null;
    latestFaceResults = null;
    faceTracks = [];
    activeParticles = [];

    // Reset Scuba Cat & Audio
    if (catVideoEl) catVideoEl.style.display = 'none';
    if (kicauAudio && !kicauAudio.paused) kicauAudio.pause();
    scubacatHoldTimer = 0;
    wavingEnergy = 0;
    peaceHoldTimer = 0;

    // Clear Canvas to Deep Black
    ctx.save();
    ctx.fillStyle = '#050507';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    canvas.style.filter = 'none';
    lastAppliedBlur = false;
    ctx.restore();

    btnToggle.textContent = 'Nyalakan Kamera';
    btnToggle.className = 'btn btn-flex btn-primary';
    if (statusBadge && statusText) {
        statusBadge.className = 'status-badge ready';
        statusText.textContent = 'Kamera Dimatikan';
    }
    if (hudStatus) hudStatus.textContent = 'Kamera Nonaktif';
    if (hudFps) hudFps.textContent = 'FPS: 0';

    resetFingerUI();
}

// ==========================================
// 3. SYNCHRONIZED VISION & RENDER LOOP
// ==========================================
function processNextFrame(now = performance.now()) {
    if (!isCameraActive) return;

    // Calculate real-time 60 FPS metric
    fpsCounter++;
    if (now - lastFpsTime >= 1000) {
        displayFps = Math.round((fpsCounter * 1000) / (now - lastFpsTime));
        fpsCounter = 0;
        lastFpsTime = now;
        if (hudFps) hudFps.textContent = `FPS: ${displayFps}`;
    }

    // Trigger AI inference only when the non-blocking inter-frame gap has elapsed
    if (video.readyState >= 2 && !isDetecting && now >= nextDetectTime) {
        runVisionInference(now);
    }

    // Render Canvas & Effects at silky smooth 60 FPS
    renderScene(latestHandResults, latestGestures, now);

    animFrameId = requestAnimationFrame(processNextFrame);
}

function runVisionInference(now) {
    if (!handLandmarker || isDetecting) return;
    isDetecting = true;

    try {
        // Fast hardware downscale to 480x270 offscreen canvas (dramatically lowers TFLite/WASM load)
        visionInputCtx.drawImage(video, 0, 0, visionInputCanvas.width, visionInputCanvas.height);

        // Run Hand Landmarker on downsampled input
        const handResults = handLandmarker.detectForVideo(visionInputCanvas, now);
        latestHandResults = handResults;

        // Run Face Detector interleaved (every 3rd vision tick = ~7-10 FPS)
        faceInferenceCounter++;
        const shouldDetectFace = (faceTracks.length === 0) || (faceInferenceCounter % FACE_STRIDE === 0);
        if (faceDetector && shouldDetectFace) {
            const faceResults = faceDetector.detectForVideo(visionInputCanvas, now);
            latestFaceResults = faceResults;
            updateFaceTracks(faceResults);
        }

        // Process Hands & Gestures
        latestGestures = processHandGestures(handResults, now);
    } catch (inferErr) {
        console.warn("Detection frame skipped:", inferErr);
    } finally {
        isDetecting = false;
        // Guarantee that the next inference never starves the 60 FPS render loop
        nextDetectTime = performance.now() + VISION_MIN_GAP_MS;
    }
}

// ==========================================
// 4. FACE TRACKING (EMA & OCCLUSION COASTING)
// ==========================================
function updateFaceTracks(faceResults) {
    const rawFaces = [];
    if (faceResults && faceResults.detections) {
        for (const d of faceResults.detections) {
            const b = d.boundingBox;
            rawFaces.push({
                x: (b.originX + b.width / 2) / video.videoWidth,
                y: (b.originY + b.height / 2) / video.videoHeight,
                w: b.width / video.videoWidth,
                h: b.height / video.videoHeight
            });
        }
    }

    // Match raw faces to existing face tracks
    const updated = [];
    const unmatchedRaw = [...rawFaces];

    for (const track of faceTracks) {
        let bestDist = 0.35;
        let bestIdx = -1;

        for (let i = 0; i < unmatchedRaw.length; i++) {
            const raw = unmatchedRaw[i];
            const dist = Math.hypot(track.x - raw.x, track.y - raw.y);
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = i;
            }
        }

        if (bestIdx !== -1) {
            const matched = unmatchedRaw.splice(bestIdx, 1)[0];
            // Exponential Moving Average (EMA) for butter-smooth tracking
            track.x = track.x + (matched.x - track.x) * FACE_EMA;
            track.y = track.y + (matched.y - track.y) * FACE_EMA;
            track.w = track.w + (matched.w - track.w) * FACE_EMA;
            track.h = track.h + (matched.h - track.h) * FACE_EMA;
            track.missedFrames = 0;
            updated.push(track);
        } else {
            // Coast through temporary occlusion (e.g. hand on nose)
            track.missedFrames++;
            if (track.missedFrames <= 6) {
                updated.push(track);
            }
        }
    }

    // Add remaining new faces
    for (const raw of unmatchedRaw) {
        updated.push({
            id: nextFaceId++,
            x: raw.x,
            y: raw.y,
            w: raw.w,
            h: raw.h,
            missedFrames: 0
        });
    }

    faceTracks = updated;
}

// ==========================================
// 5. GESTURE RECOGNITION ENGINE
// ==========================================
function processHandGestures(handResults, now) {
    const gestures = {
        peace: false,
        fingerHeart: false,
        twoHandHeart: false,
        middleFinger: false,
        scubacat: false,
        heartSpawns: [],
        cheekySpawns: []
    };

    const hands = handResults?.landmarks || [];
    if (hands.length === 0) {
        resetFingerUI();
        wavingEnergy = Math.max(0, wavingEnergy - 4);
        return gestures;
    }

    // Update Diagnostics for First Hand
    const h0 = hands[0];
    const wrist = h0[0];
    const palm = getDistance(h0[0], h0[9]);

    updateFingerUI('thumb', getDistance(h0[4], h0[5]) > palm * 0.60);
    updateFingerUI('index', isFingerExtended(h0, 5, 6, 8));
    updateFingerUI('middle', isFingerExtended(h0, 9, 10, 12));
    updateFingerUI('ring', isFingerExtended(h0, 13, 14, 16));
    updateFingerUI('pinky', isFingerExtended(h0, 17, 18, 20));

    // A. Check Middle Finger (🖕) FIRST across all hands
    let middleFingerActive = false;
    for (const hand of hands) {
        if (isMiddleFinger(hand)) {
            middleFingerActive = true;
            gestures.middleFinger = true;
            gestures.cheekySpawns.push({ x: hand[12].x, y: hand[12].y });
        }
    }

    // B. Check Korean Finger Heart (🫰)
    for (const hand of hands) {
        if (isFingerHeart(hand)) {
            gestures.fingerHeart = true;
            const spawnX = (hand[8].x + hand[4].x) / 2;
            const spawnY = (hand[8].y + hand[4].y) / 2;
            gestures.heartSpawns.push({ x: spawnX, y: spawnY });
        }
    }

    // C. Check Peace Sign (✌️) ONLY IF NO HAND is showing middle finger!
    if (!middleFingerActive) {
        for (const hand of hands) {
            if (isPeace(hand)) {
                gestures.peace = true;
                break;
            }
        }
    }

    // D. Check Two-Hand Heart (🫶)
    if (hands.length >= 2) {
        const heartCenter = isTwoHandHeart(hands[0], hands[1]);
        if (heartCenter) {
            gestures.twoHandHeart = true;
            gestures.heartSpawns.push(heartCenter);
        }
    }

    // E. Check Scuba Cat / Waving Gestures
    // Hand 1 must be touching nose/face, Hand 2 must be waving
    if (faceTracks.length > 0 && hands.length >= 2) {
        let noseHandIdx = -1;

        // Find hand touching face/nose zone
        for (let i = 0; i < hands.length; i++) {
            const h = hands[i];
            for (const face of faceTracks) {
                // Hand wrist or tips near face
                const distCenter = Math.hypot(h[0].x - face.x, h[0].y - face.y);
                const distTips = Math.hypot(h[8].x - face.x, h[8].y - face.y);
                if (distCenter < face.w * 0.95 || distTips < face.w * 0.75) {
                    noseHandIdx = i;
                    break;
                }
            }
            if (noseHandIdx !== -1) break;
        }

        if (noseHandIdx !== -1) {
            // Check remaining hand for waving
            const waveHandIdx = noseHandIdx === 0 ? 1 : 0;
            const waveHand = hands[waveHandIdx];
            const waveWrist = waveHand[0];

            let hist = handMovementHistories.get('wavingHand');
            if (!hist) {
                hist = { lastX: waveWrist.x, lastDir: 0, reversals: 0 };
                handMovementHistories.set('wavingHand', hist);
            }

            const dx = waveWrist.x - hist.lastX;
            hist.lastX = waveWrist.x;

            if (Math.abs(dx) > 0.008) {
                const dir = dx > 0 ? 1 : -1;
                if (hist.lastDir !== 0 && dir !== hist.lastDir) {
                    hist.reversals++;
                    wavingEnergy = Math.min(100, wavingEnergy + 24); // Fast charge
                }
                hist.lastDir = dir;
            } else {
                wavingEnergy = Math.max(0, wavingEnergy - 1.5);
            }

            if (wavingEnergy >= 40) {
                gestures.scubacat = true;
            }
        } else {
            wavingEnergy = Math.max(0, wavingEnergy - 3);
        }
    } else {
        wavingEnergy = Math.max(0, wavingEnergy - 3);
    }

    return gestures;
}

// ==========================================
// 6. SCENE RENDERING & EFFECT PIPELINE
// ==========================================
function renderScene(handResults, gestures, now) {
    const w = canvas.width;
    const h = canvas.height;

    // A. Peace Gesture Hysteresis (Blurring)
    if (gestures.peace) {
        peaceHoldTimer = HOLD_FRAMES;
    } else if (peaceHoldTimer > 0) {
        peaceHoldTimer--;
    }

    const isBlurActive = peaceHoldTimer > 0;
    const blurPx = parseInt(blurInput.value) || 25;

    if (isBlurActive !== lastAppliedBlur) {
        canvas.style.filter = isBlurActive ? `blur(${blurPx}px)` : 'none';
        lastAppliedBlur = isBlurActive;
    }

    // B. Draw Mirrored Webcam Video
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);

    // C. Draw Hand Skeleton if Enabled
    if (checkSkeleton.checked && handResults?.landmarks) {
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';

        for (const hand of handResults.landmarks) {
            // Bones (cyan glow)
            ctx.strokeStyle = 'rgba(59, 130, 246, 0.85)';
            for (const [p1, p2] of HAND_CONNECTIONS) {
                ctx.beginPath();
                ctx.moveTo(hand[p1].x * w, hand[p1].y * h);
                ctx.lineTo(hand[p2].x * w, hand[p2].y * h);
                ctx.stroke();
            }

            // Joints (white dots)
            ctx.fillStyle = '#ffffff';
            for (const lm of hand) {
                ctx.beginPath();
                ctx.arc(lm.x * w, lm.y * h, 3.5, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }

    ctx.restore(); // Restore unmirrored coordinate space for text & particles!

    // D. Spawn Fingertip Particles (rate-limited burst so emojis disperse gracefully)
    if (gestures.heartSpawns && gestures.heartSpawns.length > 0) {
        if (now - lastHeartParticleTime >= 120) {
            lastHeartParticleTime = now;
            for (const pt of gestures.heartSpawns) {
                if (activeParticles.length < 20) {
                    const sx = (1.0 - pt.x) * w;
                    const sy = pt.y * h;
                    activeParticles.push(new Particle(sx, sy, ['💖', '❤️', '💕', '💗', '💓', '💝']));
                }
            }
        }
        gestures.heartSpawns = [];
    }

    if (gestures.cheekySpawns && gestures.cheekySpawns.length > 0) {
        if (now - lastCheekyParticleTime >= 120) {
            lastCheekyParticleTime = now;
            for (const pt of gestures.cheekySpawns) {
                if (activeParticles.length < 20) {
                    const sx = (1.0 - pt.x) * w;
                    const sy = pt.y * h;
                    activeParticles.push(new Particle(sx, sy, ['🖕', '😜', '🤪', '😝', '👅']));
                }
            }
        }
        gestures.cheekySpawns = [];
    }

    // E. Draw 3D Halo Crowns (Manual Toggles)
    const showHeartCrown = checkCrown.checked;
    const showCheekyCrown = checkCheeky.checked;

    if ((showHeartCrown || showCheekyCrown) && faceTracks.length > 0) {
        const deltaSec = (now - lastCrownTime) / 1000;
        crownAngle += CROWN_ANGULAR_SPEED * deltaSec;
        lastCrownTime = now;

        const crownEmojis = showCheekyCrown 
            ? ['🖕', '😜', '🤪', '🖕', '😝', '👅'] 
            : ['💖', '❤️', '💕', '💗', '💓', '💝'];

        for (const face of faceTracks) {
            const fx = (1.0 - face.x) * w; // Canvas horizontal flip compensation
            const fy = face.y * h;
            const fw = face.w * w;
            const fh = face.h * h;
            draw3DCrown(ctx, fx, fy, fw, fh, crownAngle, crownEmojis);
        }

        // Ambient particles (lightweight spawn rate with particle cap)
        if (Math.random() < 0.08 && activeParticles.length < 24) {
            const rx = Math.random() * w;
            const ry = Math.random() * h * 0.7;
            activeParticles.push(new Particle(rx, ry, crownEmojis));
        }
    } else {
        lastCrownTime = now;
    }

    // F. Update & Render Floating Particles
    activeParticles = activeParticles.filter(p => p.opacity > 0);
    if (activeParticles.length > 30) {
        activeParticles.splice(0, activeParticles.length - 30);
    }
    for (const p of activeParticles) {
        p.update();
        p.draw(ctx);
    }

    // G. Scuba Cat & Kicau Audio Management
    if (gestures.scubacat) {
        scubacatHoldTimer = 30; // ~500ms latch at 60 FPS
    } else if (scubacatHoldTimer > 0) {
        scubacatHoldTimer--;
    }

    if (scubacatHoldTimer > 0) {
        if (catVideoEl && catVideoEl.style.display !== 'block') {
            catVideoEl.style.display = 'block';
        }
        if (kicauAudio && kicauAudio.paused) {
            kicauAudio.currentTime = 133; // Seek to 2:13
            kicauAudio.play().catch(e => console.warn("Kicau audio blocked:", e));
        }
    } else {
        if (catVideoEl && catVideoEl.style.display !== 'none') {
            catVideoEl.style.display = 'none';
        }
        if (kicauAudio && !kicauAudio.paused) {
            kicauAudio.pause();
        }
    }

    // H. Update HUD Badge Indicator
    updateHudStatus(isBlurActive, gestures, scubacatHoldTimer > 0);
}

function updateHudStatus(isBlurActive, gestures, isScubaActive) {
    if (!hudStatus) return;

    if (isScubaActive) {
        hudStatus.className = 'hud-pill active-cheeky';
        hudStatus.textContent = '🐱 KICAU SCUBA CAT ACTIVE';
    } else if (isBlurActive) {
        hudStatus.className = 'hud-pill active-peace';
        hudStatus.textContent = '✌️ PEACE (BLUR ACTIVE)';
    } else if (gestures.twoHandHeart) {
        hudStatus.className = 'hud-pill active-heart';
        hudStatus.textContent = '🫶 TWO-HAND HEART';
    } else if (gestures.fingerHeart) {
        hudStatus.className = 'hud-pill active-heart';
        hudStatus.textContent = '🫰 FINGER HEART';
    } else if (gestures.middleFinger) {
        hudStatus.className = 'hud-pill active-cheeky';
        hudStatus.textContent = '🖕 JARI TENGAH';
    } else {
        hudStatus.className = 'hud-pill';
        hudStatus.textContent = 'Memindai Gestur...';
    }
}

// ==========================================
// 7. UI LISTENERS & CONTROL BINDINGS
// ==========================================
function updateFingerUI(finger, isUp) {
    const el = fingerEls[finger];
    if (!el) return;
    if (isUp) {
        el.className = 'finger-status up';
        el.textContent = 'UP';
    } else {
        el.className = 'finger-status folded';
        el.textContent = 'FOLDED';
    }
}

function resetFingerUI() {
    for (const key of Object.keys(fingerEls)) {
        updateFingerUI(key, false);
    }
}

// Blur Intensity Slider
blurInput.addEventListener('input', () => {
    blurLabel.textContent = `${blurInput.value}px`;
    if (lastAppliedBlur) {
        canvas.style.filter = `blur(${blurInput.value}px)`;
    }
});

// Detection Confidence Slider
confInput.addEventListener('input', () => {
    confLabel.textContent = `${confInput.value}%`;
});

confInput.addEventListener('change', async () => {
    if (handLandmarker && isModelReady) {
        const val = parseInt(confInput.value) / 100;
        await handLandmarker.setOptions({
            minHandDetectionConfidence: val,
            minTrackingConfidence: val
        });
    }
});

// Mutually Exclusive Crown Toggles
checkCrown.addEventListener('change', () => {
    if (checkCrown.checked) {
        checkCheeky.checked = false;
    }
});

checkCheeky.addEventListener('change', () => {
    if (checkCheeky.checked) {
        checkCrown.checked = false;
    }
});

// Background Music ("Foto Kita Blur" 0:23 - 0:52)
btnMusic.addEventListener('click', toggleMusic);

function toggleMusic() {
    if (!musicAudio) return;

    if (isMusicPlaying) {
        musicAudio.pause();
        isMusicPlaying = false;
        btnMusic.className = 'btn btn-flex btn-secondary';
        btnMusic.innerHTML = `
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>
            Musik (0:23 - 0:52)
        `;
    } else {
        musicAudio.currentTime = 23; // Start at 0:23
        musicAudio.play().then(() => {
            isMusicPlaying = true;
            btnMusic.className = 'btn btn-flex btn-secondary btn-music-playing';
            btnMusic.innerHTML = `
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect></svg>
                Stop Musik
            `;
        }).catch(err => {
            console.warn("Audio autoplay blocked:", err);
            alert("Klik layar sekali terlebih dahulu untuk mengizinkan pemutaran audio.");
        });
    }
}

// Loop music between 23s and 52s
musicAudio.addEventListener('timeupdate', () => {
    if (isMusicPlaying && musicAudio.currentTime >= 52) {
        musicAudio.currentTime = 23;
    }
});

// Loop kicau audio between 133s and 146s
kicauAudio.addEventListener('timeupdate', () => {
    if (kicauAudio.currentTime >= 146) {
        kicauAudio.currentTime = 133;
    }
});

// Kamus Gestur Toggle
btnGlossary.addEventListener('click', () => {
    const isHidden = glossaryPanel.hasAttribute('hidden');
    if (isHidden) {
        glossaryPanel.removeAttribute('hidden');
        btnGlossary.setAttribute('aria-expanded', 'true');
    } else {
        glossaryPanel.setAttribute('hidden', '');
        btnGlossary.setAttribute('aria-expanded', 'false');
    }
});

btnGlossaryClose.addEventListener('click', () => {
    glossaryPanel.setAttribute('hidden', '');
    btnGlossary.setAttribute('aria-expanded', 'false');
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !glossaryPanel.hasAttribute('hidden')) {
        glossaryPanel.setAttribute('hidden', '');
        btnGlossary.setAttribute('aria-expanded', 'false');
    }
});

btnToggle.addEventListener('click', toggleCamera);

// Startup Initialization
window.addEventListener('DOMContentLoaded', () => {
    initVisionModels();
});
