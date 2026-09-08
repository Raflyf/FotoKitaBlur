// Node tests for the browser gesture predicates. Mirrors tests/test_blur.py
// so the frontend logic (the real product) is covered, not just the CLI duplicate.
// Run: node --test tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { isPeace, isMiddleFinger, isFingerHeart, isTwoHandHeart, isHandAtFace, updateScubaWaving } from "../static/gestures.js";

// Landmark ids used by gestures.js (MediaPipe hand map)
const THUMB_T = 4, INDEX_PIP = 6, INDEX_T = 8, MIDDLE_MCP = 9,
      MIDDLE_PIP = 10, MIDDLE_T = 12, RING_PIP = 14, RING_T = 16,
      PINKY_PIP = 18, PINKY_T = 20;

function makeHand(points) {
    const landmarks = Array.from({ length: 21 }, () => ({ x: 0, y: 0 }));
    for (const [idx, [x, y]] of Object.entries(points)) {
        landmarks[Number(idx)].x = x;
        landmarks[Number(idx)].y = y;
    }
    return landmarks;
}

// Canonical strict peace sign
function peaceHand() {
    return makeHand({
        [THUMB_T]: [0.05, -0.05], [INDEX_PIP]: [0.10, -0.15], [INDEX_T]: [0.15, -0.60],
        [MIDDLE_MCP]: [0.30, 0.30], [MIDDLE_PIP]: [0.25, -0.15], [MIDDLE_T]: [0.35, -0.65],
        [RING_PIP]: [0.50, 0.10], [RING_T]: [0.35, 0.00],
        [PINKY_PIP]: [0.55, 0.15], [PINKY_T]: [0.40, 0.05],
    });
}

function fistHand() {
    return makeHand({
        [THUMB_T]: [0.05, 0.05], [INDEX_PIP]: [0.10, 0.05], [INDEX_T]: [0.12, 0.10],
        [MIDDLE_MCP]: [0.30, 0.30], [MIDDLE_PIP]: [0.25, 0.10], [MIDDLE_T]: [0.30, 0.12],
        [RING_PIP]: [0.45, 0.10], [RING_T]: [0.47, 0.12],
        [PINKY_PIP]: [0.55, 0.10], [PINKY_T]: [0.58, 0.12],
    });
}

function middleFingerHand() {
    return makeHand({
        [THUMB_T]: [0.05, -0.05], [INDEX_PIP]: [0.10, 0.05], [INDEX_T]: [0.00, 0.05],
        [MIDDLE_MCP]: [0.30, 0.30], [MIDDLE_PIP]: [0.25, -0.10], [MIDDLE_T]: [0.30, -0.60],
        [RING_PIP]: [0.50, 0.10], [RING_T]: [0.40, 0.05],
        [PINKY_PIP]: [0.55, 0.15], [PINKY_T]: [0.45, 0.05],
    });
}

// Pinch: thumb tip and index tip touch but are on the SAME side of the PIP.
// Should NOT trigger a finger heart.
function pinchHand() {
    return makeHand({
        [THUMB_T]: [0.35, -0.35], [INDEX_PIP]: [0.25, -0.10], [INDEX_T]: [0.35, -0.35],
        [MIDDLE_MCP]: [0.30, 0.30], [MIDDLE_PIP]: [0.30, -0.10], [MIDDLE_T]: [0.25, 0.00],
        [RING_PIP]: [0.50, 0.10], [RING_T]: [0.45, 0.05],
        [PINKY_PIP]: [0.55, 0.15], [PINKY_T]: [0.50, 0.10],
        2: [0.05, 0.15], 5: [0.15, 0.10],
    });
}

// True crossed heart: thumb shaft and index shaft CROSS (thumb tip on right of
// PIP, index tip on left). Tips do not need to touch.
function crossedFingerHeartHand() {
    // Anti-parallel shafts forming an X:
    //   thumb: lower-left (0.05,-0.10) -> upper-right (0.40,-0.45)
    //   index: upper-right (0.40,-0.30) -> lower-left (0.05,-0.55)
    return makeHand({
        3: [0.05, -0.10], [THUMB_T]: [0.40, -0.45],
        [INDEX_PIP]: [0.40, -0.30], 7: [0.22, -0.42], [INDEX_T]: [0.05, -0.55],
        [MIDDLE_MCP]: [0.30, 0.30], [MIDDLE_PIP]: [0.30, -0.10], [MIDDLE_T]: [0.25, 0.00],
        [RING_PIP]: [0.50, 0.10], [RING_T]: [0.45, 0.05],
        [PINKY_PIP]: [0.55, 0.15], [PINKY_T]: [0.50, 0.10],
    });
}

test("strict peace detected", () => assert.equal(isPeace(peaceHand()), true));
test("fist rejected as peace", () => assert.equal(isPeace(fistHand()), false));
test("middle finger rejected as peace", () => assert.equal(isPeace(middleFingerHand()), false));
test("middle finger with loose index rejected as peace", () => {
    const hand = middleFingerHand();
    hand[8] = { x: 0.10, y: -0.30 }; // Index tip reaching half height of middle finger
    assert.equal(isPeace(hand), false);
});
test("degenerate palm rejected as peace", () => {
    assert.equal(isPeace(Array.from({ length: 21 }, () => ({ x: 0, y: 0 }))), false);
});
test("middle finger detected", () => assert.equal(isMiddleFinger(middleFingerHand()), true));
test("fist not middle finger", () => assert.equal(isMiddleFinger(fistHand()), false));
test("pinch pose not finger heart", () => assert.equal(isFingerHeart(pinchHand()), false));
test("crossed finger heart detected", () => assert.equal(isFingerHeart(crossedFingerHeartHand()), true));

// Regression: user reported a near-touch ("hanya menempel sedikit") still
// triggering heart. Two fingertips within a few pixels but segments parallel
// (or only brushing) must NOT trigger heart.
function nearTouchHand() {
    return makeHand({
        3: [0.20, -0.30], [THUMB_T]: [0.30, -0.40],
        [INDEX_PIP]: [0.20, -0.30], 7: [0.25, -0.35], [INDEX_T]: [0.31, -0.40],
        [MIDDLE_MCP]: [0.30, 0.30], [MIDDLE_PIP]: [0.30, -0.10], [MIDDLE_T]: [0.25, 0.00],
        [RING_PIP]: [0.50, 0.10], [RING_T]: [0.45, 0.05],
        [PINKY_PIP]: [0.55, 0.15], [PINKY_T]: [0.50, 0.10],
    });
}
test("near touch not finger heart", () => assert.equal(isFingerHeart(nearTouchHand()), false));
test("peace not finger heart", () => assert.equal(isFingerHeart(peaceHand()), false));

test("two hand heart detected when index and thumb tips meet", () => {
    const leftHand = makeHand({
        [MIDDLE_MCP]: [0.20, 0.30],
        8: [0.48, -0.40], // Left index tip
        4: [0.48, -0.20], // Left thumb tip
    });
    const rightHand = makeHand({
        [MIDDLE_MCP]: [0.80, 0.30],
        8: [0.52, -0.40], // Right index tip
        4: [0.52, -0.20], // Right thumb tip
    });
    const result = isTwoHandHeart(leftHand, rightHand);
    assert.ok(result !== null);
    assert.ok(typeof result.x === "number");
});

test("two hand heart rejected when tips are far apart", () => {
    const leftHand = makeHand({
        [MIDDLE_MCP]: [0.20, 0.30],
        8: [0.10, -0.40],
        4: [0.10, -0.20],
    });
    const rightHand = makeHand({
        [MIDDLE_MCP]: [0.80, 0.30],
        8: [0.90, -0.40],
        4: [0.90, -0.20],
    });
    assert.equal(isTwoHandHeart(leftHand, rightHand), null);
});

test("isHandAtFace detects hand near face center", () => {
    const hand = makeHand({
        0: [0.50, 0.48], // Wrist at face
        8: [0.50, 0.45], // Index tip at nose
    });
    const face = { x: 0.50, y: 0.45, w: 0.25, h: 0.25 };
    assert.equal(isHandAtFace(hand, face), true);
});

test("isHandAtFace rejects hand far away from face", () => {
    const hand = makeHand({
        0: [0.10, 0.80],
        8: [0.10, 0.70],
    });
    const face = { x: 0.50, y: 0.45, w: 0.25, h: 0.25 };
    assert.equal(isHandAtFace(hand, face), false);
});

test("updateScubaWaving rejects stationary camera jitter", () => {
    let state = null;
    const face = { x: 0.50, y: 0.45, w: 0.25, h: 0.25 };
    const noseWrist = { x: 0.50, y: 0.45 };
    let triggered = false;

    for (let f = 0; f < 30; f++) {
        const jitter = (Math.random() - 0.5) * 0.006;
        const waveWrist = { x: 0.75 + jitter, y: 0.50 + jitter };
        const res = updateScubaWaving(state, waveWrist, face, noseWrist);
        state = res.state;
        if (res.isWaving) triggered = true;
    }
    assert.equal(triggered, false);
});

test("updateScubaWaving rejects unidirectional drift", () => {
    let state = null;
    const face = { x: 0.50, y: 0.45, w: 0.25, h: 0.25 };
    const noseWrist = { x: 0.50, y: 0.45 };
    let triggered = false;

    for (let f = 0; f < 30; f++) {
        const waveWrist = { x: 0.65 + f * 0.008, y: 0.50 };
        const res = updateScubaWaving(state, waveWrist, face, noseWrist);
        state = res.state;
        if (res.isWaving) triggered = true;
    }
    assert.equal(triggered, false);
});

test("updateScubaWaving triggers reliably on back-and-forth waving", () => {
    let state = null;
    const face = { x: 0.50, y: 0.45, w: 0.25, h: 0.25 };
    const noseWrist = { x: 0.50, y: 0.45 };
    let triggeredCount = 0;

    for (let f = 0; f < 30; f++) {
        const waveWrist = {
            x: 0.75 + 0.06 * Math.sin((f / 10) * 2 * Math.PI),
            y: 0.50 + 0.01 * Math.cos((f / 10) * 2 * Math.PI)
        };
        const res = updateScubaWaving(state, waveWrist, face, noseWrist);
        state = res.state;
        if (res.isWaving) triggeredCount++;
    }
    assert.ok(triggeredCount >= 15, `Expected >= 15 triggered frames, got ${triggeredCount}`);
});