// Node tests for the browser gesture predicates. Mirrors tests/test_blur.py
// so the frontend logic (the real product) is covered, not just the CLI duplicate.
// Run: node --test tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { isPeace, isMiddleFinger, isFingerHeart } from "../static/gestures.js";

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
test("degenerate palm rejected as peace", () => {
    assert.equal(isPeace(Array.from({ length: 21 }, () => ({ x: 0, y: 0 }))), false);
});
test("middle finger detected", () => assert.equal(isMiddleFinger(middleFingerHand()), true));
test("fist not middle finger", () => assert.equal(isMiddleFinger(fistHand()), false));
test("pinch pose not finger heart", () => assert.equal(isFingerHeart(pinchHand()), false));
test("crossed finger heart detected", () => assert.equal(isFingerHeart(crossedFingerHeartHand()), true));
test("peace not finger heart", () => assert.equal(isFingerHeart(peaceHand()), false));