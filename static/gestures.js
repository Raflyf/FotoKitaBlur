// High-Precision Mathematical Gesture Recognition Engine
// Normalized 2D hand landmarks: 21 x { x, y } in [0, 1].

export function getDistance(p1, p2) {
    return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

export function ccw(A, B, C) {
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

/**
 * Checks if a non-thumb finger (Index, Middle, Ring, Pinky) is extended.
 * Uses both wrist-relative and MCP-relative Euclidean distances to remain
 * robust against hand tilt and perspective foreshortening.
 */
export function isFingerExtended(landmarks, mcpIdx, pipIdx, tipIdx) {
    const wrist = landmarks[0];
    const mcp = landmarks[mcpIdx];
    const pip = landmarks[pipIdx];
    const tip = landmarks[tipIdx];

    const distTipWrist = getDistance(tip, wrist);
    const distPipWrist = getDistance(pip, wrist);
    const distTipMcp = getDistance(tip, mcp);
    const distPipMcp = getDistance(pip, mcp);

    // Tip must be extended outward from MCP and farther than PIP from wrist
    const extendedFromMcp = distTipMcp > distPipMcp * 1.15;
    const extendedFromWrist = distTipWrist > distPipWrist * 1.08;

    return extendedFromMcp && extendedFromWrist;
}

/**
 * Checks if a non-thumb finger is folded or curled into the palm.
 */
export function isFingerFolded(landmarks, mcpIdx, pipIdx, tipIdx) {
    const wrist = landmarks[0];
    const mcp = landmarks[mcpIdx];
    const pip = landmarks[pipIdx];
    const tip = landmarks[tipIdx];

    const distTipWrist = getDistance(tip, wrist);
    const distPipWrist = getDistance(pip, wrist);
    const distTipMcp = getDistance(tip, mcp);
    const distPipMcp = getDistance(pip, mcp);

    // Folded if tip is closer to wrist than PIP or curled tightly toward MCP
    return (distTipWrist <= distPipWrist * 1.06) || (distTipMcp <= distPipMcp * 1.25);
}

/**
 * Strict Peace Sign (✌️):
 * Index and Middle fingers extended, Ring and Pinky folded,
 * fingers separated, thumb not extended as a third finger.
 */
export function isPeace(landmarks) {
    const wrist = landmarks[0];
    const palmSize = getDistance(landmarks[0], landmarks[9]);
    if (palmSize < 0.015) return false;

    // Strict guard: Middle finger gesture can NEVER be peace
    if (isMiddleFinger(landmarks)) return false;

    const indexUp = isFingerExtended(landmarks, 5, 6, 8);
    const middleUp = isFingerExtended(landmarks, 9, 10, 12);
    const ringFolded = isFingerFolded(landmarks, 13, 14, 16);
    const pinkyFolded = isFingerFolded(landmarks, 17, 18, 20);

    if (!indexUp || !middleUp || !ringFolded || !pinkyFolded) return false;

    // Both index and middle reach from wrist must be genuinely extended
    const indexReach = getDistance(landmarks[8], wrist);
    const middleReach = getDistance(landmarks[12], wrist);
    if (indexReach < palmSize * 0.80 || middleReach < palmSize * 0.80) return false;

    // In a peace sign, middle finger reach cannot dwarf index finger reach
    const reachRatio = middleReach / Math.max(0.001, indexReach);
    if (reachRatio > 1.25 || reachRatio < 0.75) return false;

    // Index tip must be significantly extended from its MCP knuckle
    if (getDistance(landmarks[8], landmarks[5]) < palmSize * 0.50) return false;

    // Fingers must have angular or spatial separation
    const fingerSeparation = getDistance(landmarks[8], landmarks[12]);
    if (fingerSeparation < palmSize * 0.18) return false;

    // Thumb must not be extended outward as in a three-finger sign
    const thumbDist = getDistance(landmarks[4], wrist);
    if (thumbDist > palmSize * 1.45 && thumbDist > indexReach * 0.9) {
        return false;
    }

    return true;
}

/**
 * Middle Finger (🖕):
 * Middle finger extended, Index, Ring, and Pinky folded.
 */
export function isMiddleFinger(landmarks) {
    const wrist = landmarks[0];
    const palmSize = getDistance(landmarks[0], landmarks[9]);
    if (palmSize < 0.015) return false;

    const middleUp = isFingerExtended(landmarks, 9, 10, 12);
    const indexFolded = isFingerFolded(landmarks, 5, 6, 8);
    const ringFolded = isFingerFolded(landmarks, 13, 14, 16);
    const pinkyFolded = isFingerFolded(landmarks, 17, 18, 20);

    // Middle tip must reach higher than folded index tip
    const middleReach = getDistance(landmarks[12], wrist);
    const indexReach = getDistance(landmarks[8], wrist);

    return middleUp && indexFolded && ringFolded && pinkyFolded && (middleReach > indexReach * 1.15);
}

/**
 * Korean Finger Heart (🫰):
 * The thumb shaft (IP->Tip) and index shaft (PIP->Tip) visually cross forming an X,
 * while Middle, Ring, and Pinky are folded into the palm.
 */
export function isFingerHeart(landmarks) {
    const wrist = landmarks[0];
    const palmSize = getDistance(landmarks[0], landmarks[9]);
    if (palmSize < 0.015) return false;

    // Middle, Ring, Pinky must be folded into palm
    const middleFolded = isFingerFolded(landmarks, 9, 10, 12);
    const ringFolded = isFingerFolded(landmarks, 13, 14, 16);
    const pinkyFolded = isFingerFolded(landmarks, 17, 18, 20);

    if (!middleFolded || !ringFolded || !pinkyFolded) return false;

    // Index must reach forward beyond the folded middle finger
    const indexDist = getDistance(landmarks[8], wrist);
    const middleDist = getDistance(landmarks[12], wrist);
    if (middleDist < 0.01 || (indexDist / middleDist) < 1.12) return false;

    // Thumb must sit above the index PIP (not tucked into palm)
    if (landmarks[4].y > landmarks[6].y + palmSize * 0.15) return false;

    const a = landmarks[3];
    const b = landmarks[4];
    const c = landmarks[6];
    const d = landmarks[8];

    const segmentsCross = (ccw(a, b, c) !== ccw(a, b, d)) && (ccw(c, d, a) !== ccw(c, d, b));
    return segmentsCross;
}

/**
 * Two-Hand Heart (🫶):
 * Both hands meet: index tips meet at top apex, thumb tips meet at bottom.
 */
export function isTwoHandHeart(l1, l2) {
    if (!l1 || !l2) return null;

    const palm1 = getDistance(l1[0], l1[9]);
    const palm2 = getDistance(l2[0], l2[9]);
    const avgPalm = (palm1 + palm2) / 2;
    if (avgPalm < 0.015) return null;

    const distIndexTips = getDistance(l1[8], l2[8]);
    const distThumbTips = getDistance(l1[4], l2[4]);

    // Both index tips and thumb tips must meet
    if (distIndexTips < avgPalm * 0.65 && distThumbTips < avgPalm * 0.65) {
        const indexY = (l1[8].y + l2[8].y) / 2;
        const thumbY = (l1[4].y + l2[4].y) / 2;

        if (indexY <= thumbY + avgPalm * 0.25) {
            return {
                x: (l1[8].x + l2[8].x + l1[4].x + l2[4].x) / 4,
                y: (l1[8].y + l2[8].y + l1[4].y + l2[4].y) / 4
            };
        }
    }

    return null;
}

/**
 * Identifies if a hand is holding/touching the face/nose area.
 * Checks wrist, palm center, index tip, and thumb tip against face center.
 */
export function isHandAtFace(handLandmarks, face) {
    if (!handLandmarks || !face) return false;
    const d0 = Math.hypot(handLandmarks[0].x - face.x, handLandmarks[0].y - face.y);
    const d9 = Math.hypot(handLandmarks[9].x - face.x, handLandmarks[9].y - face.y);
    const d8 = Math.hypot(handLandmarks[8].x - face.x, handLandmarks[8].y - face.y);
    const d4 = Math.hypot(handLandmarks[4].x - face.x, handLandmarks[4].y - face.y);
    return Math.min(d0, d9, d8, d4) < face.w * 0.85;
}

/**
 * Calibrated Scuba Cat Waving Motion Integrator.
 * Filters camera jitter and unidirectional drift, triggering on intentional back-and-forth oscillation.
 */
export function updateScubaWaving(state, waveWrist, face, noseWrist) {
    if (!state) {
        state = {
            energy: 0,
            lastX: null,
            lastY: null,
            strokeDir: 0,
            strokeDist: 0
        };
    }

    // Separation checks: waving hand is away from face and away from nose hand
    const distWaveFace = Math.hypot(waveWrist.x - face.x, waveWrist.y - face.y);
    const distBetweenHands = Math.hypot(waveWrist.x - noseWrist.x, waveWrist.y - noseWrist.y);

    if (distWaveFace <= face.w * 0.45 || distBetweenHands <= face.w * 0.45) {
        state.energy = Math.max(0, state.energy - 2.0);
        state.lastX = null;
        state.strokeDist = 0;
        state.strokeDir = 0;
        return { isWaving: state.energy >= 20, state };
    }

    if (state.lastX === null) {
        state.lastX = waveWrist.x;
        state.lastY = waveWrist.y;
    }

    const dx = waveWrist.x - state.lastX;
    const dy = waveWrist.y - state.lastY;
    state.lastX = waveWrist.x;
    state.lastY = waveWrist.y;

    const dist = Math.hypot(dx, dy);

    // Deadband: webcam micro-jitter (<0.007 normalized units)
    if (dist < 0.007) {
        state.energy = Math.max(0, state.energy - 1.5);
        state.strokeDist *= 0.85;
    } else {
        // Dominant axis displacement (handles horizontal and slight arc/diagonal wave)
        const primaryDelta = Math.abs(dx) >= Math.abs(dy) ? dx : dy;
        const dir = primaryDelta > 0.004 ? 1 : (primaryDelta < -0.004 ? -1 : 0);
        if (dir !== 0) {
            if (state.strokeDir === 0) {
                state.strokeDir = dir;
                state.strokeDist = Math.abs(primaryDelta);
            } else if (dir === state.strokeDir) {
                state.strokeDist += Math.abs(primaryDelta);
            } else {
                // Direction reversed! Check if previous stroke traveled a meaningful distance
                const minStroke = Math.min(face.w * 0.15, 0.028);
                if (state.strokeDist >= minStroke) {
                    state.energy = Math.min(100, state.energy + 24.0);
                }
                state.strokeDir = dir;
                state.strokeDist = Math.abs(primaryDelta);
            }
        }
    }

    return { isWaving: state.energy >= 20, state };
}