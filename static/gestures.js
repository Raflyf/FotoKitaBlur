// Pure gesture predicates, shared by main.js and the Node test suite.
// Normalized 2D hand landmarks: 21 x { x, y } in [0,1].

export function getDistance(p1, p2) {
    return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

function ccw(A, B, C) {
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

export function isPeace(landmarks) {
    const wrist = landmarks[0];
    const palmSize = getDistance(landmarks[0], landmarks[9]);
    if (palmSize < 0.01) return false;

    const indexUp  = getDistance(landmarks[8], wrist) > getDistance(landmarks[6], wrist) * 1.15;
    const middleUp = getDistance(landmarks[12], wrist) > getDistance(landmarks[10], wrist) * 1.15;
    const ringFolded  = getDistance(landmarks[16], wrist) < getDistance(landmarks[14], wrist) * 0.85;
    const pinkyFolded = getDistance(landmarks[20], wrist) < getDistance(landmarks[18], wrist) * 0.85;
    const fingersSpread = getDistance(landmarks[8], landmarks[12]) > palmSize * 0.32;
    const thumbFolded = getDistance(landmarks[4], wrist) < palmSize * 1.1;

    return indexUp && middleUp && ringFolded && pinkyFolded && fingersSpread && thumbFolded;
}

export function isMiddleFinger(landmarks) {
    const wrist = landmarks[0];
    const palmSize = getDistance(landmarks[0], landmarks[9]);
    if (palmSize < 0.01) return false;

    const middleUp = getDistance(landmarks[12], wrist) > getDistance(landmarks[10], wrist) * 1.12;
    const indexFolded = getDistance(landmarks[8], wrist) < getDistance(landmarks[6], wrist) * 1.05;
    const ringFolded  = getDistance(landmarks[16], wrist) < getDistance(landmarks[14], wrist) * 1.05;
    const pinkyFolded = getDistance(landmarks[20], wrist) < getDistance(landmarks[18], wrist) * 1.05;

    return middleUp && indexFolded && ringFolded && pinkyFolded;
}

/**
 * Korean finger heart: index extended beyond middle, middle/ring/pinky folded,
 * and the thumb shaft visually crosses the index shaft in the 2D projection.
 * Both a pinch (touching tips, same side) and a pistol (no crossing) fail the
 * segment-intersection gate, while a real crossed heart passes.
 */
export function isFingerHeart(landmarks) {
    const wrist = landmarks[0];
    const palmSize = getDistance(landmarks[0], landmarks[9]);
    if (palmSize < 0.01) return false;

    // Index must reach well beyond the (folded) middle finger.
    const indexDist = getDistance(landmarks[8], wrist);
    const middleDist = getDistance(landmarks[12], wrist);
    if (middleDist < 0.01 || indexDist / middleDist < 1.15) return false;

    // Middle/ring/pinky must be folded.
    const middleFolded = getDistance(landmarks[12], wrist) < getDistance(landmarks[10], wrist) * 1.65;
    const ringFolded   = getDistance(landmarks[16], wrist) < getDistance(landmarks[14], wrist) * 1.65;
    const pinkyFolded  = getDistance(landmarks[20], wrist) < getDistance(landmarks[18], wrist) * 1.65;
    if (!middleFolded || !ringFolded || !pinkyFolded) return false;

    // Thumb must sit above the index PIP (not tucked into the palm).
    if (landmarks[4].y > landmarks[6].y + palmSize * 0.15) return false;

    // The defining feature of a crossed heart: the thumb shaft (ip->tip, LMs
    // 3->4) VISUALLY INTERSECTS the index shaft (pip->tip, LMs 6->8). A pinch
    // touches at the tips but does not cross; a pistol never crosses.
    const a = landmarks[3];
    const b = landmarks[4];
    const c = landmarks[6];
    const d = landmarks[8];
    const segmentsCross = (ccw(a, b, c) !== ccw(a, b, d)) && (ccw(c, d, a) !== ccw(c, d, b));
    return segmentsCross;
}