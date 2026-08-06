// Pure gesture predicates, shared by main.js and the Node test suite.
// Normalized 2D hand landmarks: 21 x { x, y } in [0,1].

export function getDistance(p1, p2) {
    return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

export function isPeace(landmarks) {
    const wrist = landmarks[0];
    const palmSize = getDistance(landmarks[0], landmarks[9]);
    if (palmSize < 0.01) return false;

    // 1. Strict extension: index and middle finger must be fully extended perpendicularly
    const indexUp  = getDistance(landmarks[8], wrist) > getDistance(landmarks[6], wrist) * 1.15;
    const middleUp = getDistance(landmarks[12], wrist) > getDistance(landmarks[10], wrist) * 1.15;

    // 2. Strict fold: ring and pinky must be folded tightly
    const ringFolded  = getDistance(landmarks[16], wrist) < getDistance(landmarks[14], wrist) * 0.85;
    const pinkyFolded = getDistance(landmarks[20], wrist) < getDistance(landmarks[18], wrist) * 0.85;

    // 3. V-Shape spread: index tip and middle tip must be spread apart in a "V"
    const fingersSpread = getDistance(landmarks[8], landmarks[12]) > palmSize * 0.32;

    // 4. Thumb folded: thumb tip must be close to palm
    const thumbFolded = getDistance(landmarks[4], wrist) < palmSize * 1.1;

    return indexUp && middleUp && ringFolded && pinkyFolded && fingersSpread && thumbFolded;
}

export function isMiddleFinger(landmarks) {
    const wrist = landmarks[0];
    const palmSize = getDistance(landmarks[0], landmarks[9]);
    if (palmSize < 0.01) return false;

    // 1. Middle finger must be fully extended
    const middleUp = getDistance(landmarks[12], wrist) > getDistance(landmarks[10], wrist) * 1.12;

    // 2. Index, ring, and pinky must be folded (using a more lenient 1.05x threshold)
    const indexFolded = getDistance(landmarks[8], wrist) < getDistance(landmarks[6], wrist) * 1.05;
    const ringFolded  = getDistance(landmarks[16], wrist) < getDistance(landmarks[14], wrist) * 1.05;
    const pinkyFolded = getDistance(landmarks[20], wrist) < getDistance(landmarks[18], wrist) * 1.05;

    return middleUp && indexFolded && ringFolded && pinkyFolded;
}

/**
 * Korean finger heart 🫰: index up, middle+ring folded, thumb tip close to index tip.
 */
export function isFingerHeart(landmarks) {
    const wrist = landmarks[0];
    const palmSize = getDistance(landmarks[0], landmarks[9]); // Wrist to middle MCP
    if (palmSize < 0.01) return false;

    // Index must be fully extended and straight (at least 1.15x PIP distance from wrist)
    const indexUp = getDistance(landmarks[8], wrist) > getDistance(landmarks[6], wrist) * 1.15;

    // Middle and ring must be folded (1.10x — strict enough to reject a relaxed
    // open hand but tolerant of real-hand landmark noise).
    const middleFolded = getDistance(landmarks[12], wrist) < getDistance(landmarks[10], wrist) * 1.10;
    const ringFolded   = getDistance(landmarks[16], wrist) < getDistance(landmarks[14], wrist) * 1.10;

    if (!middleFolded || !ringFolded) return false;

    // Thumb tip (4) and index tip (8) must be PINCHED tight — an actual kiss, not a
    // relaxed adjacency. 0.55*palm still lets a natural hand read as a heart in
    // MediaPipe's normalized coords; 0.30 requires deliberate contact.
    const distThumbIndex = getDistance(landmarks[4], landmarks[8]);
    if (distThumbIndex > palmSize * 0.30) return false;

    return true;
}