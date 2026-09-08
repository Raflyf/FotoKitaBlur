import argparse
import math
import time

import cv2
import mediapipe as mp

LANDMARKS = {
    "wrist": 0,
    "thumb_ip": 3,
    "thumb_tip": 4,
    "index_pip": 6,
    "index_tip": 8,
    "middle_mcp": 9,
    "middle_pip": 10,
    "middle_tip": 12,
    "ring_pip": 14,
    "ring_tip": 16,
    "pinky_pip": 18,
    "pinky_tip": 20,
}

PEACE_HOLD_FRAMES = 5


def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def is_finger_extended(landmarks, mcp_idx, pip_idx, tip_idx):
    wrist = landmarks[0]
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    tip = landmarks[tip_idx]

    dist_tip_wrist = get_distance(tip, wrist)
    dist_pip_wrist = get_distance(pip, wrist)
    dist_tip_mcp = get_distance(tip, mcp)
    dist_pip_mcp = get_distance(pip, mcp)

    return (dist_tip_mcp > dist_pip_mcp * 1.15) and (dist_tip_wrist > dist_pip_wrist * 1.08)


def is_finger_folded(landmarks, mcp_idx, pip_idx, tip_idx):
    wrist = landmarks[0]
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    tip = landmarks[tip_idx]

    dist_tip_wrist = get_distance(tip, wrist)
    dist_pip_wrist = get_distance(pip, wrist)
    dist_tip_mcp = get_distance(tip, mcp)
    dist_pip_mcp = get_distance(pip, mcp)

    return (dist_tip_wrist <= dist_pip_wrist * 1.06) or (dist_tip_mcp <= dist_pip_mcp * 1.25)


def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


class PeaceBlurDetector:
    """Real-time peace-sign detector with gesture parity with the web frontend."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 enable_enhancement=True, blur_kernel_size=61):
        self.mp_hands = mp.solutions.hands
        self._hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # CLAHE is stateful; allocate once instead of per frame.
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if enable_enhancement else None
        self.blur_kernel_size = self._normalize_kernel(blur_kernel_size)
        self._peace_hold = 0

    @staticmethod
    def _normalize_kernel(size):
        if size < 3:
            size = 3
        if size % 2 == 0:
            size += 1
        return (size, size)

    @staticmethod
    def is_peace(landmarks):
        wrist = landmarks[0]
        palm_size = get_distance(landmarks[0], landmarks[9])
        if palm_size < 0.015:
            return False

        index_up = is_finger_extended(landmarks, 5, 6, 8)
        middle_up = is_finger_extended(landmarks, 9, 10, 12)
        ring_folded = is_finger_folded(landmarks, 13, 14, 16)
        pinky_folded = is_finger_folded(landmarks, 17, 18, 20)

        if not (index_up and middle_up and ring_folded and pinky_folded):
            return False

        finger_sep = get_distance(landmarks[8], landmarks[12])
        if finger_sep < palm_size * 0.18:
            return False

        thumb_dist = get_distance(landmarks[4], wrist)
        if thumb_dist > palm_size * 1.45 and thumb_dist > get_distance(landmarks[8], wrist) * 0.9:
            return False

        return True

    @staticmethod
    def is_middle_finger(landmarks):
        wrist = landmarks[0]
        palm_size = get_distance(landmarks[0], landmarks[9])
        if palm_size < 0.015:
            return False

        middle_up = is_finger_extended(landmarks, 9, 10, 12)
        index_folded = is_finger_folded(landmarks, 5, 6, 8)
        ring_folded = is_finger_folded(landmarks, 13, 14, 16)
        pinky_folded = is_finger_folded(landmarks, 17, 18, 20)

        middle_reach = get_distance(landmarks[12], wrist)
        index_reach = get_distance(landmarks[8], wrist)

        return middle_up and index_folded and ring_folded and pinky_folded and (middle_reach > index_reach * 1.15)

    @staticmethod
    def is_finger_heart(landmarks):
        wrist = landmarks[0]
        palm_size = get_distance(landmarks[0], landmarks[9])
        if palm_size < 0.015:
            return False

        middle_folded = is_finger_folded(landmarks, 9, 10, 12)
        ring_folded = is_finger_folded(landmarks, 13, 14, 16)
        pinky_folded = is_finger_folded(landmarks, 17, 18, 20)

        if not (middle_folded and ring_folded and pinky_folded):
            return False

        index_dist = get_distance(landmarks[8], wrist)
        middle_dist = get_distance(landmarks[12], wrist)
        if middle_dist < 0.01 or (index_dist / middle_dist) < 1.12:
            return False

        if landmarks[4].y > landmarks[6].y + palm_size * 0.15:
            return False

        a = landmarks[3]
        b = landmarks[4]
        c = landmarks[6]
        d = landmarks[8]

        segments_cross = (ccw(a, b, c) != ccw(a, b, d)) and (ccw(c, d, a) != ccw(c, d, b))
        return segments_cross

    @staticmethod
    def is_two_hand_heart(l1, l2):
        if not l1 or not l2:
            return None

        palm1 = get_distance(l1[0], l1[9])
        palm2 = get_distance(l2[0], l2[9])
        avg_palm = (palm1 + palm2) / 2.0
        if avg_palm < 0.015:
            return None

        dist_index = get_distance(l1[8], l2[8])
        dist_thumb = get_distance(l1[4], l2[4])

        if dist_index < avg_palm * 0.65 and dist_thumb < avg_palm * 0.65:
            index_y = (l1[8].y + l2[8].y) / 2.0
            thumb_y = (l1[4].y + l2[4].y) / 2.0

            if index_y <= thumb_y + avg_palm * 0.25:
                return {
                    "x": (l1[8].x + l2[8].x + l1[4].x + l2[4].x) / 4.0,
                    "y": (l1[8].y + l2[8].y + l1[4].y + l2[4].y) / 4.0
                }
        return None

    def process_frame(self, frame, blur_kernel_size=None):
        """Detect the peace sign on a (non-flipped) BGR frame.

        Applies optional CLAHE low-light enhancement, latches the blur state
        with temporal hysteresis (PEACE_HOLD_FRAMES), and returns
        (frame, blur_active, hand_count).
        """
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.clahe is not None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_result = self._hands.process(rgb)
        hand_count = len(hand_result.multi_hand_landmarks or [])

        peace_detected = False
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                if self.is_peace(hand_landmarks.landmark):
                    peace_detected = True
                    break

        if peace_detected:
            self._peace_hold = PEACE_HOLD_FRAMES
        elif self._peace_hold > 0:
            self._peace_hold -= 1

        blur_active = self._peace_hold > 0
        if blur_active:
            kernel = self._normalize_kernel(blur_kernel_size or self.blur_kernel_size[0])
            frame = cv2.GaussianBlur(frame, kernel, 0)

        return frame, blur_active, hand_count


def draw_hud(frame, fps, blur_active, hand_count):
    status = "BLUR ACTIVE" if blur_active else "PEACE TO BLUR"
    color = (0, 200, 0) if blur_active else (255, 255, 255)
    cv2.putText(frame, f"{status} | hands: {hand_count}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Peace blur detector (OpenCV + MediaPipe)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--min-confidence", type=float, default=0.5, dest="min_confidence",
                        help="Minimum detection/tracking confidence (default: 0.5)")
    parser.add_argument("--kernel", type=int, default=61, help="Blur kernel size, odd number (default: 61)")
    parser.add_argument("--no-enhance", action="store_true", help="Disable low-light CLAHE enhancement")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera index {args.camera}")
        return 1

    detector = PeaceBlurDetector(
        min_detection_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence,
        enable_enhancement=not args.no_enhance,
        blur_kernel_size=args.kernel,
    )

    fps = 0.0
    last_time = time.perf_counter()
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame, blur_active, hand_count = detector.process_frame(frame)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
            last_time = now

            draw_hud(frame, fps, blur_active, hand_count)
            cv2.imshow("Peace Blur", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
