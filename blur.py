import cv2
import mediapipe as mp

class PeaceBlurDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def get_distance(self, p1, p2):
        import math
        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2
        )

    def is_finger_extended(self, tip, pip, landmarks):
        wrist = landmarks[0]
        return self.get_distance(landmarks[tip], wrist) > self.get_distance(landmarks[pip], wrist)

    def is_peace(self, landmarks):
        index_up = self.is_finger_extended(8, 6, landmarks)
        middle_up = self.is_finger_extended(12, 10, landmarks)
        ring_up = self.is_finger_extended(16, 14, landmarks)
        pinky_up = self.is_finger_extended(20, 18, landmarks)

        return (
            index_up
            and middle_up
            and not ring_up
            and not pinky_up
        )

    def process_frame(self, frame, blur_kernel_size=(61, 61)):
        """
        Processes a single frame, flips it, detects hands,
        checks for peace sign, and applies blur if detected.
        Returns the processed frame and a boolean indicating if peace sign was detected.
        """
        frame = cv2.flip(frame, 1)
        
        # --- Pre-processing for low-light contrast enhancement (CLAHE) ---
        # Convert BGR to LAB color space to isolate the lightness channel (L)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back and convert to BGR, then to RGB for MediaPipe inference
        enhanced_bgr = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_bgr, cv2.COLOR_LAB2BGR)
        rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        # ------------------------------------------------------------------
        
        hand_result = self.hands.process(rgb)

        peace_detected = False
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                if self.is_peace(hand_landmarks.landmark):
                    peace_detected = True
                    break

        if peace_detected:
            frame = cv2.GaussianBlur(frame, blur_kernel_size, 0)

        return frame, peace_detected

if __name__ == '__main__':
    # open camera
    cap = cv2.VideoCapture(0)
    detector = PeaceBlurDetector()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame, peace_detected = detector.process_frame(frame)

        cv2.imshow("Peace Blur", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

