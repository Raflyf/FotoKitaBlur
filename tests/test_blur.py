"""Unit tests for blur.py gesture logic (no camera or MediaPipe inference needed).

Run with: python -m unittest tests.test_blur -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blur import PeaceBlurDetector  # noqa: E402


def lm(x, y):
    return SimpleNamespace(x=x, y=y)


def make_hand(points):
    landmarks = [lm(0.0, 0.0) for _ in range(21)]
    for idx, (x, y) in points.items():
        landmarks[idx] = lm(x, y)
    return landmarks


# Landmark ids used by blur.py (see LANDMARKS dict)
W, THUMB_T, INDEX_PIP, INDEX_T, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_T, RING_PIP, RING_T, PINKY_PIP, PINKY_T = (
    0, 4, 6, 8, 9, 10, 12, 14, 16, 18, 20,
)


def peace_hand():
    """A canonical strict peace sign: index+middle up, ring+pinky folded,
    V-spread, thumb tucked. All ratios comfortably pass is_peace().
    """
    return make_hand({
        THUMB_T: (0.05, -0.05),
        INDEX_PIP: (0.10, -0.15),
        INDEX_T: (0.15, -0.60),
        MIDDLE_MCP: (0.30, 0.30),
        MIDDLE_PIP: (0.25, -0.15),
        MIDDLE_T: (0.35, -0.65),
        RING_PIP: (0.50, 0.10),
        RING_T: (0.35, 0.00),
        PINKY_PIP: (0.55, 0.15),
        PINKY_T: (0.40, 0.05),
    })


def fist_hand():
    """All fingers folded toward the palm."""
    return make_hand({
        THUMB_T: (0.05, 0.05),
        INDEX_PIP: (0.10, 0.05),
        INDEX_T: (0.12, 0.10),
        MIDDLE_MCP: (0.30, 0.30),
        MIDDLE_PIP: (0.25, 0.10),
        MIDDLE_T: (0.30, 0.12),
        RING_PIP: (0.45, 0.10),
        RING_T: (0.47, 0.12),
        PINKY_PIP: (0.55, 0.10),
        PINKY_T: (0.58, 0.12),
    })


def middle_finger_hand():
    return make_hand({
        THUMB_T: (0.05, -0.05),
        INDEX_PIP: (0.10, 0.05),
        INDEX_T: (0.00, 0.05),
        MIDDLE_MCP: (0.30, 0.30),
        MIDDLE_PIP: (0.25, -0.10),
        MIDDLE_T: (0.30, -0.60),
        RING_PIP: (0.50, 0.10),
        RING_T: (0.40, 0.05),
        PINKY_PIP: (0.55, 0.15),
        PINKY_T: (0.45, 0.05),
    })


def finger_heart_hand():
    """True crossed heart: thumb-tip left of PIP, index-tip right of PIP, tips pinched."""
    return make_hand({
        THUMB_T: (0.28, -0.42),
        INDEX_PIP: (0.27, 0.00),
        INDEX_T: (0.22, -0.42),
        MIDDLE_MCP: (0.30, 0.30),
        MIDDLE_PIP: (0.30, -0.10),
        MIDDLE_T: (0.25, 0.00),
        RING_PIP: (0.50, 0.10),
        RING_T: (0.45, 0.05),
        PINKY_PIP: (0.55, 0.15),
        PINKY_T: (0.50, 0.10),
    })


class TestNormalizeKernel(unittest.TestCase):
    def test_odd_size_passthrough(self):
        self.assertEqual(PeaceBlurDetector._normalize_kernel(61), (61, 61))

    def test_even_size_rounded_up(self):
        self.assertEqual(PeaceBlurDetector._normalize_kernel(60), (61, 61))

    def test_undersized_clamped(self):
        self.assertEqual(PeaceBlurDetector._normalize_kernel(2), (3, 3))


class TestIsPeace(unittest.TestCase):
    def test_strict_peace_detected(self):
        self.assertIs(PeaceBlurDetector.is_peace(peace_hand()), True)

    def test_fist_rejected(self):
        self.assertIs(PeaceBlurDetector.is_peace(fist_hand()), False)

    def test_middle_finger_rejected(self):
        self.assertIs(PeaceBlurDetector.is_peace(middle_finger_hand()), False)

    def test_degenerate_palm_rejected(self):
        degenerate = [lm(0.0, 0.0) for _ in range(21)]
        self.assertIs(PeaceBlurDetector.is_peace(degenerate), False)


class TestOtherGestures(unittest.TestCase):
    def test_middle_finger_detected(self):
        self.assertIs(PeaceBlurDetector.is_middle_finger(middle_finger_hand()), True)

    def test_fist_not_middle_finger(self):
        self.assertIs(PeaceBlurDetector.is_middle_finger(fist_hand()), False)

    def test_finger_heart_detected(self):
        self.assertIs(PeaceBlurDetector.is_finger_heart(finger_heart_hand()), True)

    def test_peace_not_finger_heart(self):
        self.assertIs(PeaceBlurDetector.is_finger_heart(peace_hand()), False)


if __name__ == "__main__":
    unittest.main()
