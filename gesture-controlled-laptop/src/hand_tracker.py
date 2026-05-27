"""
hand_tracker.py
---------------
Thin wrapper around MediaPipe Hands that exposes:
  - landmark coordinates in pixel space
  - a fingersUp() helper that returns [thumb, index, middle, ring, pinky]
    where 1 = extended, 0 = folded

This is the single source of truth for hand data; gesture_controller.py
just consumes it.
"""

import cv2
import mediapipe as mp


# MediaPipe landmark indices for fingertips (thumb, index, middle, ring, pinky)
TIP_IDS = [4, 8, 12, 16, 20]


class HandTracker:
    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        self.landmark_list = []  # populated by find_position()

    def find_hands(self, frame, draw: bool = True):
        """Run detection. Returns the frame (with landmarks drawn if requested)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_lms, self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    def find_position(self, frame, hand_idx: int = 0):
        """
        Returns a list of [id, x, y] in pixel coordinates for the requested hand.
        Empty list if no hand is visible. Also caches into self.landmark_list.
        """
        self.landmark_list = []
        if not self.results or not self.results.multi_hand_landmarks:
            return self.landmark_list

        if hand_idx >= len(self.results.multi_hand_landmarks):
            return self.landmark_list

        h, w, _ = frame.shape
        hand = self.results.multi_hand_landmarks[hand_idx]
        for idx, lm in enumerate(hand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            self.landmark_list.append([idx, cx, cy])
        return self.landmark_list

    def fingers_up(self):
        """
        Returns [thumb, index, middle, ring, pinky] with 1 = up, 0 = down.

        Thumb is checked on the X axis (left/right of its IP joint) because
        it points sideways relative to the palm; the other four are checked
        on the Y axis (tip above the PIP joint).
        Assumes a right hand viewed in a mirrored (selfie) feed.
        """
        if not self.landmark_list or len(self.landmark_list) < 21:
            return [0, 0, 0, 0, 0]

        fingers = []
        lms = self.landmark_list

        # Thumb: tip x vs the joint two steps down the chain
        if lms[TIP_IDS[0]][1] > lms[TIP_IDS[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers: tip y vs PIP joint (tip_id - 2) y
        for i in range(1, 5):
            if lms[TIP_IDS[i]][2] < lms[TIP_IDS[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def close(self):
        self.hands.close()
