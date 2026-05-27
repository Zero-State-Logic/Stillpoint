"""
gesture_controller.py
---------------------
Main entry point. Maps hand gestures to four modes:

    MOUSE   : index only            -> move + pinch-click
    VOLUME  : thumb + index only    -> distance = system volume
    DRAW    : open palm (5 fingers) -> freehand draw on canvas
    TABS    : index + middle only   -> swipe left/right to cycle tabs

Press 'q' to quit, 'c' to clear the drawing canvas.
"""

import math
import time

import cv2
import numpy as np
import pyautogui

from hand_tracker import HandTracker

# Volume control is Windows-only. Import lazily so the rest still works on
# other OSes for testing.
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    _devices = AudioUtilities.GetSpeakers()
    _interface = _devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    _volume = cast(_interface, POINTER(IAudioEndpointVolume))
    VOL_MIN, VOL_MAX = _volume.GetVolumeRange()[:2]
    VOLUME_OK = True
except Exception as e:  # noqa: BLE001
    print(f"[warn] pycaw unavailable, volume mode disabled: {e}")
    _volume = None
    VOL_MIN, VOL_MAX = -65.0, 0.0
    VOLUME_OK = False


# ---------- tunables ----------
CAM_W, CAM_H = 1280, 720       # webcam resolution
FRAME_REDUCTION = 100          # dead-band around the camera edges
SMOOTHING = 5                  # higher = smoother but laggier cursor
CLICK_PINCH_THRESHOLD = 30     # px distance for a mouse click
TAB_SWIPE_THRESHOLD = 80       # px hand travel to count as a tab swipe
TAB_COOLDOWN = 0.8             # seconds between tab switches
MODE_SWITCH_COOLDOWN = 0.5     # debounce mode transitions

pyautogui.FAILSAFE = False     # don't crash if cursor hits a corner

SCREEN_W, SCREEN_H = pyautogui.size()


def fingers_to_mode(fingers):
    """
    fingers = [thumb, index, middle, ring, pinky]
    Returns one of: 'mouse', 'volume', 'draw', 'tabs', or None.
    """
    if fingers == [0, 1, 0, 0, 0]:
        return "mouse"
    if fingers == [1, 1, 0, 0, 0]:
        return "volume"
    if fingers == [1, 1, 1, 1, 1]:
        return "draw"
    if fingers == [0, 1, 1, 0, 0]:
        return "tabs"
    return None


def draw_hud(frame, mode, vol_pct=None):
    """Top-left heads-up display showing current mode."""
    cv2.rectangle(frame, (0, 0), (320, 70), (0, 0, 0), -1)
    cv2.putText(
        frame, f"MODE: {mode.upper()}", (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
    )
    if vol_pct is not None:
        cv2.putText(
            frame, f"VOL: {int(vol_pct)}%", (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)

    tracker = HandTracker(max_hands=1, detection_confidence=0.7, tracking_confidence=0.7)

    # Cursor smoothing state
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    # Drawing state
    canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    draw_prev = None  # last fingertip point in DRAW mode
    draw_color = (0, 255, 0)
    draw_thickness = 8

    # Tabs state
    tabs_anchor_x = None
    last_tab_time = 0.0

    # General state
    mode = "idle"
    last_mode_switch = 0.0
    vol_pct_display = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        frame = tracker.find_hands(frame, draw=True)
        lms = tracker.find_position(frame)

        if lms:
            fingers = tracker.fingers_up()
            requested = fingers_to_mode(fingers)

            now = time.time()
            if requested and requested != mode and (now - last_mode_switch) > MODE_SWITCH_COOLDOWN:
                mode = requested
                last_mode_switch = now
                # reset per-mode transient state on entry
                draw_prev = None
                tabs_anchor_x = None

            # ---------- MOUSE ----------
            if mode == "mouse":
                # index fingertip = landmark 8
                _, ix, iy = lms[8]
                # Map within a reduced rectangle so the user doesn't need to
                # reach the camera edges to reach the screen edges.
                x_mapped = np.interp(ix, (FRAME_REDUCTION, CAM_W - FRAME_REDUCTION), (0, SCREEN_W))
                y_mapped = np.interp(iy, (FRAME_REDUCTION, CAM_H - FRAME_REDUCTION), (0, SCREEN_H))

                # exponential-ish smoothing
                curr_x = prev_x + (x_mapped - prev_x) / SMOOTHING
                curr_y = prev_y + (y_mapped - prev_y) / SMOOTHING
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                cv2.circle(frame, (ix, iy), 12, (255, 0, 255), cv2.FILLED)

                # Pinch (thumb tip 4 close to index tip 8) -> click
                _, tx, ty = lms[4]
                pinch = math.hypot(tx - ix, ty - iy)
                if pinch < CLICK_PINCH_THRESHOLD:
                    cv2.circle(frame, (ix, iy), 18, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()
                    time.sleep(0.25)  # debounce so one pinch = one click

            # ---------- VOLUME ----------
            elif mode == "volume":
                _, tx, ty = lms[4]   # thumb tip
                _, ix, iy = lms[8]   # index tip
                cx, cy = (tx + ix) // 2, (ty + iy) // 2

                cv2.circle(frame, (tx, ty), 12, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (ix, iy), 12, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (tx, ty), (ix, iy), (255, 0, 255), 3)
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

                length = math.hypot(tx - ix, ty - iy)
                # 30 px (pinched) -> min vol, 250 px (spread) -> max vol
                vol_db = np.interp(length, (30, 250), (VOL_MIN, VOL_MAX))
                vol_pct_display = np.interp(length, (30, 250), (0, 100))

                if VOLUME_OK and _volume is not None:
                    _volume.SetMasterVolumeLevel(float(vol_db), None)

                # Volume bar
                bar = int(np.interp(length, (30, 250), (400, 150)))
                cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(frame, (50, bar), (85, 400), (0, 255, 0), cv2.FILLED)

            # ---------- DRAW ----------
            elif mode == "draw":
                # use index fingertip as the pen
                _, ix, iy = lms[8]
                if draw_prev is None:
                    draw_prev = (ix, iy)
                cv2.line(canvas, draw_prev, (ix, iy), draw_color, draw_thickness)
                draw_prev = (ix, iy)
                cv2.circle(frame, (ix, iy), 10, draw_color, cv2.FILLED)

            # ---------- TABS ----------
            elif mode == "tabs":
                # Use the midpoint between index and middle fingertips
                _, ix, iy = lms[8]
                _, mx, my = lms[12]
                hand_x = (ix + mx) // 2
                hand_y = (iy + my) // 2
                cv2.circle(frame, (hand_x, hand_y), 14, (0, 200, 255), cv2.FILLED)

                if tabs_anchor_x is None:
                    tabs_anchor_x = hand_x

                dx = hand_x - tabs_anchor_x
                if (now - last_tab_time) > TAB_COOLDOWN:
                    if dx > TAB_SWIPE_THRESHOLD:
                        pyautogui.hotkey("ctrl", "tab")     # next tab
                        last_tab_time = now
                        tabs_anchor_x = hand_x
                    elif dx < -TAB_SWIPE_THRESHOLD:
                        pyautogui.hotkey("ctrl", "shift", "tab")  # previous tab
                        last_tab_time = now
                        tabs_anchor_x = hand_x
            else:
                # no recognized gesture this frame; keep current mode but
                # break draw stroke so re-entering doesn't connect to old point
                draw_prev = None
                tabs_anchor_x = None
        else:
            draw_prev = None
            tabs_anchor_x = None

        # Composite the persistent canvas onto the live feed
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inv)
        frame = cv2.bitwise_or(frame, canvas)

        draw_hud(frame, mode, vol_pct_display if mode == "volume" else None)

        cv2.imshow("Gesture Controlled Laptop", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            canvas[:] = 0

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
