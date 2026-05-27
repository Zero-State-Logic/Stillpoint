# Gesture Controlled Laptop

Control your Windows laptop with your bare hand. A webcam watches your hand, MediaPipe gives back 21 landmarks per frame, and a small state machine maps finger poses to one of four interaction modes — moving the mouse, adjusting volume, drawing on screen, or flipping through tabs.

![demo](assets/demo.gif)

## Gestures

| Gesture | Mode | What it does |
|---|---|---|
| ☝️ Index finger only | **Mouse** | Move the cursor with your fingertip. Pinch thumb + index together to click. |
| 🤏 Thumb + index only | **Volume** | Distance between the two fingertips sets the system volume. Pinch to mute, spread for max. |
| 🖐️ All five fingers extended (open palm) | **Draw** | Freehand drawing on a transparent overlay using your index finger. Press `c` to clear. |
| ✌️ Index + middle finger | **Tabs** | Swipe right to advance one tab (`Ctrl+Tab`), swipe left to go back (`Ctrl+Shift+Tab`). |

Press `q` in the preview window to quit, `c` to clear the drawing canvas.

![gesture overview](assets/gestures_overview.png)

## How it works

```
webcam ──► OpenCV ──► MediaPipe Hands ──► 21 landmarks
                                              │
                                              ▼
                                   fingers_up() [thumb, index, middle, ring, pinky]
                                              │
                                              ▼
                                       mode dispatcher
                       ┌──────────────┬──────────────┬─────────────┬────────────┐
                       ▼              ▼              ▼             ▼
                    Mouse           Volume          Draw          Tabs
                  (PyAutoGUI)       (PyCaw)        (OpenCV)    (PyAutoGUI)
```

Two files do the work:

- `src/hand_tracker.py` — thin wrapper around `mediapipe.solutions.hands`. Returns landmark pixel coordinates and a `fingers_up()` helper. Reused by every mode.
- `src/gesture_controller.py` — main loop. Reads the webcam, picks a mode from the finger pattern, and dispatches to the right action. Cursor uses exponential smoothing so it isn't jittery, volume is mapped from fingertip distance to the dB range PyCaw reports, drawing keeps a persistent canvas composited onto the live feed each frame, and tab switching uses a horizontal swipe threshold with a cooldown so one swipe = one tab.

## Setup

**Requirements**: Windows 10/11, a webcam, Python 3.10 (3.11 works too; avoid 3.12+ until MediaPipe ships matching wheels).

```bash
git clone https://github.com/<you>/gesture-controlled-laptop.git
cd gesture-controlled-laptop

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python src/gesture_controller.py
```

A window titled **Gesture Controlled Laptop** will open with your webcam feed and a HUD in the top-left showing the current mode.

## Tuning

The constants at the top of `gesture_controller.py` are the main knobs:

| Constant | Default | What it controls |
|---|---|---|
| `SMOOTHING` | 5 | Higher = smoother cursor, more lag |
| `FRAME_REDUCTION` | 100 | Dead-band in pixels around the camera edges |
| `CLICK_PINCH_THRESHOLD` | 30 | Pixel distance for thumb-index pinch to register as a click |
| `TAB_SWIPE_THRESHOLD` | 80 | Horizontal hand travel (pixels) before a tab switch fires |
| `TAB_COOLDOWN` | 0.8 s | Minimum time between consecutive tab switches |
| `MODE_SWITCH_COOLDOWN` | 0.5 s | Debounce time when transitioning between modes |

## Troubleshooting

**`pip` can't find a MediaPipe wheel** — you're probably on Python 3.12+. Use 3.10 or 3.11.

**Cursor doesn't reach screen edges** — increase `FRAME_REDUCTION` so the active rectangle in the camera frame is smaller relative to the screen mapping. (Or lower it to require less hand travel.)

**Volume jumps around** — that's the natural noise from fingertip distance. Lower the upper bound of the `np.interp` call in the volume branch (e.g. from 250 to 200) so the full range is reached with less hand movement.

**`pycaw` import error on non-Windows** — expected; volume mode is no-op'd off Windows. Everything else still runs.

**Cursor hits a screen corner and PyAutoGUI crashes** — already disabled via `pyautogui.FAILSAFE = False`. If you want the safety on, remove that line.

## Project layout

```
gesture-controlled-laptop/
├── src/
│   ├── hand_tracker.py        # MediaPipe wrapper + fingers_up()
│   └── gesture_controller.py  # main loop, mode dispatch
├── assets/                    # screenshots, demo gif
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

## License

MIT.
