"""STILLPOINT: mid-air hand pointing that clicks where you meant.

Pure-Python core (filters, hand geometry, pinch detection, the pointer
pipeline, the Fitts-task maths and the simulator) depends only on numpy, so
it runs headless. The live app imports OpenCV, MediaPipe, PyAutoGUI and pycaw
lazily, only when it starts.

Author: Faysal Ali Shah. MIT licence.
"""

__version__ = "1.0.0"

VARIANTS = ("baseline", "oneeuro", "oneeuro+hysteresis", "stillpoint")
