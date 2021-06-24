import sys
import time

import pyautogui

from eye_control import Webcam

DWELL_NOOP = 0
DWELL_STOP = 1
DWELL_MOVE = 2
WKEY_UP = False

class Timer:
    def __init__(self):
        self.init = time.time()
        self.start = time.time()
        self.running = False

    def run(self):
        self.start = time.time()
        self.running = True

    def stop(self):
        self.running = False

    def log(self):
        logger = time.time() - self.init
        return f"{time.time() - self.start},"

    def elapsed(self):
        return time.time() - self.start if self.running else 0

def handle_args():
    dwell_action = DWELL_NOOP
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "stop":
            dwell_action = DWELL_STOP
        elif arg == "move":
            dwell_action = DWELL_MOVE

    return dwell_action

def handle_eye_pos(pos, prev_pos, dt, dwell_action):
    global WKEY_UP

    # check if eye position has changed
    displace_left, displace_right = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
    moved = (displace_left != 0 and displace_right != 0)
    # print(dt.log(), f"{displace_left}, {displace_right}")

    if dt.running and dt.elapsed() > 3:
        if dwell_action == DWELL_STOP:
            return moved, pos, True
        elif dwell_action == DWELL_MOVE and WKEY_UP:
            WKEY_UP = False
            pyautogui.keyDown('w')

    if not moved and not dt.running:
        dt.run()
    elif moved and dt.running:
        dt.stop()
        if dwell_action == DWELL_MOVE and not WKEY_UP:
            WKEY_UP = True
            pyautiogui.keyUp('w')

    return moved, pos, False

def main():
    dwell_action = handle_args()

    # highlight_minecraft_window()

    wc = Webcam()
    dt = Timer()
    dt.run()
    prev_pos = (0, 0)

    while True:
        pos = wc.get_eye_pos()
        moved, prev_pos, term = handle_eye_pos(pos, prev_pos, dt, dwell_action)

        if term:
            break

if __name__ == "__main__":
    main()
