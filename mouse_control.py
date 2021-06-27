import signal
import sys
import time

import pyautogui

from eye_control import Webcam

DIMENSIONS = pyautogui.size()
MOVE_SPEED = int(DIMENSIONS[0]/30)
DWELL_NOOP = 0
DWELL_STOP = 1
DWELL_MOVE = 2
WKEY_UP = False

class ProgramInterrupt(Exception):
    pass

def signal_handler(signum, frame):
    raise ProgramInterrupt

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
        return f"{time.time() - self.init},"

    def elapsed(self):
        return time.time() - self.start if self.running else 0

def handle_args():
    log_mode = "log" in sys.argv

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "stop":
            return DWELL_STOP, log_mode
        elif arg == "move":
            return DWELL_MOVE, log_mode

    return DWELL_NOOP, log_mode

def handle_eye_pos(pos, prev_pos, dt, config):
    global WKEY_UP
    dwell_action, log_mode = config

    # check if eye position has changed
    displace_left, displace_right = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
    print(dt.log(), f"{pos[0]}, {pos[1]}")
    moved = (displace_left != 0 and displace_right != 0)

    if not moved and not dt.running:
        dt.run()
    elif moved and dt.running:
        dt.stop()
        if dwell_action == DWELL_MOVE and not WKEY_UP:
            WKEY_UP = True
            pyautiogui.keyUp('w')

    if dt.running and dt.elapsed() > 3:
        if dwell_action == DWELL_STOP:
            return pos, True
        elif dwell_action == DWELL_MOVE and WKEY_UP:
            WKEY_UP = False
            pyautogui.keyDown('w')

    if not log_mode and pos[0] == pos[1]:
        pyautogui.move(pos[0] * MOVE_SPEED, 0)

    return pos, False

def main():
    from threading import Thread

    # register interrupt handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    config = handle_args()
    # highlight_minecraft_window()

    # initialize collection timer
    ct = Timer()
    ct.run()

    # initialize dwelling timer
    dt = Timer()
    dt.run()

    # initialize webcam and run thread
    wc = Webcam()
    th = Thread(target=wc.run, name='webcam')
    th.start()

    try:
        prev_pos = (0, 0)
        while True:
            if ct.elapsed() < 1/60:
                continue

            pos = wc.get_eye_pos()
            prev_pos, term = handle_eye_pos(pos, prev_pos, dt, config)
            if term:
                wc.terminate()
                break

            ct.run()

    except ProgramInterrupt:
        wc.terminate()

    # wait for webcam to shut down
    th.join()

    sys.exit(0)

if __name__ == "__main__":
    main()
