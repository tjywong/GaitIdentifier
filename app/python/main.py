"""UNO Q Linux side: receives IMU samples from the MCU over the Bridge and
either records them to CSV (for training) or classifies the activity live.

Set MODE below, then press Run in App Lab.
"""
import collections
import threading
import time
from pathlib import Path

from arduino.app_utils import App, Bridge

import features
import models

# ---------------------------------------------------------------- settings
MODE = "classify"        # "record" or "classify"
RECORD_LABEL = "walking"  # used in the file name while MODE == "record"
SMOOTH = 3               # majority vote over this many consecutive windows

ACCEL_LSB_PER_G = 2048.0    # MPU6050 at +-16 g (set in the sketch)
GYRO_LSB_PER_DPS = 16.4     # MPU6050 at +-2000 deg/s

HERE = Path(__file__).resolve().parent
RECORD_DIR = HERE.parent / "recordings"

# Codes understood by show_activity() in the sketch.
CODE = {name: i for i, name in enumerate(features.ACTIVITIES)}

# ---------------------------------------------------------------- state
lock = threading.Lock()
pending = collections.deque()   # (t_ms, ax, ay, az, gx, gy, gz) in g / dps
received = 0


def on_imu_batch(batch):
    """Bridge handler. batch = 't,ax,ay,az,gx,gy,gz;' repeated, raw ints."""
    global received
    rows = []
    for rec in batch.split(";"):
        if not rec:
            continue
        p = rec.split(",")
        if len(p) != 7:
            continue
        t, ax, ay, az, gx, gy, gz = (int(v) for v in p)
        rows.append((t, ax / ACCEL_LSB_PER_G, ay / ACCEL_LSB_PER_G, az / ACCEL_LSB_PER_G,
                     gx / GYRO_LSB_PER_DPS, gy / GYRO_LSB_PER_DPS, gz / GYRO_LSB_PER_DPS))
    with lock:
        pending.extend(rows)
        received += len(rows)


Bridge.provide("imu_batch", on_imu_batch)


def drain():
    with lock:
        rows = list(pending)
        pending.clear()
    return rows


# ---------------------------------------------------------------- record mode
class Recorder:
    def __init__(self):
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = RECORD_DIR / f"{RECORD_LABEL}_{stamp}.csv"
        self.f = open(self.path, "w")
        self.f.write("t_ms,ax,ay,az,gx,gy,gz\n")
        self.last_report = time.time()
        print(f"Recording '{RECORD_LABEL}' to {self.path}")

    def step(self):
        for r in drain():
            self.f.write("%d,%.5f,%.5f,%.5f,%.3f,%.3f,%.3f\n" % r)
        self.f.flush()
        if time.time() - self.last_report > 5:
            self.last_report = time.time()
            print(f"  {received} samples ({received / features.FS:.0f} s)")
        time.sleep(0.2)


# ---------------------------------------------------------------- classify mode
class Classifier:
    def __init__(self):
        self.peak = models.load(HERE / "model_peak.json")
        self.full = models.load(HERE / "model_full.json")
        self.buf = collections.deque(maxlen=features.WINDOW)
        self.since_last = 0
        self.hist_peak = collections.deque(maxlen=SMOOTH)
        self.hist_full = collections.deque(maxlen=SMOOTH)
        print(f"Model 1 uses peak accel along {self.peak.direction}; "
              f"model 2 is a {len(self.full.trees)}-tree random forest")

    @staticmethod
    def _vote(hist):
        return collections.Counter(hist).most_common(1)[0][0]

    def step(self):
        for r in drain():
            self.buf.append(r[1:])
            self.since_last += 1
            if len(self.buf) == features.WINDOW and self.since_last >= features.HOP:
                self.since_last = 0
                self._classify(list(self.buf))
        time.sleep(0.05)

    def _classify(self, window):
        p1 = self.peak.predict(window)
        p2 = self.full.predict(window)
        self.hist_peak.append(p1)
        self.hist_full.append(p2)
        s1, s2 = self._vote(self.hist_peak), self._vote(self.hist_full)
        print(f"peak={self.peak.feature(window):5.2f} g  "
              f"model1(peak)={s1:<9} model2(all)={s2:<9}")
        try:
            Bridge.call("show_activity", CODE[s1], CODE[s2])
        except Exception as e:  # display is optional; never stop classifying
            print("show_activity failed:", e)


worker = Recorder() if MODE == "record" else Classifier()


def loop():
    worker.step()


App.run(user_loop=loop)
