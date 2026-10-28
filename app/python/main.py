"""UNO Q Linux side: receives IMU samples from the MCU over the Bridge and
either records them to CSV (for training) or classifies the activity live.

Open http://<board-ip>:7000 to switch mode and activity label while it runs.
MODE / RECORD_LABEL below only set what happens at startup.
"""
import collections
import threading
import time
from pathlib import Path

from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

import features
import models

# ---------------------------------------------------------------- settings
MODE = "classify"        # at startup: "idle", "record" or "classify"
RECORD_LABEL = "walking"  # at startup; used in the file name while recording
SMOOTH = 3               # majority vote over this many consecutive windows

ACCEL_LSB_PER_G = 2048.0    # MPU6050 at +-16 g (set in the sketch)
GYRO_LSB_PER_DPS = 16.4     # MPU6050 at +-2000 deg/s

HERE = Path(__file__).resolve().parent
RECORD_DIR = HERE.parent / "recordings"
MODES = ("idle", "record", "classify")

# Codes understood by show_activity() in the sketch; -1 blanks that half.
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


def show(peak_code, full_code):
    try:
        Bridge.call("show_activity", peak_code, full_code)
    except Exception as e:  # display is optional; never stop the app over it
        print("show_activity failed:", e)


# ---------------------------------------------------------------- record mode
class Recorder:
    def __init__(self, label):
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.label = label
        self.path = RECORD_DIR / f"{label}_{stamp}.csv"
        self.f = open(self.path, "w")
        self.f.write("t_ms,ax,ay,az,gx,gy,gz\n")
        self.samples = 0
        print(f"Recording '{label}' to {self.path}")

    def step(self):
        rows = drain()
        for r in rows:
            self.f.write("%d,%.5f,%.5f,%.5f,%.3f,%.3f,%.3f\n" % r)
        self.f.flush()
        self.samples += len(rows)
        time.sleep(0.2)

    def close(self):
        self.f.close()
        print(f"Saved {self.path.name}: {self.samples / features.FS:.0f} s")

    def status(self):
        return {"file": self.path.name, "seconds": round(self.samples / features.FS, 1)}


# ---------------------------------------------------------------- classify mode
class Classifier:
    def __init__(self):
        self.peak = models.load(HERE / "model_peak.json")
        self.full = models.load(HERE / "model_full.json")
        self.buf = collections.deque(maxlen=features.WINDOW)
        self.since_last = 0
        self.hist_peak = collections.deque(maxlen=SMOOTH)
        self.hist_full = collections.deque(maxlen=SMOOTH)
        self.last = None
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
        peak = self.peak.feature(window)
        self.hist_peak.append(self.peak.predict(window))
        self.hist_full.append(self.full.predict(window))
        s1, s2 = self._vote(self.hist_peak), self._vote(self.hist_full)
        self.last = {"peak_g": round(peak, 2), "model1": s1, "model2": s2}
        print(f"peak={peak:5.2f} g  model1(peak)={s1:<9} model2(all)={s2:<9}")
        show(CODE[s1], CODE[s2])

    def close(self):
        show(-1, -1)

    def status(self):
        return {"direction": self.peak.direction, "prediction": self.last,
                "warming_up": self.last is None}


class Idle:
    def step(self):
        drain()
        time.sleep(0.2)

    def close(self):
        pass

    def status(self):
        return {}


# ---------------------------------------------------------------- web control page
ui = WebUI()


class Controller:
    """Owns the current mode. Web handlers only queue requests; the app loop
    applies them, so the recorder/classifier is only touched from one thread."""

    def __init__(self, mode, label):
        self.mode, self.label = "idle", label
        self.worker = Idle()
        self.error = None
        self.requests = collections.deque()
        self.recordings = []
        self.rate_hz, self._rate_t, self._rate_n = 0.0, time.time(), 0
        self._last_push = 0.0
        self._switch(mode, label)

    def request(self, **kwargs):
        with lock:
            self.requests.append(kwargs)

    def _switch(self, mode, label):
        self.worker.close()
        self.worker, self.mode, self.label = Idle(), "idle", label
        drain()  # don't let samples from before the switch leak into the new file
        try:
            if mode == "record":
                self.worker = Recorder(label)
            elif mode == "classify":
                self.worker = Classifier()
            self.mode, self.error = mode, None
        except Exception as e:  # e.g. model files missing
            self.error = f"Couldn't start {mode}: {e}"
            print(self.error)
        self._scan_recordings()

    def _apply(self, req):
        if "delete" in req:
            names = {p.name for p in RECORD_DIR.glob("*.csv")}
            current = self.worker.path.name if isinstance(self.worker, Recorder) else None
            if req["delete"] in names and req["delete"] != current:
                (RECORD_DIR / req["delete"]).unlink()
                print(f"Deleted {req['delete']}")
                self._scan_recordings()
            return
        mode = req.get("mode", self.mode)
        label = req.get("label", self.label)
        if mode not in MODES or label not in features.ACTIVITIES:
            return
        # A new label while recording starts a new file, so each file has one activity.
        if mode != self.mode or (mode == "record" and label != self.label):
            self._switch(mode, label)
        else:
            self.label = label

    def _scan_recordings(self):
        out = []
        for p in sorted(RECORD_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
            with open(p) as f:
                n = sum(1 for _ in f) - 1
            out.append({"name": p.name, "seconds": round(n / features.FS)})
        self.recordings = out

    def status(self):
        return {"mode": self.mode, "label": self.label, "error": self.error,
                "rate_hz": self.rate_hz, "labels": features.ACTIVITIES,
                "recordings": self.recordings, **self.worker.status()}

    def step(self):
        with lock:
            reqs = list(self.requests)
            self.requests.clear()
        for r in reqs:
            self._apply(r)
        self.worker.step()

        now = time.time()
        if now - self._rate_t >= 2:
            self.rate_hz = round((received - self._rate_n) / (now - self._rate_t), 1)
            self._rate_t, self._rate_n = now, received
        if now - self._last_push >= 0.5:
            self._last_push = now
            ui.send_message("status", self.status())


ctl = Controller(MODE, RECORD_LABEL)

ui.on_connect(lambda sid: ctl.request())  # empty request -> fresh status soon
ui.on_message("set_mode", lambda sid, data: ctl.request(mode=data.get("mode")))
ui.on_message("set_label", lambda sid, data: ctl.request(label=data.get("label")))
ui.on_message("delete_recording", lambda sid, data: ctl.request(delete=data.get("name")))


def loop():
    ctl.step()


App.run(user_loop=loop)
