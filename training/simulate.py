"""Generate rough synthetic ankle-IMU recordings to smoke-test the pipeline.

    python training/simulate.py && python training/train.py --data data_synthetic

These are hand-made approximations (stride frequency, impact spikes, shin
swing), NOT real data. Scores on them say the code works, not that the
wearable works. Train the real models on data recorded with the device.

Assumed mounting: x up the shin, y forward, z lateral (out of the leg).
"""
import argparse
import math
import random
from pathlib import Path

FS = 50
ROOT = Path(__file__).resolve().parent.parent


def rotate(v, rx, ry, rz):
    x, y, z = v
    y, z = y * math.cos(rx) - z * math.sin(rx), y * math.sin(rx) + z * math.cos(rx)
    x, z = x * math.cos(ry) + z * math.sin(ry), -x * math.sin(ry) + z * math.cos(ry)
    x, y = x * math.cos(rz) - y * math.sin(rz), x * math.sin(rz) + y * math.cos(rz)
    return x, y, z


def session(activity, seconds, rng):
    mount = [rng.gauss(0, math.radians(8)) for _ in range(3)]  # strap never sits the same
    air = activity == "biking" and rng.random() < 0.5            # lying on back, air-biking
    f = {"standing": 0, "walking": rng.uniform(0.8, 1.05),
         "running": rng.uniform(1.3, 1.6), "biking": rng.uniform(1.0, 1.5)}[activity]
    swing = {"standing": 0, "walking": math.radians(rng.uniform(30, 40)),
             "running": math.radians(rng.uniform(45, 65)), "biking": math.radians(rng.uniform(20, 30))}[activity]
    impact = {"standing": 0, "walking": rng.uniform(1.0, 2.0),
              "running": rng.uniform(3.5, 6.5), "biking": 0}[activity]
    base_tilt = math.radians(-70 if air else (30 if activity == "biking" else 0))
    rows, phase = [], rng.random()
    for n in range(int(seconds * FS)):
        t = n / FS
        phase += f / FS * (1 + rng.gauss(0, 0.02))
        sway = math.radians(1.5) * math.sin(2 * math.pi * 0.3 * t) if activity == "standing" else 0
        theta = base_tilt + swing * math.sin(2 * math.pi * phase) + sway   # shin angle (sagittal)
        omega = swing * 2 * math.pi * f * math.cos(2 * math.pi * phase)    # rad/s about z
        # Gravity seen in the shin frame plus centripetal/tangential motion.
        ax, ay = math.cos(theta), math.sin(theta)
        ay += 0.25 * swing * (2 * math.pi * f) ** 2 / 9.81 * -math.sin(2 * math.pi * phase)
        if impact:
            dt = (phase % 1.0) / f
            spike = impact * math.exp(-dt / 0.025) * (1 if dt < 0.15 else 0)
            ax += spike
            ay -= 0.4 * spike
        az = 0.05 * math.sin(2 * math.pi * f * 2 * t)
        gz = math.degrees(omega)
        gx = 0.15 * gz * math.sin(2 * math.pi * phase)
        gy = 0.1 * gz * math.cos(2 * math.pi * phase)
        a = rotate((ax, ay, az), *mount)
        g = rotate((gx, gy, gz), *mount)
        a = [v + rng.gauss(0, 0.03) for v in a]
        g = [v + rng.gauss(0, 1.5) for v in g]
        rows.append((int(t * 1000), *a, *g))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data_synthetic"))
    ap.add_argument("--sessions", type=int, default=4)
    args = ap.parse_args()
    rng = random.Random(0)
    for activity in ["standing", "walking", "running", "biking"]:
        d = Path(args.out) / activity
        d.mkdir(parents=True, exist_ok=True)
        for s in range(args.sessions):
            with open(d / f"{activity}_sim{s}.csv", "w") as f:
                f.write("t_ms,ax,ay,az,gx,gy,gz\n")
                for r in session(activity, rng.uniform(60, 90), rng):
                    f.write("%d,%.5f,%.5f,%.5f,%.3f,%.3f,%.3f\n" % r)
    print(f"Wrote synthetic recordings to {args.out}")


if __name__ == "__main__":
    main()
