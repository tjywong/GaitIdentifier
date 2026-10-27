"""Feature extraction shared by the board (inference) and the laptop (training).

Pure Python on purpose: the same code runs on the UNO Q's Linux side without
numpy, and training uses the exact same numbers the board will see.

A window is a list of samples (ax, ay, az, gx, gy, gz) in g and deg/s.
"""
import math

FS = 50            # Hz, set by the sketch
WINDOW = 128       # samples (2.56 s)
HOP = 64           # samples (50 % overlap -> one prediction every 1.28 s)

ACTIVITIES = ["standing", "walking", "running", "biking"]

# Model 1 may look at the peak along exactly one signed accelerometer direction.
DIRECTIONS = ["+x", "-x", "+y", "-y", "+z", "-z"]
_AXIS = {"x": 0, "y": 1, "z": 2}


def peak_accel(window, direction):
    """Peak acceleration (g) along one signed axis, e.g. '+x' or '-z'."""
    sign = 1.0 if direction[0] == "+" else -1.0
    i = _AXIS[direction[1]]
    return max(sign * s[i] for s in window)


# ---------------------------------------------------------------- model 2

CHANNELS = ["ax", "ay", "az", "amag", "gx", "gy", "gz", "gmag"]
PER_CHANNEL = ["mean", "std", "min", "max", "range", "mad",
               "zcr", "dom_freq", "dom_frac", "jerk"]
PAIRS = [("ax", "ay"), ("ax", "az"), ("ay", "az"),
         ("gx", "gy"), ("gx", "gz"), ("gy", "gz")]

FEATURE_NAMES = ([f"{c}_{f}" for c in CHANNELS for f in PER_CHANNEL]
                 + [f"corr_{a}_{b}" for a, b in PAIRS])

# DFT bins covering 0.3-6 Hz: cadence for walking/running/pedalling lives here.
_BIN_LO = max(1, int(0.3 * WINDOW / FS))
_BIN_HI = int(6.0 * WINDOW / FS)
_TWIDDLE = {k: [(math.cos(2 * math.pi * k * n / WINDOW),
                 math.sin(2 * math.pi * k * n / WINDOW)) for n in range(WINDOW)]
            for k in range(_BIN_LO, _BIN_HI + 1)}


def _channels(window):
    ch = {"ax": [], "ay": [], "az": [], "gx": [], "gy": [], "gz": []}
    for ax, ay, az, gx, gy, gz in window:
        ch["ax"].append(ax); ch["ay"].append(ay); ch["az"].append(az)
        ch["gx"].append(gx); ch["gy"].append(gy); ch["gz"].append(gz)
    ch["amag"] = [math.sqrt(a * a + b * b + c * c)
                  for a, b, c in zip(ch["ax"], ch["ay"], ch["az"])]
    ch["gmag"] = [math.sqrt(a * a + b * b + c * c)
                  for a, b, c in zip(ch["gx"], ch["gy"], ch["gz"])]
    return ch


def _spectrum_peak(centered):
    """Dominant frequency (Hz) in the cadence band and its share of band power."""
    best_k, best_p, total = _BIN_LO, 0.0, 0.0
    for k, tw in _TWIDDLE.items():
        re = im = 0.0
        for x, (c, s) in zip(centered, tw):
            re += x * c
            im -= x * s
        p = re * re + im * im
        total += p
        if p > best_p:
            best_k, best_p = k, p
    return best_k * FS / WINDOW, (best_p / total if total > 0 else 0.0)


def _channel_features(x):
    n = len(x)
    mean = sum(x) / n
    c = [v - mean for v in x]
    var = sum(v * v for v in c) / n
    std = math.sqrt(var)
    lo, hi = min(x), max(x)
    mad = sum(abs(v) for v in c) / n
    zcr = sum(1 for a, b in zip(c, c[1:]) if (a < 0) != (b < 0)) / (n - 1)
    dom_f, dom_frac = _spectrum_peak(c)
    jerk = sum(abs(b - a) for a, b in zip(x, x[1:])) * FS / (n - 1)
    return [mean, std, lo, hi, hi - lo, mad, zcr, dom_f, dom_frac, jerk], c, std


def _corr(ca, sa, cb, sb):
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return sum(a * b for a, b in zip(ca, cb)) / (len(ca) * sa * sb)


def full_features(window):
    """Feature vector for model 2 (all IMU outputs). Order = FEATURE_NAMES."""
    ch = _channels(window)
    feats, centered, stds = [], {}, {}
    for name in CHANNELS:
        f, c, s = _channel_features(ch[name])
        feats.extend(f)
        centered[name], stds[name] = c, s
    for a, b in PAIRS:
        feats.append(_corr(centered[a], stds[a], centered[b], stds[b]))
    return feats
