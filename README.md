# Gait Identifier

An ankle-worn Arduino UNO Q + Cytron MPU6050 that classifies **standing, walking, running and biking** (real biking, or lying on your back and air-biking) with two non-neural models:

| | Input | Classifier |
|---|---|---|
| **Model 1** | Peak acceleration along **one** signed axis in a 2.56 s window, measured from the window's mean so gravity drops out and the value reflects movement, not sensor tilt | Up to 3 thresholds (a 4-leaf decision tree on that single number). Training tries all six directions (±x, ±y, ±z) and keeps the best. |
| **Model 2** | All IMU outputs: accel x/y/z, gyro x/y/z, plus their magnitudes | Random forest on 86 features: mean, std, min, max, range, mean absolute deviation, zero-crossing rate, dominant cadence frequency and its power share, and jerk for each channel, plus axis correlations |

Both models run live on the board. The LED matrix shows **model 1 on the left, model 2 on the right**: `S` standing, `W` walking, `R` running, `B` biking.

## Hardware

| MPU6050 (Cytron) | UNO Q |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | SDA |
| SCL | SCL |

Leave AD0 unconnected (I2C address 0x68). Power the UNO Q from a USB-C power bank so it can be worn.

**Mounting matters.** Strap the IMU to the outside of the ankle, just above the ankle bone. Put it in the **same orientation every time**. The recordings in `data/` were made with the board's x axis pointing down the shin (`ax` ≈ −1 g when standing). The models learn from your recordings, so a rotated sensor looks like different data.

## How it works

```
MPU6050 --I2C 50 Hz--> MCU (sketch.ino) --Bridge, 10 samples/msg--> Linux (main.py)
                          ^                                            |
                          '-------- show_activity(m1, m2) --------------'
```

- `app/sketch/sketch.ino`: sets the MPU6050 to ±16 g / ±2000 °/s (running impacts at the ankle exceed 8 g), turns on its 44 Hz low-pass filter, samples at 50 Hz, and streams raw readings.
- `app/python/main.py`: serves the control page on port 7000 and starts in classify mode (`MODE` at the top). Record mode writes CSVs; classify mode slides a 128-sample window (64-sample hop), runs both models, and smooths each with a majority vote over 3 windows.
- `app/python/features.py`: feature extraction in pure Python, used **both** by the board and by training, so the numbers can't drift apart.
- `app/python/models.py`: pure-Python inference for the exported JSON models (no numpy or scikit-learn needed on the board).
- `training/train.py`: trains both models, cross-validates, and exports `app/python/model_*.json`.

## Workflow

### 1. Record

1. Copy `app/` into App Lab as an app. On the board, apps live in `~/ArduinoApps/`.
2. Run the app and open the control page at `http://<board-ip>:7000` on your phone or laptop. Pick the **activity** and press **Record**.
3. Do that one activity, then press **Idle** (or pick the next activity, which starts a new file). Each recording is saved as `recordings/<label>_<time>.csv` in the app folder. The page shows how many seconds you have, a count per activity, and a delete button for bad takes. `MODE` / `RECORD_LABEL` at the top of `main.py` only set what happens at startup.
4. Record **at least 2 sessions (3+ is better) of 1–3 minutes per activity**, on different occasions. For biking, include air-biking sessions if you'll test with air-biking. The first and last 5 s of each file are discarded.

### 2. Train (on your laptop)

```bash
scp -r arduino@<board>.local:ArduinoApps/<app>/recordings/* data/   # or adb pull
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python training/train.py
```

Files are labelled by folder (`data/walking/…`) or by file-name prefix (`walking_….csv`). The script prints and saves to `training/report.txt`:
- the cross-validated score for each direction model 1 could use, plus the chosen thresholds in g
- confusion matrices and precision/recall for both models
- model 2's most important features
- a check that the exported pure-Python models agree with scikit-learn

Cross-validation is **grouped by recording**: windows from one session never appear in both the training and the test set. Overlapping windows from the same session would otherwise make the score look better than it is. If an activity has only one recording, the script falls back to a within-session split and warns you.

### 3. Deploy

Copy the new `model_peak.json` and `model_full.json` into the app's `python/` folder, run the app again, and press **Classify** on the control page. The page and the LED matrix show both models' predictions live. The console prints model 1's peak value and both predictions every 1.28 s.

## Synthetic data (pipeline test only)

`training/simulate.py` generates hand-made approximations of ankle signals, so the code can be tested end to end without hardware:

```bash
.venv/bin/python training/simulate.py && .venv/bin/python training/train.py --data data_synthetic
```

The `model_*.json` files in `app/python/` are trained on the real recordings in `data/`, not on this synthetic data. Retrain on your own recordings if you mount the sensor differently.

## Expected limitations

- **Model 1, standing vs. biking:** a single peak value hardly separates slow pedalling from standing. Expect most of model 1's errors here and between walking and biking. Model 2 uses gyro rotation and cadence, so it should separate them much better.
- **Real biking vs. air-biking:** these put gravity on different axes. If you'll use both, record both under `biking`.
- **Transitions:** windows spanning a change of activity are ambiguous. The 3-window vote adds about 2.5 s of lag.
