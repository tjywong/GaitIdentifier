"""Train both activity models from recorded CSVs and export them for the board.

    python training/train.py                 # reads data/, writes app/python/model_*.json
    python training/train.py --data data_synthetic

A recording's label is its folder name (data/walking/foo.csv) or, failing
that, the file-name prefix before '_' (walking_20261027-101500.csv).

Accuracy is cross-validated by recording: windows from one recording never
appear in both train and test, because overlapping windows from the same
session would otherwise leak and inflate the score.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app" / "python"))
import features  # noqa: E402
import models    # noqa: E402

LABELS = features.ACTIVITIES


# ---------------------------------------------------------------- data
def label_for(path):
    if path.parent.name in LABELS:
        return path.parent.name
    prefix = path.stem.split("_")[0]
    return prefix if prefix in LABELS else None


def load_windows(data_dir, trim_s):
    """Returns windows, labels, group ids (one per recording), and file names."""
    windows, labels, groups, names = [], [], [], []
    trim = int(trim_s * features.FS)
    for path in sorted(Path(data_dir).rglob("*.csv")):
        label = label_for(path)
        if label is None:
            print(f"  skip {path} (no activity in folder or file name)")
            continue
        with open(path) as f:
            rows = [tuple(float(r[k]) for k in ("ax", "ay", "az", "gx", "gy", "gz"))
                    for r in csv.DictReader(f)]
        rows = rows[trim:len(rows) - trim]
        n0 = len(windows)
        for start in range(0, len(rows) - features.WINDOW + 1, features.HOP):
            windows.append(rows[start:start + features.WINDOW])
            labels.append(label)
            groups.append(len(names))
        print(f"  {label:<9} {path.name}: {len(windows) - n0} windows")
        names.append(path.name)
    return windows, np.array(labels), np.array(groups), names


def splits(y, groups, n_splits=5):
    """Recording-level CV when every class has >=2 recordings, else a
    chronological split inside each recording (weaker, but honest about it)."""
    per_class = min(len(set(groups[y == c])) for c in set(y))
    if per_class >= 2:
        k = min(n_splits, per_class)
        print(f"Cross-validation: {k}-fold, grouped by recording")
        return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
                    .split(np.zeros(len(y)), y, groups))
    print("WARNING: some activity has only one recording, so the test set comes from\n"
          "         the same sessions (last 30% of each, minus one window gap).\n"
          "         Record >=2 sessions per activity for a trustworthy score.")
    train, test = [], []
    for g in set(groups):
        idx = np.flatnonzero(groups == g)
        cut = int(len(idx) * 0.7)
        train.extend(idx[:cut])
        test.extend(idx[cut + 2:])  # skip overlapping windows at the boundary
    return [(np.array(train), np.array(test))]


def cross_val_predict(make_model, X, y, folds):
    pred = np.empty(len(y), dtype=object)
    tested = np.zeros(len(y), dtype=bool)
    for tr, te in folds:
        pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
        tested[te] = True
    return pred[tested].astype(str), y[tested]


def report(title, y_true, y_pred):
    labels = [c for c in LABELS if c in set(y_true)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    lines = [f"== {title} ==",
             f"balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}",
             "confusion matrix (rows = true, cols = predicted):",
             " " * 11 + "".join(f"{c[:8]:>9}" for c in labels)]
    lines += [f"{c:<11}" + "".join(f"{v:>9}" for v in row) for c, row in zip(labels, cm)]
    lines.append(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------- model 1
def peak_tree():
    # One feature, at most 4 leaves -> at most 3 thresholds for 4 activities.
    return DecisionTreeClassifier(max_leaf_nodes=4, class_weight="balanced", random_state=0)


def export_peak(tree, direction):
    t = tree.tree_
    cuts = sorted({float(v) for v, f in zip(t.threshold, t.feature) if f >= 0})
    probes = ([cuts[0] - 1] + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])] + [cuts[-1] + 1]
              if cuts else [0.0])
    labels = [str(v) for v in tree.predict(np.array(probes).reshape(-1, 1))]
    # Drop cut points that separate two intervals with the same label.
    keep_cuts, keep_labels = [], [labels[0]]
    for cut, lab in zip(cuts, labels[1:]):
        if lab != keep_labels[-1]:
            keep_cuts.append(cut)
            keep_labels.append(lab)
    return {"type": "peak_thresholds", "direction": direction,
            "thresholds": keep_cuts, "labels": keep_labels}


# ---------------------------------------------------------------- model 2
def make_forest(n_trees):
    return lambda: RandomForestClassifier(
        n_estimators=n_trees, max_depth=10, min_samples_leaf=2,
        class_weight="balanced", random_state=0, n_jobs=-1)


def export_forest(forest, top_k=15):
    trees = []
    for est in forest.estimators_:
        t = est.tree_
        value = t.value[:, 0, :]
        value = value / value.sum(axis=1, keepdims=True)
        trees.append({
            "feature": [int(v) for v in t.feature],
            "threshold": [round(float(v), 6) for v in t.threshold],
            "left": [int(v) for v in t.children_left],
            "right": [int(v) for v in t.children_right],
            "value": [[round(float(p), 4) for p in row] for row in value],
        })
    order = np.argsort(forest.feature_importances_)[::-1][:top_k]
    return {"type": "random_forest", "classes": [str(c) for c in forest.classes_],
            "feature_names": features.FEATURE_NAMES, "trees": trees,
            "top_features": [[features.FEATURE_NAMES[i], round(float(forest.feature_importances_[i]), 4)]
                             for i in order]}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "app" / "python"))
    ap.add_argument("--trim", type=float, default=5.0,
                    help="seconds dropped from each end of a recording (putting it on, etc.)")
    ap.add_argument("--trees", type=int, default=60)
    args = ap.parse_args()

    print(f"Loading {args.data}")
    windows, y, groups, _ = load_windows(args.data, args.trim)
    if len(set(y)) < 2:
        sys.exit("Need recordings for at least two activities.")
    folds = splits(y, groups)
    out = Path(args.out)
    texts = []

    # Model 1: try every signed axis, keep the one that cross-validates best.
    print("\nModel 1: peak acceleration along one direction")
    peaks = {d: np.array([[features.peak_accel(w, d)] for w in windows]) for d in features.DIRECTIONS}
    scores = {}
    for d, X in peaks.items():
        yp, yt = cross_val_predict(peak_tree, X, y, folds)
        scores[d] = balanced_accuracy_score(yt, yp)
        print(f"  {d}: balanced accuracy {scores[d]:.3f}")
    best = max(scores, key=scores.get)
    for c in LABELS:
        if c in set(y):
            v = peaks[best][y == c, 0]
            print(f"  peak {best} for {c:<9}: median {np.median(v):5.2f} g  "
                  f"(10-90%: {np.percentile(v, 10):5.2f}-{np.percentile(v, 90):5.2f})")
    yp, yt = cross_val_predict(peak_tree, peaks[best], y, folds)
    texts.append(report(f"Model 1 (peak accel along {best}), cross-validated", yt, yp))
    spec1 = export_peak(peak_tree().fit(peaks[best], y), best)
    print("  thresholds (g):", [round(t, 3) for t in spec1["thresholds"]], "->", spec1["labels"])

    # Model 2: random forest on features from all six axes.
    print("\nModel 2: random forest on all IMU outputs")
    X = np.array([features.full_features(w) for w in windows])
    yp, yt = cross_val_predict(make_forest(args.trees), X, y, folds)
    texts.append(report("Model 2 (all IMU outputs, random forest), cross-validated", yt, yp))
    forest = make_forest(args.trees)().fit(X, y)
    spec2 = export_forest(forest)
    print("Most important features:")
    for name, imp in spec2["top_features"][:8]:
        print(f"  {name:<16} {imp:.3f}")

    # Check the exported pure-Python models reproduce sklearn exactly.
    m1, m2 = models.PeakModel(spec1), models.ForestModel(spec2)
    agree1 = np.mean([m1.predict(w) == p for w, p in
                      zip(windows, peak_tree().fit(peaks[best], y).predict(peaks[best]))])
    agree2 = np.mean([m2.predict_features(list(x)) == p for x, p in zip(X, forest.predict(X))])
    print(f"\nExported model agreement with sklearn: model 1 {agree1:.4f}, model 2 {agree2:.4f}")

    (out / "model_peak.json").write_text(json.dumps(spec1, indent=1))
    (out / "model_full.json").write_text(json.dumps(spec2, separators=(",", ":")))
    (ROOT / "training" / "report.txt").write_text(
        f"data: {args.data}\nwindows: {len(y)}  recordings: {len(set(groups))}\n\n"
        + "\n\n".join(texts) + "\nModel 1 thresholds (g): "
        + json.dumps({"direction": best, "thresholds": spec1["thresholds"], "labels": spec1["labels"]}))
    print(f"Wrote {out / 'model_peak.json'}, {out / 'model_full.json'}, training/report.txt")


if __name__ == "__main__":
    main()
