"""Inference for the two exported (non-neural) models. Pure Python, no numpy.

Both JSON files are written by training/train.py.
"""
import bisect
import json

from features import full_features, peak_accel


class PeakModel:
    """Model 1: thresholds on the peak acceleration along one signed axis.

    The trained decision tree is flattened to sorted cut points; interval i
    (between thresholds[i-1] and thresholds[i]) maps to labels[i].
    """

    def __init__(self, spec):
        # Older exports used the raw peak (gravity included); their thresholds
        # don't fit features.peak_accel any more.
        if spec.get("feature") != "peak_minus_mean":
            raise ValueError("model_peak.json is from the old raw-peak feature; "
                             "retrain with training/train.py")
        self.direction = spec["direction"]
        self.thresholds = spec["thresholds"]
        self.labels = spec["labels"]

    def feature(self, window):
        return peak_accel(window, self.direction)

    def predict(self, window):
        # bisect_left keeps sklearn's rule: value <= threshold goes left.
        return self.labels[bisect.bisect_left(self.thresholds, self.feature(window))]


class ForestModel:
    """Model 2: random forest over features from all six IMU axes."""

    def __init__(self, spec):
        self.classes = spec["classes"]
        self.trees = spec["trees"]

    @staticmethod
    def _leaf(tree, x):
        node = 0
        left, right = tree["left"], tree["right"]
        feat, thr = tree["feature"], tree["threshold"]
        while left[node] != -1:
            node = left[node] if x[feat[node]] <= thr[node] else right[node]
        return tree["value"][node]

    def predict_proba_features(self, x):
        votes = [0.0] * len(self.classes)
        for tree in self.trees:
            for i, p in enumerate(self._leaf(tree, x)):
                votes[i] += p
        n = len(self.trees)
        return [v / n for v in votes]

    def predict_features(self, x):
        proba = self.predict_proba_features(x)
        return self.classes[max(range(len(proba)), key=proba.__getitem__)]

    def predict(self, window):
        return self.predict_features(full_features(window))


def load(path):
    with open(path) as f:
        spec = json.load(f)
    return PeakModel(spec) if spec["type"] == "peak_thresholds" else ForestModel(spec)
