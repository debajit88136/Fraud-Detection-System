import numpy as np
from collections import deque


class DriftDetector:
    def __init__(self, reference_stats, feature_columns, window_size=200, threshold=3.0):
        self.ref_mean = reference_stats["mean"]
        self.ref_std = reference_stats["std"]
        self.feature_columns = feature_columns
        self.window = deque(maxlen=window_size)
        self.threshold = threshold

    def add_transaction(self, feature_dict):
        self.window.append(feature_dict)

    def check_drift(self):
        if len(self.window) < 30:
            return {"drift_detected": False, "reason": "not enough data yet", "flagged_features": {}}

        flagged = {}
        for col in self.feature_columns:
            if col not in self.ref_mean:
                continue
            live_values = [t[col] for t in self.window if col in t]
            if not live_values:
                continue
            live_mean = np.mean(live_values)
            ref_mean = self.ref_mean[col]
            ref_std = self.ref_std.get(col, 1e-6) or 1e-6

            z_shift = abs(live_mean - ref_mean) / ref_std
            if z_shift > self.threshold:
                flagged[col] = round(float(z_shift), 3)

        return {
            "drift_detected": len(flagged) > 0,
            "flagged_features": flagged,
            "window_size": len(self.window),
        }
