import numpy as np


def apply_lower_bound(pred, gga_gap, margin=0.1, mask=None):
    pred = np.asarray(pred, dtype=float)
    gga_gap = np.asarray(gga_gap, dtype=float)
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    bounded = pred.copy()
    bounded[mask] = np.maximum(bounded[mask], gga_gap[mask] - float(margin))
    return bounded
