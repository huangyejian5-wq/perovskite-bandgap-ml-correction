import numpy as np


def is_metal_by_exp(eg_exp, threshold=0.05):
    eg_exp = np.asarray(eg_exp, dtype=float)
    return eg_exp <= float(threshold)


def is_false_metal(eg_gga, eg_exp, threshold=0.05):
    eg_gga = np.asarray(eg_gga, dtype=float)
    eg_exp = np.asarray(eg_exp, dtype=float)
    return (eg_gga <= float(threshold)) & (eg_exp > float(threshold))
