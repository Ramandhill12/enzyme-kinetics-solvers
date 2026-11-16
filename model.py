"""Mass-action enzyme kinetics equations."""

import numpy as np

def michaelis_menten(t, y, k1, k_1, k2):
    """Rates of change of [S, E, C, P] under mass-action kinetics."""
    s, e, c, p = y
    bind = k1 * e * s
    unbind = k_1 * c
    cat = k2 * c
    return np.array([
        -bind + unbind,
        -bind + unbind + cat,
        bind - unbind - cat,
        cat,
    ])

PARAMS = (1.0, 0.5, 0.3)                 # k1, k_1, k2
Y0 = np.array([10.0, 1.0, 0.0, 0.0])     # S, E, C, P at t = 0
T_END = 20.0
