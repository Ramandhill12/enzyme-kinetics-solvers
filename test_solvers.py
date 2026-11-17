"""Tests for the solvers and model."""

import numpy as np
from scipy.integrate import solve_ivp
from model import michaelis_menten, PARAMS, Y0, T_END
from solvers import solve, euler_step, rk4_step


def decay(t, y):
    """y' = -y, whose exact solution is exp(-t)."""
    return -y


def test_rk4_exponential_decay():
    """RK4 matches the exact solution of y' = -y."""
    _, y = solve(decay, np.array([1.0]), 1.0, 100, rk4_step)
    assert abs(y[-1, 0] - np.exp(-1.0)) < 1e-9


def test_rk4_is_fourth_order():
    """Halving the step cuts RK4's error by about 16."""
    errors = []
    for n in (10, 20):
        _, y = solve(decay, np.array([1.0]), 1.0, n, rk4_step)
        errors.append(abs(y[-1, 0] - np.exp(-1.0)))
    assert 14 < errors[0] / errors[1] < 18


def test_euler_is_first_order():
    """Halving the step cuts Euler's error by about 2."""
    errors = []
    for n in (100, 200):
        _, y = solve(decay, np.array([1.0]), 1.0, n, euler_step)
        errors.append(abs(y[-1, 0] - np.exp(-1.0)))
    assert 1.9 < errors[0] / errors[1] < 2.1


def test_enzyme_is_conserved():
    """Free plus bound enzyme stays equal to the starting enzyme."""
    _, y = solve(michaelis_menten, Y0, T_END, 400, rk4_step, *PARAMS)
    assert np.allclose(y[:, 1] + y[:, 2], Y0[1])


def test_substrate_is_conserved():
    """Substrate, complex and product always add up to the starting substrate."""
    _, y = solve(michaelis_menten, Y0, T_END, 400, rk4_step, *PARAMS)
    assert np.allclose(y[:, 0] + y[:, 2] + y[:, 3], Y0[0])


def test_rk4_matches_scipy():
    """RK4 on the enzyme model agrees with SciPy's high-precision solver."""
    y_ref = solve_ivp(michaelis_menten, (0, T_END), Y0, args=PARAMS,
                      method="DOP853", rtol=1e-13, atol=1e-13).y[:, -1]
    _, y = solve(michaelis_menten, Y0, T_END, 400, rk4_step, *PARAMS)
    assert np.linalg.norm(y[-1] - y_ref) < 1e-5