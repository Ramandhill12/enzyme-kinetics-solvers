"""Euler and fourth-order Runge-Kutta solvers."""
import numpy as np

def euler_step(f, t, y, h, *args):
    """One forward Euler step."""
    return y + h * f(t, y, *args)

def solve(f, y0, t_end, n_steps, step, *args):
    """Integrate y' = f(t, y) from 0 to t_end in n_steps equal steps."""
    h = t_end / n_steps
    t = np.linspace(0.0, t_end, n_steps + 1)
    y = np.empty((n_steps + 1, len(y0)))
    y[0] = y0
    for i in range(n_steps):
        y[i + 1] = step(f, t[i], y[i], h, *args)
    return t, y

def rk4_step(f, t, y, h, *args):
    """One classical fourth-order Runge-Kutta step."""
    k1 = f(t, y, *args)
    k2 = f(t + h / 2, y + h / 2 * k1, *args)
    k3 = f(t + h / 2, y + h / 2 * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)