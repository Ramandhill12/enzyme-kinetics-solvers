"""Check the solvers against SciPy and against the conservation laws."""

import numpy as np
from scipy.integrate import solve_ivp
from model import michaelis_menten, PARAMS, Y0, T_END
from solvers import solve, euler_step, rk4_step

N_STEPS = 400

# Check 1: compare with SciPy's high-precision solver
ref = solve_ivp(michaelis_menten, (0, T_END), Y0, args=PARAMS,
                method="DOP853", rtol=1e-13, atol=1e-13)
y_ref = ref.y[:, -1]
print("SciPy reference [S, E, C, P]:", y_ref)

for name, step in [("Euler", euler_step), ("RK4", rk4_step)]:
    t, y = solve(michaelis_menten, Y0, T_END, N_STEPS, step, *PARAMS)
    error = np.linalg.norm(y[-1] - y_ref)
    print(f"{name} final state:", y[-1], f" error: {error:.1e}")

# Check 2: conservation laws, at every time step
t, y = solve(michaelis_menten, Y0, T_END, N_STEPS, rk4_step, *PARAMS)
s, e, c, p = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

enzyme_total = e + c
substrate_total = s + c + p

print("Largest drift in E + C:    ", np.max(np.abs(enzyme_total - Y0[1])))
print("Largest drift in S + C + P:", np.max(np.abs(substrate_total - Y0[0])))
python check_solvers.py