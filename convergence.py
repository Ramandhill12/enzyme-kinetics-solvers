"""Error analysis and convergence plots."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model import michaelis_menten, PARAMS, Y0, T_END
from solvers import solve, euler_step, rk4_step

y_ref = solve_ivp(michaelis_menten, (0, T_END), Y0, args=PARAMS,
                  method="DOP853", rtol=1e-13, atol=1e-13).y[:, -1]

n_steps = np.array([400, 800, 1600, 3200, 6400])
h = T_END / n_steps

for name, step, theory in [("Euler", euler_step, 1), ("RK4", rk4_step, 4)]:
    errors = np.array([
        np.linalg.norm(solve(michaelis_menten, Y0, T_END, n, step, *PARAMS)[1][-1] - y_ref)
        for n in n_steps
    ])
    order = np.polyfit(np.log(h), np.log(errors), 1)[0]
    print(f"{name}: measured order {order:.2f}")
    print("  error ratios when h halves:", np.round(errors[:-1] / errors[1:], 1))
    plt.loglog(h, errors, "o-", label=f"{name} (measured order {order:.2f})")
    plt.loglog(h, errors[0] * (h / h[0]) ** theory, "k--", linewidth=1)

plt.minorticks_off()
plt.xticks(h, [f"{x:.4f}" for x in h])
plt.xlabel("Step size h")
plt.ylabel("Error at final time")
plt.title("Convergence of Euler and RK4 (dashed: theoretical slopes 1 and 4)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/convergence.png", dpi=200)
plt.show()