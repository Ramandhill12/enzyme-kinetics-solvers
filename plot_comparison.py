"""Compare Euler and RK4 with a deliberately small number of steps."""

import matplotlib.pyplot as plt
from model import michaelis_menten, PARAMS, Y0, T_END
from solvers import solve, euler_step, rk4_step

N_STEPS = 100

t, y_euler = solve(michaelis_menten, Y0, T_END, N_STEPS, euler_step, *PARAMS)
t, y_rk4 = solve(michaelis_menten, Y0, T_END, N_STEPS, rk4_step, *PARAMS)

labels = ["Substrate S", "Free enzyme E", "Complex C", "Product P"]
colours = ["C0", "C1", "C2", "C3"]
for i, (label, colour) in enumerate(zip(labels, colours)):
    plt.plot(t, y_rk4[:, i], "-", color=colour, label=f"{label} (RK4)")
    plt.plot(t, y_euler[:, i], "--", color=colour, label=f"{label} (Euler)")

plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title(f"Euler (dashed) vs RK4 (solid), {N_STEPS} steps")
plt.legend(fontsize=8)
plt.savefig("figures/euler_vs_rk4.png", dpi=200)
plt.show()