"""Plot the four concentrations against time."""

import matplotlib.pyplot as plt
from model import michaelis_menten, PARAMS, Y0, T_END
from solvers import solve, euler_step

t, y = solve(michaelis_menten, Y0, T_END, 400, euler_step, *PARAMS)

labels = ["Substrate S", "Free enzyme E", "Complex C", "Product P"]
for i, label in enumerate(labels):
    plt.plot(t, y[:, i], label=label)

plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Enzyme kinetics, forward Euler (400 steps)")
plt.legend()
plt.savefig("figures/concentrations.png", dpi=200)
plt.show()