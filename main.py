import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры
n0 = 1.61
beta = 0.45
alpha0_deg = 60
y0 = 0
length_factor = 2.5
num_points = 1000

alpha0 = np.deg2rad(alpha0_deg)
period = 2 * np.pi / beta
x_max = length_factor * period
x = np.linspace(0, x_max, num_points)

# Аналитическое решение
y_analytical = y0 * np.cos(beta * x) + (alpha0 / beta) * np.sin(beta * x)
alpha_analytical = -y0 * beta * np.sin(beta * x) + alpha0 * np.cos(beta * x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
ax1.plot(x, y_analytical, 'b', linewidth=2, label='α₀={}°'.format(alpha0_deg))
ax1.set_xlabel("x (мм)"); ax1.set_ylabel("y (мм)")
ax1.set_title("Траектория луча (аналитически)")
ax1.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax1.grid(True, alpha=0.3); ax1.legend()

ax2.plot(y_analytical, np.rad2deg(alpha_analytical), 'r', linewidth=2)
ax2.set_xlabel("y (мм)"); ax2.set_ylabel("α (град)")
ax2.set_title("Фазовый портрет (α vs y)")
ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax2.axvline(0, color='k', linestyle='--', linewidth=0.8)
ax2.grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig('images/waveguide_analytical.png')
plt.close()

# Численное решение
def beam_eq(x, Y):
    y, alpha = Y
    return [alpha, -beta**2 * y]

sol = solve_ivp(beam_eq, [0, x_max], [y0, alpha0], t_eval=x)
y_numerical = sol.y[0]
alpha_numerical = sol.y[1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
ax1.plot(x, y_numerical, 'b', linewidth=2, label='α₀={}°'.format(alpha0_deg))
ax1.set_xlabel("x (мм)"); ax1.set_ylabel("y (мм)")
ax1.set_title("Траектория луча (численно)")
ax1.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax1.grid(True, alpha=0.3); ax1.legend()

ax2.plot(y_numerical, np.rad2deg(alpha_numerical), 'r', linewidth=2)
ax2.set_xlabel("y (мм)"); ax2.set_ylabel("α (град)")
ax2.set_title("Фазовый портрет (α vs y)")
ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax2.axvline(0, color='k', linestyle='--', linewidth=0.8)
ax2.grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig('images/waveguide_numerical.png')
plt.close()

# Результаты
print(f"Период траектории: {period:.2f} мм")
print(f"Фокусное расстояние: {1/beta:.2f} мм")
