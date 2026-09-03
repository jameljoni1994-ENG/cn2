"""Generate the CN2 project banner (banner.png)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

fig = plt.figure(figsize=(16, 4.9), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1000)
ax.set_ylim(0, 300)
ax.axis("off")

# background gradient
grad = np.linspace(0.06, 0.16, 1000).reshape(1, -1)
ax.imshow(grad, extent=[0, 1000, 0, 300], aspect="auto", cmap="gray_r",
          zorder=0, alpha=0.55)

# accent underline
ax.plot([40, 620], [52, 52], color="#d62728", lw=4, solid_capstyle="round", zorder=3)

# Title
ax.text(40, 200, "CN", fontsize=88, weight="bold", color="white", zorder=4,
        fontfamily="DejaVu Sans")
ax.text(176, 205, "$^2$", fontsize=58, color="#d62728", zorder=4,
        fontfamily="DejaVu Sans")
ax.text(250, 200, "Checkpointed Newton-Nesterov", fontsize=40, weight="bold",
        color="white", zorder=4, va="baseline", fontfamily="DejaVu Sans")

# Tagline
ax.text(40, 135, "Stage-wise switching between Nesterov AGD and damped Newton",
        fontsize=22, color="#c8c8c8", zorder=4, fontfamily="DejaVu Sans")
ax.text(40, 100, "for nonconvex optimization", fontsize=22, color="#c8c8c8",
        zorder=4, fontfamily="DejaVu Sans")

# Two-phase diagram on the right
bx0, by0, bw, bh = 690, 70, 260, 170
ax.add_patch(Rectangle((bx0, by0), bw, bh, facecolor="#1a1a2e", edgecolor="#d62728",
                       lw=2, zorder=3))

# Phase L arrow (linear) -> phase S arrow (superlinear)
def arrow(x, y, dx, dy, color, z=3):
    ax.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy), arrowstyle="-|>",
                                 mutation_scale=28, lw=4, color=color, zorder=z))

arrow(bx0 + 20, by0 + 125, 70, 0, "#1f77b4")
arrow(bx0 + 120, by0 + 125, 70, 0, "#d62728")

# lambda decay curve: linear then superlinear
t = np.linspace(0, 1, 100)
lam_curve = np.where(t < 0.6, 0.9 - 0.5 * t, 0.9 - 0.5 * t)  # placeholder
# realistic: slow linear then fast superlinear collapse
lam = np.exp(-1.0 * t) - np.exp(-6.0 * np.clip(t - 0.6, 0, None))
lam = (lam - lam.min()) / (lam.max() - lam.min())
x_line = bx0 + 20 + 2.4 * (t * 90) * (130 / 130)
y_line = by0 + 30 + lam * 110.0
ax.plot(x_line, y_line, color="white", lw=3, zorder=4)
ax.text(bx0 + 20, by0 + 155, r"$\lambda = \sqrt{g^T(H+\varepsilon I)^{-1}g}$",
        fontsize=17, color="white", zorder=4, fontfamily="DejaVu Sans")
ax.text(bx0 + 20, by0 + 12, "two-phase decay:  linear (p$\\approx$1)  \u2192  superlinear (p$\\to$2)",
        fontsize=13, color="#9ad0ff", zorder=4, fontfamily="DejaVu Sans")

plt.savefig("banner.png", dpi=150, facecolor="#0d1117")
print("saved banner.png")