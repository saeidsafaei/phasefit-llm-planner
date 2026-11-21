import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2 = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

def box(x, y, w, h, title, body, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc="white", ec=col, lw=1.4))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.6, color=INK, weight="bold")
    ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=6.4, color=INK2, linespacing=1.35)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=INK2))

box(1, 31, 21, 17, "Inputs", "model config\nworkload trace\np99 TTFT / TPOT\naccelerators, inventory\nallowed precisions", INK2)
box(1, 3, 21, 22, "Calibration (once)", "22 public TensorRT-LLM\nmeasurements\n→ η, η₄, M½, τ, κ\nvalidated on 32 held-out\npoints (other model,\nvendor, MLPerf)", VIOLET)
box(27, 25, 22, 23, "① Enumerate + ② model", "configs (h, q, t)\nh ∈ 7 accelerators\nq ∈ {BF16, FP8, FP4}\nt ∈ {1,2,4,8} / DEP{4,8}\niteration time Eq.(5)\npower Eq.(11)", BLUE)
box(54, 31, 21, 17, "③a Prefill sizing", "screen: p99 prompt\nwithin TTFT\nsimulate → min n\nmeeting p99 TTFT", ORANGE)
box(54, 9, 21, 17, "③b Decode sizing", "batch cap from TPOT\nLittle's-law estimate\nsimulate → min n\nmeeting p99 TPOT", AQUA)
box(79, 17, 20, 25, "④ Combine", "pairing policy Π\n(same vendor? same\nprecision? FP4?)\ninventory limits /\nmixed pools (β, f)\n→ min energy\n→ end-to-end check", INK)
arrow(22, 39.5, 27, 39.5)
arrow(22, 14, 34, 25)
arrow(49, 40, 54, 40)
arrow(49, 30, 54, 20)
arrow(75, 39.5, 79, 33)
arrow(75, 17.5, 79, 24)
ax.text(64.5, 28.5, "independent\n(Eq. 15)", ha="center", va="center", fontsize=6.2, color=INK2, style="italic")
fig.tight_layout()
fig.savefig("../figs/fig_overview.png", dpi=300); fig.savefig("../figs/fig_overview.pdf")
