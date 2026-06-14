"""Generate the GitHub social-preview card (1280x640) — pure matplotlib, reproducible.

    python experiments/social_card.py   ->   assets/social-card.png

Clean, technical, text-crisp (no AI image generation): the claim equation, the
headline guarantee, and three claim chips. Doubles as the README hero image.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, FG, MUT, ACC = "#0d1117", "#e6edf3", "#8b949e", "#3fb950"
W, H, DPI = 12.8, 6.4, 100

fig = plt.figure(figsize=(W, H), dpi=DPI)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
rend = fig.canvas.get_renderer()


def text_w(t):
    """width of a drawn text in figure-fraction units."""
    return t.get_window_extent(rend).width / (W * DPI)


# eyebrow (left) + repo url (right)
fig.text(0.055, 0.885, "verified-circuits", color=MUT, fontsize=19,
         family="monospace", weight="bold")
fig.text(0.055, 0.825, "mechanistic interpretability, made provable",
         color=MUT, fontsize=15)
fig.text(0.945, 0.885, "github.com/brian-naughton/verified-circuits",
         color=MUT, fontsize=13, family="monospace", ha="right")

# hero equation — Spec == Circuit == Model  (== in accent green)
y = 0.585
x = 0.055
for txt, col in [("Spec", FG), ("  ==  ", ACC), ("Circuit", FG),
                 ("  ==  ", ACC), ("Model", FG)]:
    t = fig.text(x, y, txt, color=col, fontsize=50, weight="bold")
    x += text_w(t)

# subtitle + the headline guarantee
fig.text(0.055, 0.45, "a kernel-checked mechanistic circuit for a learned transformer",
         color=FG, fontsize=20)
fig.text(0.055, 0.375,
         "proved correct on every one of 65,536 inputs — and you can re-check it yourself",
         color=ACC, fontsize=18, weight="bold")

# three claim chips, sized to their actual text width, laid out left-to-right
chips = ["Lean proof · propext + Quot.sound",
         "rigorous margin ≥ +8.0950",
         "re-checkable · torch-free"]
pad, gap, cy, ch, cfs = 0.013, 0.018, 0.10, 0.082, 12.5
cx = 0.055
for c in chips:
    probe = fig.text(0, 0, c, fontsize=cfs, family="monospace")   # measure
    w = text_w(probe) + 2 * pad
    probe.remove()
    ax.add_patch(FancyBboxPatch((cx, cy), w, ch,
                 boxstyle="round,pad=0,rounding_size=0.018",
                 linewidth=1.2, edgecolor=MUT, facecolor="#161b22",
                 transform=ax.transAxes, clip_on=False))
    fig.text(cx + pad, cy + ch / 2, c, color=FG, fontsize=cfs,
             family="monospace", va="center")
    cx += w + gap

os.makedirs("assets", exist_ok=True)
fig.savefig("assets/social-card.png", facecolor=BG)
print("wrote assets/social-card.png  (1280x640)")
