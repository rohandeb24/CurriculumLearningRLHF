import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make all fonts bigger globally
sns.set(style="whitegrid", font_scale=1.6)
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 18

# Categories (requested order)
categories = [
    "Prompt length",
    "Latent structure",
    "Entropy",
    "Bootstrap ensemble",
    "Priority Experience",
]

# Data
uniform    = [33.6, 32.5, 30.0, 29.6, 28.1]
curriculum = [53.0, 56.3, 68.4, 69.2, 70.2]
tie        = [13.4, 11.3, 1.6, 1.2, 1.7]  # sums ~ 100

x = np.arange(len(categories))
width = 0.22

palette = sns.color_palette("Set2", 3)
c_curr, c_uniform, c_tie = palette

fig, ax = plt.subplots(figsize=(15, 5))

bars_uniform = ax.bar(x - width, uniform, width,
                      label="Uniform", color=c_uniform)
bars_curr = ax.bar(x, curriculum, width,
                   label="Curriculum", color=c_curr)
bars_tie = ax.bar(x + width, tie, width,
                  label="Tie", color=c_tie)

ax.set_ylabel("Percentage of Responses Preferred", fontsize=18)
ax.set_title("stanfordnlp/SHP", fontsize=22)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=0, ha="center", fontsize=16)
ax.set_ylim(0, 100)

# Ensure ticks are big enough
ax.tick_params(axis="both", which="major", labelsize=16)

ax.legend(
    title="Method",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0.,
    fontsize=18,
    title_fontsize=18,
)

def autolabel(bars):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=13,
        )

for bars in [bars_uniform, bars_curr, bars_tie]:
    autolabel(bars)

fig.tight_layout()

# Save as PDF
plt.savefig("stanford_shp_barplot.pdf", bbox_inches="tight")

plt.show()
