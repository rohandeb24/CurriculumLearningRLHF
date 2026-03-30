import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make fonts larger globally
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

# Data in order: Length, Latent(Cluster), Entropy, Bootstrap, PEC
uniform    = [37.8, 38.5, 34.0, 34.8, 30.6]
curriculum = [56.9, 56.0, 62.2, 62.8, 65.8]
tie        = [5.3,  5.5,  3.8,  2.4,  3.6]

x = np.arange(len(categories))
width = 0.22

# Softer colors from seaborn
palette = sns.color_palette("Set2", 3)
c_curr, c_uniform, c_tie = palette

fig, ax = plt.subplots(figsize=(15, 5))  # slightly taller for big fonts

bars_uniform = ax.bar(x - width, uniform, width,
                      label="Uniform", color=c_uniform)
bars_curr = ax.bar(x, curriculum, width,
                   label="Curriculum", color=c_curr)
bars_tie = ax.bar(x + width, tie, width,
                  label="Tie", color=c_tie)

ax.set_ylabel("Percentage of Responses Preferred", fontsize=18)
ax.set_title("Anthropic/hh-rlhf", fontsize=22)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=0, ha="center", fontsize=16)
ax.set_ylim(0, 100)

# Bigger tick labels (in case)
ax.tick_params(axis="both", which="major", labelsize=16)

# Move legend outside so it does not cover labels
ax.legend(
    title="Method",
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    borderaxespad=0.,
    fontsize=18,
    title_fontsize=18,
)

# Add percentage labels on bars
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
plt.savefig("anthropic_hh_rlhf_barplot.pdf", bbox_inches="tight")
plt.show()
