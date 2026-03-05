import matplotlib.pyplot as plt

# Data points for NCF Convergence
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
recall_scores = [0.3966, 0.4492, 0.5305, 0.5669, 0.5903, 0.6150, 0.6367, 0.6539, 0.6677, 0.6754]

# Configuration for publication-style aesthetics
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5
})

# Create figure with 16:9 aspect ratio
# Setting figsize to 10x5.625 inches at 300 DPI results in a 3000x1688px image
fig, ax = plt.subplots(figsize=(10, 5.625), dpi=300)

# Plotting the data
ax.plot(epochs, recall_scores,
        color="#1d4ed8",  # High-contrast professional blue
        marker="o",
        markersize=10,
        linewidth=3,
        markerfacecolor="white",
        markeredgewidth=2.5,
        zorder=3)

# Title and Labels
ax.set_title("NCF Convergence on Fixed Split (MSD, filtered)", pad=30, fontweight="bold")
ax.set_xlabel("Epoch", labelpad=15)
ax.set_ylabel("Recall@10", labelpad=15)

# Axis Configuration
ax.set_xticks(epochs)
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
ax.set_ylim(0.35, 0.70)

# Grid and Spines
ax.grid(True, which='major', axis='both', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Final adjustment and save
plt.tight_layout()
plt.savefig("ncf_convergence_fixed_split.png", bbox_inches="tight", dpi=300)
print("Success: High-resolution PNG saved as 'ncf_convergence_fixed_split.png'")