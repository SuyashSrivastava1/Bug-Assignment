import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data
projects = ['Eclipse', 'Firefox', 'Core', 'T-bird']
devs = [209, 854, 2033, 341]
comps = [23, 50, 179, 22]
avg_bugs = [47.8, 18.0, 42.1, 15.3]
singletons = [24.4, 42.2, 35.2, 47.5]
top10 = [64.6, 53.7, 49.3, 72.6]

# Set style
plt.style.use('ggplot')
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
x = np.arange(len(projects))
width = 0.35

# Subplot 1: Pool Size and Granularity (Log Scale)
ax1 = axes[0]
rects1 = ax1.bar(x - width/2, devs, width, label='Developers', color='#2b83ba')
rects2 = ax1.bar(x + width/2, comps, width, label='Components', color='#d7191c')
ax1.set_ylabel('Count (Log Scale)')
ax1.set_title('Candidate Pool & Component Granularity')
ax1.set_xticks(x)
ax1.set_xticklabels(projects)
ax1.set_yscale('log')
ax1.legend()

# Subplot 2: Sparsity & Concentration (%)
ax2 = axes[1]
rects3 = ax2.bar(x - width/2, singletons, width, label='Singleton Devs (%)', color='#fdae61')
rects4 = ax2.bar(x + width/2, top10, width, label='Top-10 Devs (%)', color='#abdda4')
ax2.set_ylabel('Percentage (%)')
ax2.set_title('Sparsity & Concentration')
ax2.set_xticks(x)
ax2.set_xticklabels(projects)
ax2.set_ylim(0, 100)
ax2.legend()

# Subplot 3: Training Signal
ax3 = axes[2]
rects5 = ax3.bar(x, avg_bugs, width*1.5, color='#4575b4')
ax3.set_ylabel('Bugs per Developer')
ax3.set_title('Average Training Signal Density')
ax3.set_xticks(x)
ax3.set_xticklabels(projects)

# Add values on top of bars
def autolabel(rects, ax, fmt='%.1f'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(fmt % height if type(height) == float else str(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1, ax1, '%d')
autolabel(rects2, ax1, '%d')
autolabel(rects3, ax2, '%.1f%%')
autolabel(rects4, ax2, '%.1f%%')
autolabel(rects5, ax3, '%.1f')

plt.tight_layout()

output_path = Path('documentation/figures/dataset_characteristics.pdf')
plt.savefig(output_path, format='pdf', bbox_inches='tight')
plt.savefig('documentation/figures/dataset_characteristics.png', format='png', dpi=300, bbox_inches='tight')
print(f"Figures saved to {output_path}")
