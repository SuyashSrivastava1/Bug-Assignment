import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

# Data from Table 5
projects = ['Eclipse', 'Firefox', 'Core', 'T-bird']
wsm_top1 = [19.50, 16.70, 19.70, 15.00]
ltr_top1 = [25.50, 20.30, 25.10, 18.20]
wsm_top3 = [46.50, 34.00, 37.60, 30.20]
ltr_top3 = [51.00, 39.50, 42.50, 37.80]
wsm_top5 = [59.00, 43.10, 46.10, 39.80]
ltr_top5 = [63.50, 48.20, 51.60, 46.20]
wsm_mrr = [0.3647, 0.2869, 0.3207, 0.2619]
ltr_mrr = [0.4120, 0.3307, 0.3756, 0.3041]

# Set style
plt.style.use('ggplot')
out_dir = Path('documentation/figures')
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Grouped Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(projects))
width = 0.35
ax.bar(x - width/2, wsm_top1, width, label='WSM Baseline', color='#d7191c')
ax.bar(x + width/2, ltr_top1, width, label='LTR (XGBRanker)', color='#2b83ba')
ax.set_ylabel('Top-1 Accuracy (%)')
ax.set_title('Top-1 Accuracy Comparison: WSM vs LTR')
ax.set_xticks(x)
ax.set_xticklabels(projects)
ax.legend()
plt.tight_layout()
plt.savefig(out_dir / 'perf_grouped_bar.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'perf_grouped_bar.png', bbox_inches='tight', dpi=300)
plt.close()

# 2. Relative Improvement
fig, ax = plt.subplots(figsize=(8, 5))
rel_imp = [(l - w) / w * 100 for l, w in zip(ltr_top1, wsm_top1)]
bars = ax.bar(projects, rel_imp, color='#abdda4')
ax.set_ylabel('Relative Improvement in Top-1 (%)')
ax.set_title('LTR Relative Improvement Over Baseline')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'+{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(out_dir / 'perf_relative_imp.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'perf_relative_imp.png', bbox_inches='tight', dpi=300)
plt.close()

# 3. Radar Chart (for Eclipse as an example, or all)
def plot_radar():
    categories = ['Top-1', 'Top-3', 'Top-5', 'MRR*100']
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Plot Eclipse
    idx = 0 # Eclipse
    wsm_vals = [wsm_top1[idx], wsm_top3[idx], wsm_top5[idx], wsm_mrr[idx]*100]
    ltr_vals = [ltr_top1[idx], ltr_top3[idx], ltr_top5[idx], ltr_mrr[idx]*100]
    
    wsm_vals += wsm_vals[:1]
    ltr_vals += ltr_vals[:1]
    
    ax.plot(angles, wsm_vals, linewidth=2, linestyle='solid', label='WSM (Eclipse)', color='#d7191c')
    ax.fill(angles, wsm_vals, alpha=0.1, color='#d7191c')
    ax.plot(angles, ltr_vals, linewidth=2, linestyle='solid', label='LTR (Eclipse)', color='#2b83ba')
    ax.fill(angles, ltr_vals, alpha=0.1, color='#2b83ba')
    
    plt.xticks(angles[:-1], categories)
    ax.set_title('Multi-metric Radar: Eclipse IDE')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(out_dir / 'perf_radar.pdf', bbox_inches='tight')
    plt.savefig(out_dir / 'perf_radar.png', bbox_inches='tight', dpi=300)
    plt.close()

plot_radar()

# 10. Scatter: Developers vs Accuracy
fig, ax = plt.subplots(figsize=(8, 5))
devs = [209, 854, 2033, 341]
ax.scatter(devs, ltr_top1, color='#2b83ba', s=100)
for i, txt in enumerate(projects):
    ax.annotate(txt, (devs[i], ltr_top1[i]), xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('Number of Unique Developers')
ax.set_ylabel('LTR Top-1 Accuracy (%)')
ax.set_title('Assignment Difficulty vs. Candidate Pool Size')
# Add trendline
z = np.polyfit(devs, ltr_top1, 1)
p = np.poly1d(z)
ax.plot(devs, p(devs), "r--", alpha=0.5)
plt.tight_layout()
plt.savefig(out_dir / 'scatter_devs_vs_acc.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'scatter_devs_vs_acc.png', bbox_inches='tight', dpi=300)
plt.close()

print("Performance graphs generated.")
