import matplotlib.pyplot as plt
import csv
from collections import Counter
from pathlib import Path
import numpy as np

out_dir = Path('documentation/figures')
out_dir.mkdir(parents=True, exist_ok=True)
plt.style.use('ggplot')

datasets = {
    'Eclipse': Path('data/eclipse/final dataset for work ecllipse.csv'),
    'Firefox': Path('data/mozilla_firefox/mozilla_firefox.csv'),
    'Core': Path('data/mozilla_core/mozilla_core.csv'),
    'T-bird': Path('data/thunderbird/thunderbird.csv'),
}

# 6. Developer Long-Tail Distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, (name, path) in enumerate(datasets.items()):
    ax = axes[i]
    if not path.exists():
        ax.set_title(f'{name} Not Found')
        continue
        
    dev_counts = Counter()
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = (row.get('Assignee Real Name') or row.get('Assignee', '')).strip()
            if a: dev_counts[a] += 1
            
    counts = sorted(list(dev_counts.values()), reverse=True)
    x = np.arange(len(counts))
    
    # Plot power law
    ax.plot(x, counts, color='#d7191c', linewidth=2)
    ax.fill_between(x, counts, alpha=0.3, color='#d7191c')
    ax.set_yscale('log')
    ax.set_title(f'Developer Activity Long Tail: {name}')
    ax.set_xlabel('Developer Rank')
    ax.set_ylabel('Bugs Fixed (Log Scale)')

plt.tight_layout()
plt.savefig(out_dir / 'dev_long_tail.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'dev_long_tail.png', bbox_inches='tight', dpi=300)
plt.close()

# 7. Severity Distribution
fig, ax = plt.subplots(figsize=(10, 6))
severities = ['critical', 'normal', 'minor']
colors = ['#d7191c', '#fdae61', '#abdda4']
width = 0.2
x = np.arange(len(datasets))

all_sev_counts = {sev: [] for sev in severities}

for name, path in datasets.items():
    sev_counts = {'critical': 0, 'normal': 0, 'minor': 0}
    total = 0
    if path.exists():
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # We map severity similarly to our pipeline
                s = row.get('Severity', '').lower()
                if s in ['blocker', 'critical', 'major']: mapped = 'critical'
                elif s in ['trivial', 'minor', 'enhancement']: mapped = 'minor'
                else: mapped = 'normal'
                sev_counts[mapped] += 1
                total += 1
    
    for sev in severities:
        pct = (sev_counts[sev] / total * 100) if total > 0 else 0
        all_sev_counts[sev].append(pct)

for i, sev in enumerate(severities):
    offset = (i - 1) * width
    ax.bar(x + offset, all_sev_counts[sev], width, label=sev.capitalize(), color=colors[i])

ax.set_ylabel('Percentage of Bugs (%)')
ax.set_title('Bug Severity Distribution Across Projects')
ax.set_xticks(x)
ax.set_xticklabels(datasets.keys())
ax.legend()

plt.tight_layout()
plt.savefig(out_dir / 'severity_distribution.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'severity_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

print("Data distribution graphs generated.")
