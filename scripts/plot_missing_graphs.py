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

# 8. Bug Volume Over Time
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (name, path) in enumerate(datasets.items()):
    ax = axes[i]
    if not path.exists():
        ax.set_title(f'{name} Not Found')
        continue
        
    year_counts = Counter()
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            opened = row.get('Opened', '')
            if opened and len(opened) >= 4:
                # Basic extraction assuming year is the first 4 chars or explicitly somewhere
                # The Opened format is like "2001-10-10 21:34:00" or similar
                year = opened[:4]
                if year.isdigit() and 1990 <= int(year) <= 2026:
                    year_counts[year] += 1
            
    if year_counts:
        years = sorted(year_counts.keys())
        counts = [year_counts[y] for y in years]
        
        ax.plot(years, counts, marker='o', color='#2b83ba', linewidth=2)
        ax.fill_between(years, counts, alpha=0.3, color='#2b83ba')
        ax.set_title(f'Bug Submissions Over Time: {name}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Bugs')
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(out_dir / 'bug_volume_time.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'bug_volume_time.png', bbox_inches='tight', dpi=300)
plt.close()

# 9. Component Heatmap (Top 10 components)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, (name, path) in enumerate(datasets.items()):
    ax = axes[i]
    if not path.exists():
        continue
        
    comp_counts = Counter()
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = row.get('Component', '').strip()
            if c: comp_counts[c] += 1
            
    top_comps = comp_counts.most_common(10)
    if not top_comps:
        continue
        
    comps, counts = zip(*top_comps)
    y_pos = np.arange(len(comps))
    
    # We will use a barh instead of a strict heatmap because it's cleaner for 1D distributions
    # but we can color it using a colormap to look like a heatmap
    import matplotlib.cm as cm
    norm = plt.Normalize(min(counts), max(counts))
    colors = cm.YlOrRd(norm(counts))
    
    ax.barh(y_pos, counts, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comps)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_title(f'Top 10 Components by Bug Volume: {name}')
    ax.set_xlabel('Total Bugs')

plt.tight_layout()
plt.savefig(out_dir / 'component_heatmap.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'component_heatmap.png', bbox_inches='tight', dpi=300)
plt.close()

print("Graphs 8 and 9 generated successfully.")
