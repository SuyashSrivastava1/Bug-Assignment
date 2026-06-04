import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
import json

out_dir = Path('documentation/figures')
out_dir.mkdir(parents=True, exist_ok=True)
plt.style.use('ggplot')

models = {
    'Eclipse': 'models/ltr_ranker_eclipse.json',
    'Firefox': 'models/ltr_ranker_firefox.json',
    'Core': 'models/ltr_ranker_core.json',
    'Thunderbird': 'models/ltr_ranker_thunderbird.json'
}

feature_names = [
    "experience", "fix_time", "success_rate", "workload", "domain_skill", "knn_affinity",
    "component_match", "num_past_bugs_fixed", "cosine_sim_max", "cosine_sim_mean",
    "severity_is_critical", "severity_is_normal", "severity_is_minor",
    "proto_experience", "proto_fix_time", "proto_success_rate", "proto_workload", "proto_domain_skill", "proto_knn_affinity"
]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, (proj, path) in enumerate(models.items()):
    ax = axes[i]
    if not Path(path).exists():
        ax.set_title(f'{proj} Model Not Found')
        continue
    
    # Load model
    booster = xgb.Booster()
    booster.load_model(path)
    booster.feature_names = feature_names
    
    # Get importance
    importance = booster.get_score(importance_type='gain')
    sorted_idx = sorted(importance, key=importance.get, reverse=True)[:10] # Top 10
    
    y_pos = range(len(sorted_idx))
    scores = [importance[f] for f in sorted_idx]
    
    # Plot
    ax.barh(y_pos, scores, color='#4575b4')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_idx)
    ax.invert_yaxis()
    ax.set_title(f'Feature Importance (Gain): {proj}')
    ax.set_xlabel('F-Score (Gain)')

plt.tight_layout()
plt.savefig(out_dir / 'feature_importance_all.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'feature_importance_all.png', bbox_inches='tight', dpi=300)
plt.close()

print("Feature importance graph generated.")
