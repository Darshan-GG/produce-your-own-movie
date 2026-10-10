"""
Iteration 006: Ridge regression with frequency-encoded crew + genres.
Baseline linear model for stacking ensemble diversity.
"""

import sys
sys.path.insert(0, '/Users/darshan/GaminGaza Webapps/produce_your_own_movie')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import time
import json
import os

from feature_template import (
    load_and_preprocess,
    CURATED_DIRECTORS, CURATED_HEROES, CURATED_HEROINES,
    CURATED_VILLAINS, CURATED_COMPOSERS,
)

OUTPUT_DIR = '/Users/darshan/GaminGaza Webapps/produce_your_own_movie/logs/iteration_006'

print("Loading data...")
start_time = time.time()

X, y, _, ids, _, _ = load_and_preprocess(as_category=False)

print(f"Dataset shape: {X.shape}, target shape: {y.shape}")
print(f"Target mean: {y.mean():.4f}, std: {y.std():.4f}")

CREW_COLS = ['director', 'hero', 'heroine', 'villain', 'music_composer']
GENRE_COLS = [c for c in X.columns if c.startswith('genre_')]

# Curated flags
X['is_curated_director'] = X['director'].isin(CURATED_DIRECTORS).astype(int)
X['is_curated_hero'] = X['hero'].isin(CURATED_HEROES).astype(int)
X['is_curated_heroine'] = X['heroine'].isin(CURATED_HEROINES).astype(int)
X['is_curated_villain'] = X['villain'].isin(CURATED_VILLAINS).astype(int)
X['is_curated_composer'] = X['music_composer'].isin(CURATED_COMPOSERS).astype(int)

CURATED_FLAG_COLS = [
    'is_curated_director', 'is_curated_hero', 'is_curated_heroine',
    'is_curated_villain', 'is_curated_composer'
]
FEATURE_COLS = CREW_COLS + GENRE_COLS + CURATED_FLAG_COLS

print(f"Total features: {len(FEATURE_COLS)}")
print(f"  Crew cols: {len(CREW_COLS)}, Genre cols: {len(GENRE_COLS)}, Curated flags: {len(CURATED_FLAG_COLS)}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = float(y.mean())

# Try 3 alpha values
alpha_results = {}

for alpha in [0.1, 1.0, 10.0]:
    print(f"\n--- Alpha = {alpha} ---")
    oof = np.zeros(len(X))
    fold_rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Frequency encode crew (train fold only — no leakage)
        for col in CREW_COLS:
            freq_map = X_tr[col].astype(str).value_counts(normalize=True).to_dict()
            X_tr[col] = X_tr[col].astype(str).map(freq_map).fillna(0.0)
            X_val[col] = X_val[col].astype(str).map(freq_map).fillna(0.0)

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[FEATURE_COLS].astype(float))
        X_val_s = scaler.transform(X_val[FEATURE_COLS].astype(float))

        # Ridge
        model = Ridge(alpha=alpha)
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_val_s)
        oof[val_idx] = preds

        fold_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold+1}: RMSE = {fold_rmse:.4f}")

    cv_rmse = float(np.sqrt(mean_squared_error(y, oof)))
    cv_std = float(np.std(fold_rmses))
    print(f"  CV RMSE = {cv_rmse:.4f} ± {cv_std:.4f}")

    alpha_results[alpha] = {
        'cv_rmse': cv_rmse,
        'cv_std': cv_std,
        'fold_rmses': fold_rmses,
        'oof': oof.tolist(),
    }

# Pick best alpha
best_alpha = min(alpha_results, key=lambda a: alpha_results[a]['cv_rmse'])
best_result = alpha_results[best_alpha]

print(f"\n=== Best alpha: {best_alpha} ===")
print(f"Best CV RMSE: {best_result['cv_rmse']:.4f} ± {best_result['cv_std']:.4f}")

duration = time.time() - start_time

# Mean baseline
mean_pred = np.full(len(y), global_mean)
mean_rmse = float(np.sqrt(mean_squared_error(y, mean_pred)))
print(f"Mean baseline RMSE: {mean_rmse:.4f}")

# Save OOF predictions (best alpha)
best_oof = np.array(alpha_results[best_alpha]['oof'])
oof_df = pd.DataFrame({
    'tmdb_id': ids.values,
    'oof_pred': best_oof,
    'actual': y.values,
})
oof_path = os.path.join(OUTPUT_DIR, 'oof_preds.csv')
oof_df.to_csv(oof_path, index=False)
print(f"Saved OOF preds: {oof_path}")

# Build result.json
all_alpha_results = {
    str(a): {
        'cv_rmse': alpha_results[a]['cv_rmse'],
        'cv_std': alpha_results[a]['cv_std'],
        'fold_rmses': alpha_results[a]['fold_rmses'],
    }
    for a in [0.1, 1.0, 10.0]
}

# Determine insight
best_rmse = best_result['cv_rmse']
gap_vs_mean = mean_rmse - best_rmse
if best_rmse < 0.90:
    status = 'success'
    key_insight = (
        f"Ridge (alpha={best_alpha}) achieves CV RMSE={best_rmse:.4f} vs mean baseline {mean_rmse:.4f}; "
        f"linear model captures {gap_vs_mean:.4f} RMSE improvement and provides diverse error patterns "
        f"suitable as a stacking base learner alongside XGBoost/LightGBM."
    )
else:
    status = 'success'  # still ran, just noting it didn't beat threshold
    key_insight = (
        f"Ridge (alpha={best_alpha}) CV RMSE={best_rmse:.4f} did not beat 0.90 threshold "
        f"(mean baseline={mean_rmse:.4f}); frequency encoding of crew provides limited linear signal — "
        f"still useful for stacking diversity but weaker than tree models."
    )

result = {
    'status': status,
    'cv_rmse': best_result['cv_rmse'],
    'cv_std': best_result['cv_std'],
    'duration_seconds': round(duration, 2),
    'model': 'Ridge',
    'n_features': len(FEATURE_COLS),
    'feature_breakdown': {
        'crew_freq_encoded': len(CREW_COLS),
        'genre_binary': len(GENRE_COLS),
        'curated_flags': len(CURATED_FLAG_COLS),
    },
    'best_alpha': best_alpha,
    'all_alpha_results': all_alpha_results,
    'mean_baseline_rmse': mean_rmse,
    'improvement_vs_baseline': round(gap_vs_mean, 4),
    'success_criterion': 'RMSE < 0.90',
    'success_criterion_met': best_rmse < 0.90,
    'key_insight': key_insight,
}

result_path = os.path.join(OUTPUT_DIR, 'result.json')
with open(result_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved result.json: {result_path}")

print("\n=== EXPERIMENT COMPLETE ===")
print(f"Status: {status}")
print(f"Best alpha: {best_alpha}")
print(f"CV RMSE: {best_result['cv_rmse']:.4f} ± {best_result['cv_std']:.4f}")
print(f"Mean baseline: {mean_rmse:.4f}")
print(f"Duration: {duration:.1f}s")
print("\nAlpha sweep:")
for a in [0.1, 1.0, 10.0]:
    r = alpha_results[a]
    print(f"  alpha={a}: RMSE={r['cv_rmse']:.4f} ± {r['cv_std']:.4f}")
