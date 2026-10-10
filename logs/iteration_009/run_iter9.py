import sys
sys.path.insert(0, '/Users/darshan/GaminGaza Webapps/produce_your_own_movie')
import pandas as pd, numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb, time, json, os, itertools

from feature_template import (load_and_preprocess,
    CURATED_DIRECTORS, CURATED_HEROES, CURATED_HEROINES, CURATED_VILLAINS, CURATED_COMPOSERS)

OUTPUT_DIR = '/Users/darshan/GaminGaza Webapps/produce_your_own_movie/logs/iteration_009'
os.makedirs(OUTPUT_DIR, exist_ok=True)

X, y, _, ids, _, _ = load_and_preprocess(as_category=True)
CREW_COLS = ['director','hero','heroine','villain','music_composer']
GENRE_COLS = [c for c in X.columns if c.startswith('genre_')]

X['is_curated_director'] = (X['director'].astype(str).isin(CURATED_DIRECTORS)).astype(int)
X['is_curated_hero'] = (X['hero'].astype(str).isin(CURATED_HEROES)).astype(int)
X['is_curated_heroine'] = (X['heroine'].astype(str).isin(CURATED_HEROINES)).astype(int)
X['is_curated_villain'] = (X['villain'].astype(str).isin(CURATED_VILLAINS)).astype(int)
X['is_curated_composer'] = (X['music_composer'].astype(str).isin(CURATED_COMPOSERS)).astype(int)
CURATED_FLAG_COLS = ['is_curated_director','is_curated_hero','is_curated_heroine','is_curated_villain','is_curated_composer']
FEATURE_COLS = CREW_COLS + GENRE_COLS + CURATED_FLAG_COLS

print(f"Dataset: {len(X)} rows, {len(FEATURE_COLS)} features")
print(f"CREW_COLS: {CREW_COLS}")
print(f"GENRE_COLS count: {len(GENRE_COLS)}")
print(f"CURATED_FLAG_COLS: {CURATED_FLAG_COLS}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid = list(itertools.product([31, 63, 127], [3, 5, 10]))  # num_leaves x min_child_samples
grid_results = []
best_cv = float('inf')
best_oof = None
best_config = None
best_iters_best = None

start = time.time()
for num_leaves, min_child in grid:
    print(f"\n--- num_leaves={num_leaves}, min_child_samples={min_child} ---")
    oof_preds = np.zeros(len(X))
    fold_rmses = []
    fold_best_iters = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr = X.iloc[train_idx][FEATURE_COLS].copy()
        X_val = X.iloc[val_idx][FEATURE_COLS].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        for col in CREW_COLS:
            cats = X_tr[col].astype(str).unique().tolist()
            X_tr[col] = pd.Categorical(X_tr[col].astype(str), categories=cats)
            X_val[col] = pd.Categorical(X_val[col].astype(str), categories=cats)
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=num_leaves,
            min_child_samples=min_child, random_state=42, verbose=-1
        )
        model.fit(X_tr, y_tr, categorical_feature=CREW_COLS,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        fold_best_iters.append(model.best_iteration_)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        print(f"  fold {fold+1}: RMSE={fold_rmses[-1]:.4f}, best_iter={model.best_iteration_}")
    cv_rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
    print(f"  OOF CV RMSE={cv_rmse:.4f}, std={np.std(fold_rmses):.4f}")
    grid_results.append({'num_leaves': num_leaves, 'min_child_samples': min_child,
                         'cv_rmse': cv_rmse, 'cv_std': float(np.std(fold_rmses)),
                         'best_iters': fold_best_iters})
    if cv_rmse < best_cv:
        best_cv = cv_rmse
        best_oof = oof_preds.copy()
        best_config = {'num_leaves': num_leaves, 'min_child_samples': min_child}
        best_iters_best = fold_best_iters

duration = time.time() - start
cv_std = next(r['cv_std'] for r in grid_results if r['num_leaves']==best_config['num_leaves'] and r['min_child_samples']==best_config['min_child_samples'])

print(f"\n=== BEST CONFIG: num_leaves={best_config['num_leaves']}, min_child_samples={best_config['min_child_samples']} ===")
print(f"Best OOF CV RMSE: {best_cv:.4f}")
print(f"CV Std: {cv_std:.4f}")
print(f"Best iterations: {best_iters_best}")
print(f"Duration: {duration:.1f}s")

# Save OOF predictions (best config)
oof_df = pd.DataFrame({'tmdb_id': ids.values, 'oof_pred': best_oof, 'target': y.values})
oof_df.to_csv(f'{OUTPUT_DIR}/oof_preds.csv', index=False)
print(f"OOF preds saved to {OUTPUT_DIR}/oof_preds.csv")

# Sort grid_results by cv_rmse for display
sorted_grid = sorted(grid_results, key=lambda x: x['cv_rmse'])
print("\n=== Full Grid Results (sorted by RMSE) ===")
for r in sorted_grid:
    marker = " <-- BEST" if r['num_leaves']==best_config['num_leaves'] and r['min_child_samples']==best_config['min_child_samples'] else ""
    print(f"  num_leaves={r['num_leaves']:3d}, min_child={r['min_child_samples']:2d} | RMSE={r['cv_rmse']:.4f} ± {r['cv_std']:.4f}{marker}")

# Build result.json
beat_iter8 = best_cv < 0.8179
result = {
    "iteration": 9,
    "status": "success",
    "cv_rmse": round(best_cv, 6),
    "cv_std": round(cv_std, 6),
    "duration_seconds": round(duration, 2),
    "model": "lightgbm",
    "best_config": {
        "num_leaves": best_config['num_leaves'],
        "min_child_samples": best_config['min_child_samples'],
        "n_estimators": 500,
        "learning_rate": 0.05,
        "random_state": 42,
        "early_stopping_rounds": 50
    },
    "best_iterations": best_iters_best,
    "vs_iter8_rmse": 0.8179,
    "beat_iter8": beat_iter8,
    "grid_results": sorted_grid,
    "key_insight": (
        f"Best config: num_leaves={best_config['num_leaves']}, min_child_samples={best_config['min_child_samples']}. "
        f"OOF CV RMSE={best_cv:.4f} — {'beats' if beat_iter8 else 'does not beat'} Iter 8 baseline of 0.8179."
    )
}

with open(f'{OUTPUT_DIR}/result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nresult.json saved to {OUTPUT_DIR}/result.json")
print("Done.")
