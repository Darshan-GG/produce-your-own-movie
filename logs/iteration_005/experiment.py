"""
Iteration 5: XGBoost with OOF target-encoded crew + curated flags + genres
Hypothesis: XGBoost may generalize differently from LightGBM on this small dataset,
useful for later stacking diversity.
"""

import sys
sys.path.insert(0, '/Users/darshan/GaminGaza Webapps/produce_your_own_movie')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import time
import json
import os

from feature_template import (
    load_and_preprocess,
    CURATED_DIRECTORS, CURATED_HEROES, CURATED_HEROINES,
    CURATED_VILLAINS, CURATED_COMPOSERS
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ──────────────────────────────────────────────────────────────
X, y, _, ids, _, _ = load_and_preprocess(as_category=False)
print(f"Dataset: {X.shape[0]} rows, target mean={y.mean():.4f}, std={y.std():.4f}")

CREW_COLS = ['director', 'hero', 'heroine', 'villain', 'music_composer']
GENRE_COLS = [c for c in X.columns if c.startswith('genre_')]

# ── Curated flags ──────────────────────────────────────────────────────────
X['is_curated_director']  = X['director'].isin(CURATED_DIRECTORS).astype(int)
X['is_curated_hero']      = X['hero'].isin(CURATED_HEROES).astype(int)
X['is_curated_heroine']   = X['heroine'].isin(CURATED_HEROINES).astype(int)
X['is_curated_villain']   = X['villain'].isin(CURATED_VILLAINS).astype(int)
X['is_curated_composer']  = X['music_composer'].isin(CURATED_COMPOSERS).astype(int)

CURATED_FLAG_COLS = [
    'is_curated_director', 'is_curated_hero', 'is_curated_heroine',
    'is_curated_villain', 'is_curated_composer'
]
FEATURE_COLS = CREW_COLS + GENRE_COLS + CURATED_FLAG_COLS

print(f"Feature set: {len(FEATURE_COLS)} features "
      f"({len(CREW_COLS)} crew TE + {len(GENRE_COLS)} genres + {len(CURATED_FLAG_COLS)} flags)")

# ── Cross-validation ───────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
global_mean = float(y.mean())
smoothing = 10.0
fold_rmses = []
best_iters = []

last_model = None

start = time.time()
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_tr  = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # OOF target encode crew columns (identical to Iter 4)
    for col in CREW_COLS:
        stats = y_tr.groupby(X_tr[col].astype(str)).agg(['sum', 'count'])
        stats['smoothed'] = (
            (stats['sum'] + smoothing * global_mean) / (stats['count'] + smoothing)
        )
        X_tr[col]  = X_tr[col].astype(str).map(stats['smoothed']).fillna(global_mean)
        X_val[col] = X_val[col].astype(str).map(stats['smoothed']).fillna(global_mean)

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='rmse',
        verbosity=0
    )
    model.fit(
        X_tr[FEATURE_COLS], y_tr,
        eval_set=[(X_val[FEATURE_COLS], y_val)],
        verbose=False
    )

    best_iters.append(int(model.best_iteration))
    preds = model.predict(X_val[FEATURE_COLS])
    oof_preds[val_idx] = preds

    fold_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    fold_rmses.append(fold_rmse)
    last_model = model
    print(f"  Fold {fold+1}: RMSE={fold_rmse:.4f}, best_iter={model.best_iteration}")

duration = time.time() - start
cv_rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
cv_std  = float(np.std(fold_rmses))

print(f"\nCV RMSE : {cv_rmse:.4f} ± {cv_std:.4f}")
print(f"Duration: {duration:.1f}s")
print(f"Best iterations: {best_iters}")

# ── Feature importance (last fold) ────────────────────────────────────────
fi = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': last_model.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("\nTop 10 features:")
print(fi.head(10).to_string(index=False))

# ── OOF predictions ───────────────────────────────────────────────────────
oof_df = pd.DataFrame({'tmdb_id': ids.values, 'oof_pred': oof_preds, 'actual': y.values})
oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_preds.csv'), index=False)

# ── Feature importance CSV ────────────────────────────────────────────────
fi.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

# ── Build key insight ─────────────────────────────────────────────────────
beat_iter1 = cv_rmse < 0.8679
beat_iter4 = cv_rmse < 0.8572
if beat_iter4:
    key_insight = (
        f"XGBoost (RMSE={cv_rmse:.4f}) beats LightGBM Iter 4 (0.8572), "
        "providing strong diversity for stacking."
    )
elif beat_iter1:
    key_insight = (
        f"XGBoost (RMSE={cv_rmse:.4f}) beats Iter 1 baseline (0.8679) "
        "but not LightGBM Iter 4 (0.8572); still useful ensemble member."
    )
else:
    key_insight = (
        f"XGBoost (RMSE={cv_rmse:.4f}) did not beat Iter 1 baseline (0.8679); "
        "may need tuning but can still offer ensemble diversity."
    )

# ── result.json ───────────────────────────────────────────────────────────
result = {
    "status": "success",
    "iteration": 5,
    "model": "xgboost",
    "cv_rmse": round(cv_rmse, 6),
    "cv_std": round(cv_std, 6),
    "fold_rmses": [round(r, 6) for r in fold_rmses],
    "duration_seconds": round(duration, 1),
    "n_features": len(FEATURE_COLS),
    "feature_cols": FEATURE_COLS,
    "best_iterations": best_iters,
    "hyperparams": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
        "random_state": 42
    },
    "top5_features": fi.head(5)[['feature', 'importance']].to_dict(orient='records'),
    "comparison": {
        "iter1_lgbm_baseline": 0.8679,
        "iter4_lgbm_best": 0.8572,
        "beat_iter1": beat_iter1,
        "beat_iter4": beat_iter4
    },
    "key_insight": key_insight
}

with open(os.path.join(OUTPUT_DIR, 'result.json'), 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nResult: {result['status']}")
print(f"Beat Iter 1 (0.8679): {beat_iter1}")
print(f"Beat Iter 4 (0.8572): {beat_iter4}")
print(f"Key insight: {key_insight}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
