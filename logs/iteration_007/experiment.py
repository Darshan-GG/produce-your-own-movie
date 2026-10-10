"""
Iteration 7 — XGBoost with increased regularization (shallower trees, lower LR).
Hypothesis: reduce fold variance from Iter 5 (std=0.040) while beating RMSE=0.8440.
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
    CURATED_VILLAINS, CURATED_COMPOSERS,
)

OUTPUT_DIR = '/Users/darshan/GaminGaza Webapps/produce_your_own_movie/logs/iteration_007'

# ── Load data ──────────────────────────────────────────────────────────────────
X, y, _, ids, _, _ = load_and_preprocess(as_category=False)

CREW_COLS = ['director', 'hero', 'heroine', 'villain', 'music_composer']
GENRE_COLS = [c for c in X.columns if c.startswith('genre_')]

# Curated flags (same as Iter 5)
X['is_curated_director'] = X['director'].isin(CURATED_DIRECTORS).astype(int)
X['is_curated_hero']     = X['hero'].isin(CURATED_HEROES).astype(int)
X['is_curated_heroine']  = X['heroine'].isin(CURATED_HEROINES).astype(int)
X['is_curated_villain']  = X['villain'].isin(CURATED_VILLAINS).astype(int)
X['is_curated_composer'] = X['music_composer'].isin(CURATED_COMPOSERS).astype(int)

CURATED_FLAG_COLS = [
    'is_curated_director', 'is_curated_hero', 'is_curated_heroine',
    'is_curated_villain', 'is_curated_composer',
]
FEATURE_COLS = CREW_COLS + GENRE_COLS + CURATED_FLAG_COLS

# ── CV setup ───────────────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds    = np.zeros(len(X))
global_mean  = float(y.mean())
smoothing    = 10.0
fold_rmses   = []
best_iters   = []
fold_importances = []

start = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr  = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_tr  = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # OOF target encoding (no leakage)
    for col in CREW_COLS:
        stats = y_tr.groupby(X_tr[col].astype(str)).agg(['sum', 'count'])
        stats['smoothed'] = (
            (stats['sum'] + smoothing * global_mean) / (stats['count'] + smoothing)
        )
        X_tr[col]  = X_tr[col].astype(str).map(stats['smoothed']).fillna(global_mean)
        X_val[col] = X_val[col].astype(str).map(stats['smoothed']).fillna(global_mean)

    # Iter 7 hyperparams: shallower trees + more aggressive regularisation
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='rmse',
        verbosity=0,
    )
    model.fit(
        X_tr[FEATURE_COLS], y_tr,
        eval_set=[(X_val[FEATURE_COLS], y_val)],
        verbose=False,
    )

    best_iters.append(int(model.best_iteration))
    preds = model.predict(X_val[FEATURE_COLS])
    oof_preds[val_idx] = preds
    fold_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    fold_rmses.append(fold_rmse)
    print(f"  Fold {fold+1}/5 | best_iter={model.best_iteration:4d} | RMSE={fold_rmse:.6f}")

    # Collect importances (gain)
    imp = model.get_booster().get_score(importance_type='gain')
    fold_importances.append(imp)

duration  = time.time() - start
cv_rmse   = float(np.sqrt(mean_squared_error(y, oof_preds)))
cv_std    = float(np.std(fold_rmses))

print(f"\nCV RMSE : {cv_rmse:.6f}")
print(f"CV Std  : {cv_std:.6f}")
print(f"Duration: {duration:.1f}s")

# ── Feature importance (mean gain across folds) ────────────────────────────────
all_keys = set()
for d in fold_importances:
    all_keys.update(d.keys())

mean_imp = {}
for k in all_keys:
    vals = [d.get(k, 0.0) for d in fold_importances]
    mean_imp[k] = float(np.mean(vals))

imp_df = (
    pd.DataFrame.from_dict(mean_imp, orient='index', columns=['importance'])
    .sort_values('importance', ascending=False)
    .reset_index()
    .rename(columns={'index': 'feature'})
)
# Normalise to sum=1
total = imp_df['importance'].sum()
if total > 0:
    imp_df['importance'] = imp_df['importance'] / total

top5 = imp_df.head(5).to_dict('records')

# ── Save outputs ───────────────────────────────────────────────────────────────
oof_df = pd.DataFrame({'tmdb_id': ids.values, 'oof_pred': oof_preds, 'actual': y.values})
oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_preds.csv'), index=False)

imp_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

iter5_rmse = 0.8440
delta = cv_rmse - iter5_rmse
comparison = 'better' if delta < -0.0001 else ('worse' if delta > 0.0001 else 'same')

result = {
    'status': 'success',
    'iteration': 7,
    'model': 'xgboost',
    'cv_rmse': round(cv_rmse, 6),
    'cv_std': round(cv_std, 6),
    'fold_rmses': [round(r, 6) for r in fold_rmses],
    'duration_seconds': round(duration, 1),
    'n_features': len(FEATURE_COLS),
    'feature_cols': FEATURE_COLS,
    'best_iterations': best_iters,
    'hyperparams': {
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'max_depth': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'early_stopping_rounds': 50,
        'random_state': 42,
    },
    'top5_features': top5,
    'comparison_to_iter5': {
        'iter5_cv_rmse': iter5_rmse,
        'iter5_cv_std': 0.040387,
        'delta_rmse': round(delta, 6),
        'verdict': comparison,
        'std_reduced': cv_std < 0.040387,
    },
    'key_insight': (
        f"Shallower trees (depth=3) + lower LR (0.03) + stronger subsampling: "
        f"RMSE={cv_rmse:.4f} vs Iter5={iter5_rmse:.4f} "
        f"({'improved' if delta < 0 else 'regressed'}), "
        f"std={cv_std:.4f} vs Iter5 std=0.0404."
    ),
}

with open(os.path.join(OUTPUT_DIR, 'result.json'), 'w') as f:
    json.dump(result, f, indent=2)

print("\nresult.json written.")
print(f"Top-5 features: {[t['feature'] for t in top5]}")
