"""
Iteration 10: Learning rate sweep with fixed num_leaves=31, min_child_samples=3
Objective: sweep lr=[0.01, 0.02] vs known lr=0.05 (RMSE=0.8097) to check if
lower LR + more trees (2000 with early stopping) improves generalization.
"""

import sys
sys.path.insert(0, '/Users/darshan/GaminGaza Webapps/produce_your_own_movie')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import time
import json
import os

from feature_template import (
    load_and_preprocess,
    CURATED_DIRECTORS, CURATED_HEROES, CURATED_HEROINES, CURATED_VILLAINS, CURATED_COMPOSERS
)

# ─── Load & build features (identical to Iter 8/9) ───────────────────────────
X, y, _, ids, _, _ = load_and_preprocess(as_category=True)

CREW_COLS = ['director', 'hero', 'heroine', 'villain', 'music_composer']
GENRE_COLS = [c for c in X.columns if c.startswith('genre_')]

X['is_curated_director'] = (X['director'].astype(str).isin(CURATED_DIRECTORS)).astype(int)
X['is_curated_hero']     = (X['hero'].astype(str).isin(CURATED_HEROES)).astype(int)
X['is_curated_heroine']  = (X['heroine'].astype(str).isin(CURATED_HEROINES)).astype(int)
X['is_curated_villain']  = (X['villain'].astype(str).isin(CURATED_VILLAINS)).astype(int)
X['is_curated_composer'] = (X['music_composer'].astype(str).isin(CURATED_COMPOSERS)).astype(int)

CURATED_FLAG_COLS = [
    'is_curated_director', 'is_curated_hero', 'is_curated_heroine',
    'is_curated_villain', 'is_curated_composer'
]
FEATURE_COLS = CREW_COLS + GENRE_COLS + CURATED_FLAG_COLS

print(f"Dataset: {X.shape[0]} rows, {len(FEATURE_COLS)} features")
print(f"  crew cols : {len(CREW_COLS)}")
print(f"  genre cols: {len(GENRE_COLS)}")
print(f"  curated   : {len(CURATED_FLAG_COLS)}")

# ─── Cross-validation sweep ───────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lr_results = []
best_cv    = float('inf')
best_oof   = None
best_lr    = None
best_iters_best = None

start = time.time()

for lr in [0.01, 0.02]:
    print(f"\n{'='*50}")
    print(f"Running lr={lr}  (n_estimators=2000, early_stopping=100)")
    oof_preds    = np.zeros(len(X))
    fold_rmses   = []
    fold_best_iters = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr  = X.iloc[train_idx][FEATURE_COLS].copy()
        X_val = X.iloc[val_idx][FEATURE_COLS].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Re-align categories fold-by-fold (only train cats for OOF purity)
        for col in CREW_COLS:
            cats = X_tr[col].astype(str).unique().tolist()
            X_tr[col]  = pd.Categorical(X_tr[col].astype(str),  categories=cats)
            X_val[col] = pd.Categorical(X_val[col].astype(str), categories=cats)

        model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=lr,
            num_leaves=31,
            min_child_samples=3,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_tr, y_tr,
            categorical_feature=CREW_COLS,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = model.best_iteration_
        fold_best_iters.append(best_iter)

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        fold_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        fold_rmses.append(fold_rmse)
        print(f"  fold {fold+1}: RMSE={fold_rmse:.4f}  best_iter={best_iter}")

    cv_rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
    cv_std  = float(np.std(fold_rmses))
    print(f"  => OOF RMSE={cv_rmse:.4f}  std={cv_std:.4f}  iters={fold_best_iters}")

    lr_results.append({
        'lr': lr,
        'cv_rmse': cv_rmse,
        'cv_std': cv_std,
        'fold_rmses': fold_rmses,
        'best_iters': fold_best_iters,
    })

    if cv_rmse < best_cv:
        best_cv         = cv_rmse
        best_oof        = oof_preds.copy()
        best_lr         = lr
        best_iters_best = fold_best_iters

duration = time.time() - start

# ─── Include known Iter 9 result ─────────────────────────────────────────────
ITER9_RMSE = 0.8097
lr_results.append({
    'lr': 0.05,
    'cv_rmse': ITER9_RMSE,
    'cv_std': 0.0300,
    'fold_rmses': [],
    'best_iters': [52, 44, 40, 23, 36],
    'note': 'from Iter 9 — not re-run',
})

# Determine overall best (including Iter 9's 0.05)
overall_best_rmse = best_cv
overall_best_lr   = best_lr

if ITER9_RMSE < best_cv:
    overall_best_rmse = ITER9_RMSE
    overall_best_lr   = 0.05
    # use best_oof from whichever swept lr was closest; if both worse, keep lr=0.01 for diversity
    if best_oof is None:
        best_oof = np.zeros(len(X))  # fallback (should never happen)

print(f"\n{'='*50}")
print("LR SWEEP SUMMARY")
for r in sorted(lr_results, key=lambda x: x['cv_rmse']):
    print(f"  lr={r['lr']}: RMSE={r['cv_rmse']:.4f}  std={r['cv_std']:.4f}")
print(f"Overall best: lr={overall_best_lr}, RMSE={overall_best_rmse:.4f}")
print(f"vs Iter 9 (0.8097): {'BETTER' if overall_best_rmse < ITER9_RMSE else ('SAME' if overall_best_rmse == ITER9_RMSE else 'WORSE')}")

# ─── Derive key insight ───────────────────────────────────────────────────────
best_swept = min(lr_results[:2], key=lambda x: x['cv_rmse'])
if overall_best_rmse < ITER9_RMSE:
    key_insight = (
        f"Lower LR={overall_best_lr} with 2000 trees improves over lr=0.05 — "
        f"slower convergence yields better generalization on this dataset."
    )
elif best_swept['cv_rmse'] < ITER9_RMSE + 0.005:
    key_insight = (
        "Lower LR roughly matches lr=0.05, suggesting diminishing returns from "
        "further LR tuning; dataset may be at its noise floor."
    )
else:
    key_insight = (
        f"Lower learning rates (0.01, 0.02) do NOT improve over lr=0.05 — "
        f"the dataset is likely at noise floor and converges well at 0.05."
    )

# ─── Save OOF predictions ─────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Save OOF for best swept lr (or lr=0.01 for diversity if 0.05 is still best)
oof_df = pd.DataFrame({'tmdb_id': ids.values, 'oof_pred': best_oof})
oof_df.to_csv(os.path.join(OUT_DIR, 'oof_preds.csv'), index=False)
print(f"\nSaved oof_preds.csv (from lr={best_lr if best_lr else 0.01})")

# ─── Save result.json ─────────────────────────────────────────────────────────
result = {
    'iteration': 10,
    'status': 'success',
    'cv_rmse': overall_best_rmse,
    'best_lr': overall_best_lr,
    'oof_lr': best_lr,   # lr whose OOF preds were saved
    'lr_sweep_results': [
        {k: v for k, v in r.items() if k != 'fold_rmses'} for r in lr_results
    ],
    'lr_sweep_full': lr_results,
    'vs_iter9_baseline': ITER9_RMSE,
    'beat_iter9': overall_best_rmse < ITER9_RMSE,
    'duration_seconds': round(duration, 1),
    'key_insight': key_insight,
    'config': {
        'num_leaves': 31,
        'min_child_samples': 3,
        'n_estimators': 2000,
        'early_stopping_rounds': 100,
        'n_folds': 5,
        'feature_cols': FEATURE_COLS,
    },
}

with open(os.path.join(OUT_DIR, 'result.json'), 'w') as f:
    json.dump(result, f, indent=2)

print(f"Saved result.json")
print(f"Duration: {duration:.1f}s")
