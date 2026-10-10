"""
Iteration 11 — Film Count Features
Phase 2, regression, LightGBM
Adds 5 film-count features (director/hero/heroine/villain/composer)
computed from training fold only to avoid leakage.
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
    CURATED_DIRECTORS, CURATED_HEROES, CURATED_HEROINES,
    CURATED_VILLAINS, CURATED_COMPOSERS
)

OUTPUT_DIR = '/Users/darshan/GaminGaza Webapps/produce_your_own_movie/logs/iteration_011'

# ── Load base features ──────────────────────────────────────────────────────
X, y, _, ids, _, _ = load_and_preprocess(as_category=True)

CREW_COLS = ['director', 'hero', 'heroine', 'villain', 'music_composer']
GENRE_COLS = [c for c in X.columns if c.startswith('genre_')]

# Curated flags (computed globally — binary membership, no leakage risk)
X['is_curated_director']  = X['director'].astype(str).isin(CURATED_DIRECTORS).astype(int)
X['is_curated_hero']      = X['hero'].astype(str).isin(CURATED_HEROES).astype(int)
X['is_curated_heroine']   = X['heroine'].astype(str).isin(CURATED_HEROINES).astype(int)
X['is_curated_villain']   = X['villain'].astype(str).isin(CURATED_VILLAINS).astype(int)
X['is_curated_composer']  = X['music_composer'].astype(str).isin(CURATED_COMPOSERS).astype(int)

CURATED_FLAG_COLS = [
    'is_curated_director', 'is_curated_hero', 'is_curated_heroine',
    'is_curated_villain', 'is_curated_composer'
]
COUNT_COLS = [
    'n_films_director', 'n_films_hero', 'n_films_heroine',
    'n_films_villain', 'n_films_composer'
]

# Placeholder columns — will be filled per fold
for col in COUNT_COLS:
    X[col] = 0.0

FEATURE_COLS = CREW_COLS + GENRE_COLS + CURATED_FLAG_COLS + COUNT_COLS

print(f"Dataset: {len(X)} rows, {len(FEATURE_COLS)} features")
print(f"Target mean={y.mean():.4f}, std={y.std():.4f}")

# ── Cross-validation ─────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds  = np.zeros(len(X))
fold_rmses = []
best_iters = []
fi_accum   = None   # accumulate importances across all folds

start = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr  = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # ── Film counts from training fold only ──────────────────────────────────
    crew_raw = ['director', 'hero', 'heroine', 'villain', 'music_composer']
    for col, cnt_col in zip(crew_raw, COUNT_COLS):
        count_map = X_tr[col].astype(str).value_counts().to_dict()
        X_tr[cnt_col]  = X_tr[col].astype(str).map(count_map).fillna(1).astype(float)
        X_val[cnt_col] = X_val[col].astype(str).map(count_map).fillna(1).astype(float)

    # ── Fix categories per fold ──────────────────────────────────────────────
    for col in CREW_COLS:
        cats = X_tr[col].astype(str).unique().tolist()
        X_tr[col]  = pd.Categorical(X_tr[col].astype(str),  categories=cats)
        X_val[col] = pd.Categorical(X_val[col].astype(str), categories=cats)

    # ── Train ────────────────────────────────────────────────────────────────
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=3,
        random_state=42,
        verbose=-1
    )
    model.fit(
        X_tr[FEATURE_COLS], y_tr,
        categorical_feature=CREW_COLS,
        eval_set=[(X_val[FEATURE_COLS], y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    )

    best_iters.append(int(model.best_iteration_))
    preds = model.predict(X_val[FEATURE_COLS])
    oof_preds[val_idx] = preds

    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    fold_rmses.append(rmse)
    print(f"  Fold {fold+1} RMSE={rmse:.5f}  best_iter={model.best_iteration_}")

    # Accumulate feature importances
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    if fi_accum is None:
        fi_accum = fi.copy()
    else:
        fi_accum = fi_accum + fi

duration = time.time() - start

# ── Metrics ──────────────────────────────────────────────────────────────────
cv_rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
cv_std  = float(np.std(fold_rmses))

print(f"\nCV RMSE: {cv_rmse:.5f}  ± {cv_std:.5f}")
print(f"Duration: {duration:.1f}s")

# ── Save OOF predictions ─────────────────────────────────────────────────────
oof_df = pd.DataFrame({'id': ids.values, 'oof_pred': oof_preds, 'target': y.values})
oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_preds.csv'), index=False)

# ── Save feature importances ─────────────────────────────────────────────────
fi_df = (
    fi_accum
    .reset_index()
    .rename(columns={'index': 'feature', 0: 'importance'})
    .sort_values('importance', ascending=False)
    .reset_index(drop=True)
)
fi_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

top5 = fi_df.head(5)[['feature', 'importance']].to_dict(orient='records')
print("\nTop-5 features:")
for row in top5:
    print(f"  {row['feature']}: {row['importance']}")

# Film count ranks
count_ranks = []
for cnt in COUNT_COLS:
    rank = fi_df[fi_df['feature'] == cnt].index[0] + 1
    count_ranks.append({'feature': cnt, 'rank': rank})
print("\nFilm-count feature ranks:")
for r in count_ranks:
    print(f"  {r['feature']}: rank {r['rank']}/{len(FEATURE_COLS)}")

iter9_rmse = 0.8097
delta = cv_rmse - iter9_rmse
beat = cv_rmse < iter9_rmse

# ── Save result.json ─────────────────────────────────────────────────────────
result = {
    "iteration":         11,
    "status":            "success",
    "cv_rmse":           round(cv_rmse, 6),
    "cv_std":            round(cv_std, 6),
    "fold_rmses":        [round(r, 6) for r in fold_rmses],
    "best_iterations":   best_iters,
    "duration_seconds":  round(duration, 2),
    "model":             "lightgbm",
    "n_features":        len(FEATURE_COLS),
    "feature_cols":      FEATURE_COLS,
    "iter9_baseline":    iter9_rmse,
    "delta_vs_iter9":    round(delta, 6),
    "beat_iter9":        bool(beat),
    "top5_features":     top5,
    "film_count_ranks":  count_ranks,
    "key_insight":       (
        f"Film-count features {'improved' if beat else 'did not improve'} over Iter 9 "
        f"(delta={delta:+.4f}); counts proxy experience/exposure of crew members."
    ),
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(os.path.join(OUTPUT_DIR, 'result.json'), 'w') as f:
    json.dump(result, f, indent=2, cls=NumpyEncoder)

print(f"\nResults written to {OUTPUT_DIR}")
print(f"Beat Iter 9 (0.8097): {beat}  (delta={delta:+.5f})")
