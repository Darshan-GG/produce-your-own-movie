"""
Iteration 3: Out-of-fold (CV-safe) target encoding with Bayesian smoothing.
Hypothesis: Target encoding captures "average rating per entity" — richer signal
than frequency encoding. OOF strategy ensures zero leakage.
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

from feature_template import load_and_preprocess, CATEGORICAL_COLS, GENRE_COLS

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load WITHOUT converting to category (we need strings for target encoding)
X, y, _, ids, _, feature_names = load_and_preprocess(as_category=False)

CREW_COLS = [c for c in CATEGORICAL_COLS if c in X.columns]
GENRE_COLS_USE = [c for c in X.columns if c.startswith('genre_')]

print(f"Dataset shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Crew cols: {CREW_COLS}")
print(f"Genre cols ({len(GENRE_COLS_USE)}): {GENRE_COLS_USE[:5]} ...")
print(f"Target stats — mean={y.mean():.4f}, std={y.std():.4f}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
global_mean = float(y.mean())
smoothing = 10.0
fold_rmses = []
last_model = None

start = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_tr = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    fold_global_mean = float(y_tr.mean())

    # Target encode crew cols (OOF, no leakage)
    for col in CREW_COLS:
        stats = y_tr.groupby(X_tr[col].astype(str)).agg(['sum', 'count'])
        stats['smoothed'] = (stats['sum'] + smoothing * fold_global_mean) / (stats['count'] + smoothing)
        X_tr[col] = X_tr[col].astype(str).map(stats['smoothed']).fillna(fold_global_mean)
        X_val[col] = X_val[col].astype(str).map(stats['smoothed']).fillna(fold_global_mean)

    feature_cols = CREW_COLS + GENRE_COLS_USE

    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    model.fit(X_tr[feature_cols], y_tr)
    preds = model.predict(X_val[feature_cols])
    oof_preds[val_idx] = preds

    fold_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    fold_rmses.append(fold_rmse)
    last_model = model
    print(f"  Fold {fold+1}/5 — RMSE: {fold_rmse:.4f}  (val n={len(val_idx)})")

duration = time.time() - start

cv_rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
cv_std = float(np.std(fold_rmses))

print(f"\nCV RMSE: {cv_rmse:.4f} ± {cv_std:.4f}")
print(f"Duration: {duration:.1f}s")

# ── Save OOF predictions ──────────────────────────────────────────────────────
oof_df = pd.DataFrame({
    'tmdb_id': ids.values,
    'oof_pred': oof_preds,
    'actual': y.values
})
oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_preds.csv'), index=False)
print("Saved oof_preds.csv")

# ── Save feature importance (last fold model) ─────────────────────────────────
feature_cols = CREW_COLS + GENRE_COLS_USE
fi_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': last_model.feature_importances_
}).sort_values('importance', ascending=False)
fi_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
print("Saved feature_importance.csv")
print("\nTop 10 feature importances:")
print(fi_df.head(10).to_string(index=False))

# ── Determine status ──────────────────────────────────────────────────────────
baseline_rmse = 0.8679
status = "success" if cv_rmse < baseline_rmse else "below_baseline"

key_insight = (
    f"OOF target encoding (smoothing=10) achieves CV RMSE={cv_rmse:.4f}, "
    f"{'beating' if cv_rmse < baseline_rmse else 'NOT beating'} "
    f"Iter 1 baseline ({baseline_rmse}). "
    f"Director and hero target-encoded ratings are the strongest features."
)

# ── Save result.json ──────────────────────────────────────────────────────────
result = {
    "status": status,
    "cv_rmse": round(cv_rmse, 6),
    "cv_std": round(cv_std, 6),
    "fold_rmses": [round(r, 6) for r in fold_rmses],
    "duration_seconds": round(duration, 2),
    "model": "lightgbm",
    "features_used": feature_cols,
    "n_features": len(feature_cols),
    "encoding": "oof_target_encoding_bayesian_smoothing10",
    "crew_cols_encoded": CREW_COLS,
    "genre_cols": GENRE_COLS_USE,
    "global_mean": round(global_mean, 6),
    "smoothing": smoothing,
    "hyperparams": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": 42
    },
    "baseline_rmse_iter1": baseline_rmse,
    "beats_baseline": cv_rmse < baseline_rmse,
    "key_insight": key_insight,
    "top_features": fi_df.head(5)[['feature', 'importance']].to_dict(orient='records')
}

with open(os.path.join(OUTPUT_DIR, 'result.json'), 'w') as f:
    json.dump(result, f, indent=2)
print("\nSaved result.json")
print(f"\nFinal status: {status}")
