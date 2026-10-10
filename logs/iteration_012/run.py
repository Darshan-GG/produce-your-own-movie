"""
Iteration 012: XGBoost with native categorical support (enable_categorical=True)
+ curated binary flags. Regression on tmdb_rating.
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

OUTPUT_DIR = '/Users/darshan/GaminGaza Webapps/produce_your_own_movie/logs/iteration_012'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"XGBoost version: {xgb.__version__}")

# Load data with as_category=False so crew cols are plain strings first
X, y, _, ids, _, _ = load_and_preprocess(as_category=False)

CREW_COLS = ['director', 'hero', 'heroine', 'villain', 'music_composer']
GENRE_COLS = [c for c in X.columns if c.startswith('genre_')]

# Add curated binary flags (before casting to category)
X['is_curated_director']  = X['director'].isin(CURATED_DIRECTORS).astype(int)
X['is_curated_hero']      = X['hero'].isin(CURATED_HEROES).astype(int)
X['is_curated_heroine']   = X['heroine'].isin(CURATED_HEROINES).astype(int)
X['is_curated_villain']   = X['villain'].isin(CURATED_VILLAINS).astype(int)
X['is_curated_composer']  = X['music_composer'].isin(CURATED_COMPOSERS).astype(int)
CURATED_FLAG_COLS = [
    'is_curated_director', 'is_curated_hero', 'is_curated_heroine',
    'is_curated_villain', 'is_curated_composer',
]

FEATURE_COLS = CREW_COLS + GENRE_COLS + CURATED_FLAG_COLS
print(f"Total features: {len(FEATURE_COLS)}")
print(f"  Crew cols:          {len(CREW_COLS)}")
print(f"  Genre cols:         {len(GENRE_COLS)}")
print(f"  Curated flag cols:  {len(CURATED_FLAG_COLS)}")
print(f"Dataset shape: {X.shape}, target shape: {y.shape}")

# Cast crew cols to category
for col in CREW_COLS:
    X[col] = X[col].astype('category')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
fold_rmses = []
best_iters = []

start = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr  = X.iloc[train_idx][FEATURE_COLS].copy()
    X_val = X.iloc[val_idx][FEATURE_COLS].copy()
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Align categories: val may have unseen values — fit categories on train only
    for col in CREW_COLS:
        cats = X_tr[col].cat.categories.tolist()
        X_tr[col]  = pd.Categorical(X_tr[col],  categories=cats)
        X_val[col] = pd.Categorical(X_val[col], categories=cats)

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method='hist',
        early_stopping_rounds=50,
        eval_metric='rmse',
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration
    best_iters.append(int(best_iter))

    preds = model.predict(X_val)
    oof_preds[val_idx] = preds

    fold_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    fold_rmses.append(fold_rmse)
    print(f"  Fold {fold+1}: RMSE={fold_rmse:.4f}  best_iter={best_iter}")

duration = time.time() - start
cv_rmse  = float(np.sqrt(mean_squared_error(y, oof_preds)))
cv_std   = float(np.std(fold_rmses))

print(f"\nOOF CV RMSE : {cv_rmse:.4f}")
print(f"CV Std      : {cv_std:.4f}")
print(f"Best iters  : {best_iters}")
print(f"Duration    : {duration:.1f}s")

# ── Save OOF predictions ──────────────────────────────────────────────────────
oof_df = pd.DataFrame({'id': ids.values, 'oof_pred': oof_preds, 'actual': y.values})
oof_df.to_csv(f'{OUTPUT_DIR}/oof_preds.csv', index=False)
print(f"Saved oof_preds.csv")

# ── Save result.json ──────────────────────────────────────────────────────────
success = cv_rmse < 0.90
result = {
    'status':           'success' if success else 'below_criterion',
    'cv_rmse':          round(cv_rmse, 6),
    'cv_std':           round(cv_std, 6),
    'duration_seconds': round(duration, 1),
    'model':            'xgboost_native_categorical',
    'xgboost_version':  xgb.__version__,
    'best_iterations':  best_iters,
    'fold_rmses':       [round(r, 6) for r in fold_rmses],
    'n_features':       len(FEATURE_COLS),
    'feature_groups': {
        'crew_categorical': CREW_COLS,
        'genre_binary':     GENRE_COLS,
        'curated_flags':    CURATED_FLAG_COLS,
    },
    'hyperparameters': {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'enable_categorical': True,
        'tree_method': 'hist',
        'early_stopping_rounds': 50,
    },
    'key_insight': (
        f"XGBoost native categorical (enable_categorical=True) achieved CV RMSE {cv_rmse:.4f}; "
        f"diverse base learner for stacking alongside LightGBM (different categorical split logic)."
    ),
    'success_criterion': 'RMSE < 0.90',
    'meets_criterion': success,
}

with open(f'{OUTPUT_DIR}/result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved result.json")
print(f"\nStatus: {'SUCCESS' if success else 'BELOW CRITERION (RMSE >= 0.90)'}")
