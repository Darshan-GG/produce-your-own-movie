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
    CURATED_VILLAINS, CURATED_COMPOSERS,
)

OUTPUT_DIR = '/Users/darshan/GaminGaza Webapps/produce_your_own_movie/logs/iteration_008'

# Load WITH category dtype for crew cols
X, y, _, ids, _, _ = load_and_preprocess(as_category=True)

CREW_COLS = ['director', 'hero', 'heroine', 'villain', 'music_composer']
GENRE_COLS = [c for c in X.columns if c.startswith('genre_')]

# Add curated flags (using str comparison before category dtype causes issues)
X['is_curated_director']  = X['director'].astype(str).isin(CURATED_DIRECTORS).astype(int)
X['is_curated_hero']      = X['hero'].astype(str).isin(CURATED_HEROES).astype(int)
X['is_curated_heroine']   = X['heroine'].astype(str).isin(CURATED_HEROINES).astype(int)
X['is_curated_villain']   = X['villain'].astype(str).isin(CURATED_VILLAINS).astype(int)
X['is_curated_composer']  = X['music_composer'].astype(str).isin(CURATED_COMPOSERS).astype(int)

CURATED_FLAG_COLS = [
    'is_curated_director', 'is_curated_hero', 'is_curated_heroine',
    'is_curated_villain', 'is_curated_composer',
]
FEATURE_COLS = CREW_COLS + GENRE_COLS + CURATED_FLAG_COLS

print(f"Dataset shape: {X.shape}, Target shape: {y.shape}")
print(f"Features: {len(FEATURE_COLS)} total ({len(CREW_COLS)} crew cat + {len(GENRE_COLS)} genre + {len(CURATED_FLAG_COLS)} flags)")
print(f"Curated flag sums: {X[CURATED_FLAG_COLS].sum().to_dict()}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
fold_rmses = []
best_iters = []
fi_accum = None

start = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr  = X.iloc[train_idx][FEATURE_COLS].copy()
    X_val = X.iloc[val_idx][FEATURE_COLS].copy()
    y_tr  = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # Re-categorize to consistent categories per fold (avoid unknown category in val)
    for col in CREW_COLS:
        cats = X_tr[col].astype(str).unique().tolist()
        X_tr[col]  = pd.Categorical(X_tr[col].astype(str),  categories=cats)
        X_val[col] = pd.Categorical(X_val[col].astype(str), categories=cats)

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        categorical_feature=CREW_COLS,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    best_iters.append(int(model.best_iteration_))
    preds = model.predict(X_val)
    oof_preds[val_idx] = preds
    fold_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    fold_rmses.append(fold_rmse)
    print(f"  Fold {fold+1}: RMSE={fold_rmse:.4f}, best_iter={model.best_iteration_}")

    # Accumulate feature importances (mean across folds)
    fi = pd.DataFrame({'feature': FEATURE_COLS, 'importance': model.feature_importances_})
    if fi_accum is None:
        fi_accum = fi.copy()
    else:
        fi_accum['importance'] += fi['importance']

duration = time.time() - start

cv_rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
cv_std  = float(np.std(fold_rmses))

print(f"\n=== CV RMSE: {cv_rmse:.4f} ± {cv_std:.4f} ===")
print(f"Duration: {duration:.1f}s")
print(f"Best iterations: {best_iters}")

# Average feature importances
fi_accum['importance'] /= 5
fi_accum = fi_accum.sort_values('importance', ascending=False).reset_index(drop=True)
print("\nTop 10 feature importances:")
print(fi_accum.head(10).to_string(index=False))

# --- Save outputs ---

# OOF predictions
oof_df = pd.DataFrame({'id': ids.values, 'oof_pred': oof_preds, 'actual': y.values})
oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_preds.csv'), index=False)

# Feature importances
fi_accum.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

# Result JSON
ITER5_BEST = 0.8440
delta = round(cv_rmse - ITER5_BEST, 4)
status = 'success' if cv_rmse < ITER5_BEST else 'worse_than_baseline'

top5 = fi_accum.head(5)[['feature', 'importance']].to_dict('records')

result = {
    'iteration': 8,
    'status': status,
    'cv_rmse': round(cv_rmse, 4),
    'cv_std':  round(cv_std, 4),
    'fold_rmses': [round(r, 4) for r in fold_rmses],
    'best_iterations': best_iters,
    'duration_seconds': round(duration, 1),
    'model': 'lightgbm',
    'n_features': len(FEATURE_COLS),
    'feature_groups': {
        'crew_categorical': len(CREW_COLS),
        'genre_binary': len(GENRE_COLS),
        'curated_flags': len(CURATED_FLAG_COLS),
    },
    'hyperparameters': {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 5,
        'early_stopping_rounds': 50,
        'random_state': 42,
    },
    'vs_iter5_best': {
        'iter5_rmse': ITER5_BEST,
        'delta': delta,
        'better': cv_rmse < ITER5_BEST,
    },
    'top5_importances': top5,
    'key_insight': (
        'LightGBM native categorical handling with min_child_samples=5 '
        f'{"beat" if cv_rmse < ITER5_BEST else "did not beat"} Iter 5 baseline; '
        f'delta vs 0.8440: {delta:+.4f}. '
        'Curated flags added alongside native crew category columns.'
    ),
}

with open(os.path.join(OUTPUT_DIR, 'result.json'), 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}")
print(f"Status: {status}")
print(f"vs Iter 5 (0.8440): delta = {delta:+.4f}")
