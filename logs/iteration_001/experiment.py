"""
Iteration 001 — Genre-only baseline
Phase 1: Regression, 16 binary genre columns, LightGBM default params.
Objective: Beat predict-mean RMSE=0.90.
"""

import sys
import os
import json
import time

# Add project root to path so feature_template is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

from feature_template import load_and_preprocess, GENRE_COLS, ID_COL

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

GENRE_FEATURES = [
    "genre_drama", "genre_comedy", "genre_romance", "genre_action",
    "genre_thriller", "genre_crime", "genre_adventure", "genre_history",
    "genre_family", "genre_horror", "genre_mystery", "genre_fantasy",
    "genre_animation", "genre_music", "genre_war", "genre_biography",
]

def main():
    start_time = time.time()

    print("Loading and preprocessing data...")
    X_full, y, _, ids, _, _ = load_and_preprocess(as_category=False)

    # Filter to only the 16 genre columns
    available = [c for c in GENRE_FEATURES if c in X_full.columns]
    missing = [c for c in GENRE_FEATURES if c not in X_full.columns]
    if missing:
        print(f"WARNING: missing genre columns: {missing}")
    X = X_full[available].copy()

    print(f"Dataset shape: {X.shape}, target shape: {y.shape}")
    print(f"Features used ({len(available)}): {available}")
    print(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    feature_importance_dict = {col: 0 for col in available}

    lgb_params = dict(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )

    fold_rmses = []
    print("\nRunning 5-fold CV...")
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        fold_rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold_idx+1}: RMSE={fold_rmse:.4f}")

        # Accumulate feature importances
        for col, imp in zip(available, model.feature_importances_):
            feature_importance_dict[col] += imp

    # Overall OOF RMSE
    cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    cv_std = np.std(fold_rmses)
    duration = time.time() - start_time

    print(f"\nOOF RMSE: {cv_rmse:.4f} ± {cv_std:.4f}")
    print(f"Predict-mean baseline: 0.90")
    print(f"Beat baseline: {cv_rmse < 0.90}")

    # --- Save OOF predictions ---
    oof_df = pd.DataFrame({
        "tmdb_id": ids.values,
        "oof_pred": oof_preds,
        "actual": y.values,
    })
    oof_path = os.path.join(OUTPUT_DIR, "oof_preds.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")

    # --- Save feature importances ---
    avg_importance = {k: v / 5 for k, v in feature_importance_dict.items()}
    fi_df = pd.DataFrame(
        sorted(avg_importance.items(), key=lambda x: x[1], reverse=True),
        columns=["feature", "importance"],
    )
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"Feature importances saved to {fi_path}")
    print("\nTop feature importances:")
    print(fi_df.to_string(index=False))

    # --- Derive key insight ---
    beat_baseline = cv_rmse < 0.90
    top_genre = fi_df.iloc[0]["feature"] if len(fi_df) > 0 else "N/A"
    if beat_baseline:
        key_insight = (
            f"Genre-only features beat the predict-mean baseline (RMSE={cv_rmse:.4f} < 0.90), "
            f"with {top_genre} being the most predictive genre."
        )
    else:
        key_insight = (
            f"Genre-only features (RMSE={cv_rmse:.4f}) did NOT beat the predict-mean baseline (0.90), "
            f"indicating genres alone provide minimal rating signal; crew features are needed."
        )

    # --- Save result.json ---
    result = {
        "status": "success",
        "cv_rmse": round(float(cv_rmse), 6),
        "cv_std": round(float(cv_std), 6),
        "duration_seconds": round(duration, 2),
        "model": "LightGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42)",
        "features_used": available,
        "n_features": len(available),
        "key_insight": key_insight,
        "fold_rmses": [round(float(r), 6) for r in fold_rmses],
        "beat_baseline": bool(beat_baseline),
        "baseline_rmse": 0.90,
    }
    result_path = os.path.join(OUTPUT_DIR, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {result_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
