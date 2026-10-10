"""
Iteration 002: Genre binary cols + frequency-encoded crew categoricals.
Hypothesis: crew identity (director, hero, heroine, villain, music_composer)
encoded by appearance frequency adds signal beyond genre-only baseline.
"""

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/Users/darshan/GaminGaza Webapps/produce_your_own_movie')

warnings.filterwarnings("ignore")

from feature_template import (
    load_and_preprocess,
    CATEGORICAL_COLS,
    GENRE_COLS,
    frequency_encode,
)
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

OUTPUT_DIR = Path(__file__).parent

def main():
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # 1. Load data — as_category=False so we can freq-encode numerically  #
    # ------------------------------------------------------------------ #
    X, y, _, ids, _, feature_names = load_and_preprocess(as_category=False)

    # Identify genre cols present in X
    genre_cols_present = [c for c in GENRE_COLS if c in X.columns]
    cat_cols_present   = [c for c in CATEGORICAL_COLS if c in X.columns]

    # Features we will actually use: genre binary cols + categorical crew cols
    use_cols = genre_cols_present + cat_cols_present
    X_model = X[use_cols].copy()

    print(f"Dataset shape: {X_model.shape}")
    print(f"Genre cols: {len(genre_cols_present)}")
    print(f"Categorical cols to freq-encode: {cat_cols_present}")
    print(f"Target shape: {y.shape}, nulls: {y.isna().sum()}")

    # ------------------------------------------------------------------ #
    # 2. 5-fold CV with frequency encoding inside each fold               #
    # ------------------------------------------------------------------ #
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds  = np.full(len(y), np.nan)
    fold_rmses = []
    feature_importances = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_model)):
        X_tr_raw = X_model.iloc[train_idx].copy()
        X_va_raw = X_model.iloc[val_idx].copy()
        y_tr     = y.iloc[train_idx]
        y_va     = y.iloc[val_idx]

        # Frequency encode using TRAIN fold frequencies only (no leakage)
        X_tr_enc, X_va_enc = frequency_encode(
            X_train=X_tr_raw,
            X_val=X_va_raw,
            cols=cat_cols_present,
        )

        # All columns are now numeric — build LightGBM datasets
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_tr_enc, y_tr)

        val_pred = model.predict(X_va_enc)
        oof_preds[val_idx] = val_pred

        fold_rmse = np.sqrt(mean_squared_error(y_va, val_pred))
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold_idx+1}/5 | RMSE: {fold_rmse:.4f} | val_n={len(val_idx)}")

        # Collect feature importances
        fi = pd.Series(model.feature_importances_, index=X_tr_enc.columns)
        feature_importances.append(fi)

    # ------------------------------------------------------------------ #
    # 3. Aggregate results                                                 #
    # ------------------------------------------------------------------ #
    # Overall OOF RMSE (on full array, not mean of fold RMSEs)
    valid_mask = ~np.isnan(oof_preds)
    cv_rmse    = float(np.sqrt(mean_squared_error(y[valid_mask], oof_preds[valid_mask])))
    cv_std     = float(np.std(fold_rmses))
    duration   = time.time() - t0

    print(f"\nOverall OOF RMSE : {cv_rmse:.4f}")
    print(f"Per-fold std     : {cv_std:.4f}")
    print(f"Duration         : {duration:.1f}s")

    # Beat baseline?
    baseline_rmse = 0.8679
    beat_baseline = cv_rmse < baseline_rmse
    print(f"Beat Iter-1 baseline ({baseline_rmse})? {'YES' if beat_baseline else 'NO'}")

    # ------------------------------------------------------------------ #
    # 4. Feature importance                                               #
    # ------------------------------------------------------------------ #
    mean_fi = pd.concat(feature_importances, axis=1).mean(axis=1).sort_values(ascending=False)
    fi_df = mean_fi.reset_index()
    fi_df.columns = ["feature", "importance"]
    fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    print("\nTop-10 features:")
    print(fi_df.head(10).to_string(index=False))

    # ------------------------------------------------------------------ #
    # 5. OOF predictions CSV                                              #
    # ------------------------------------------------------------------ #
    oof_df = pd.DataFrame({
        "tmdb_id":  ids.values,
        "oof_pred": oof_preds,
        "actual":   y.values,
    })
    oof_df.to_csv(OUTPUT_DIR / "oof_preds.csv", index=False)

    # ------------------------------------------------------------------ #
    # 6. result.json                                                      #
    # ------------------------------------------------------------------ #
    # Key insight
    crew_importance = float(mean_fi[cat_cols_present].sum())
    genre_importance = float(mean_fi[genre_cols_present].sum())
    top_feature = mean_fi.index[0]

    if beat_baseline:
        key_insight = (
            f"Frequency-encoded crew adds meaningful signal (total crew importance "
            f"{crew_importance:.0f} vs genre {genre_importance:.0f}); "
            f"top feature: {top_feature}."
        )
    else:
        key_insight = (
            f"Frequency encoding did NOT beat genre-only baseline "
            f"(RMSE {cv_rmse:.4f} vs {baseline_rmse}); "
            f"crew freq importance {crew_importance:.0f}, top feature: {top_feature}."
        )

    result = {
        "status":           "success" if not np.isnan(cv_rmse) else "terminal_failure",
        "cv_rmse":          round(cv_rmse, 4),
        "cv_std":           round(cv_std, 4),
        "duration_seconds": round(duration, 1),
        "model":            "LightGBM (n_estimators=100, lr=0.1, num_leaves=31)",
        "features_used":    use_cols,
        "n_features":       len(use_cols),
        "beat_baseline":    beat_baseline,
        "baseline_rmse":    baseline_rmse,
        "per_fold_rmses":   [round(r, 4) for r in fold_rmses],
        "top_10_features":  fi_df.head(10)["feature"].tolist(),
        "crew_total_importance":  round(crew_importance, 1),
        "genre_total_importance": round(genre_importance, 1),
        "key_insight":      key_insight,
    }

    with open(OUTPUT_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")
    return result


if __name__ == "__main__":
    main()
