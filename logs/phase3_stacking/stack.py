import sys, os
sys.path.insert(0, '/Users/darshan/GaminGaza Webapps/produce_your_own_movie')
import pandas as pd, numpy as np, json, time
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

BASE_DIR = '/Users/darshan/GaminGaza Webapps/produce_your_own_movie'

# Load all OOF files and align on tmdb_id
oof_files = {
    'lgbm_native_best': 'logs/iteration_009/oof_preds.csv',   # 0.8097
    'lgbm_native_mcs5': 'logs/iteration_008/oof_preds.csv',   # 0.8179
    'xgb_native':       'logs/iteration_012/oof_preds.csv',   # 0.8404
    'xgb_target_enc':   'logs/iteration_005/oof_preds.csv',   # 0.8440
    'ridge_linear':     'logs/iteration_006/oof_preds.csv',   # 0.8522
    'lgbm_target_enc':  'logs/iteration_004/oof_preds.csv',   # 0.8572
}

def load_oof(path):
    df = pd.read_csv(path)
    # Normalize column names: id -> tmdb_id, target -> actual
    df.columns = [c.strip() for c in df.columns]
    if 'id' in df.columns and 'tmdb_id' not in df.columns:
        df = df.rename(columns={'id': 'tmdb_id'})
    if 'target' in df.columns and 'actual' not in df.columns:
        df = df.rename(columns={'target': 'actual'})
    return df

dfs = {}
for name, path in oof_files.items():
    df = load_oof(os.path.join(BASE_DIR, path))
    dfs[name] = df.set_index('tmdb_id')['oof_pred']

# Build meta-feature matrix aligned on tmdb_id
# Use iter_009 as reference for actual values and index
ref_df = load_oof(os.path.join(BASE_DIR, 'logs/iteration_009/oof_preds.csv'))
ref_df = ref_df.set_index('tmdb_id')

# Find common index across all OOF files
common_ids = ref_df.index
for name, series in dfs.items():
    common_ids = common_ids.intersection(series.index)

print(f"Common tmdb_ids across all OOF files: {len(common_ids)}")

actual = ref_df.loc[common_ids, 'actual']
y = actual.values

# Build meta matrix
meta = pd.DataFrame({name: dfs[name].loc[common_ids] for name in oof_files.keys()})
X_meta = meta.values

print(f"Meta-feature matrix shape: {X_meta.shape}")
print(f"Columns (base models): {list(meta.columns)}")
print()

# Strategy 4 (reference): Best-model-only (Iter 9 alone)
rmse_best_single = float(np.sqrt(mean_squared_error(y, X_meta[:, 0])))  # lgbm_native_best
print(f"[Reference] Best single model (Iter 9): RMSE = {rmse_best_single:.4f}")

# Strategy 2: Simple weighted average (weights = 1/RMSE, normalized)
rmses_known = {
    'lgbm_native_best': 0.8097,
    'lgbm_native_mcs5': 0.8179,
    'xgb_native':       0.8404,
    'xgb_target_enc':   0.8440,
    'ridge_linear':     0.8522,
    'lgbm_target_enc':  0.8572,
}
weights_raw = {k: 1/v for k, v in rmses_known.items()}
total_w = sum(weights_raw.values())
weights = {k: w/total_w for k, w in weights_raw.items()}
weighted_pred = sum(X_meta[:, i] * weights[name] for i, name in enumerate(oof_files.keys()))
rmse_weighted = float(np.sqrt(mean_squared_error(y, weighted_pred)))
print(f"[Strategy 2] Weighted average:    RMSE = {rmse_weighted:.4f}")
print(f"  Weights: {', '.join(f'{k}={v:.4f}' for k,v in weights.items())}")

# Strategy 1: Ridge meta-learner with 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_ridge = np.zeros(len(X_meta))
ridge_coefs_per_fold = []
for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_meta)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_meta[train_idx])
    X_val = scaler.transform(X_meta[val_idx])
    m = Ridge(alpha=1.0, positive=True)  # positive=True: no negative weights
    m.fit(X_tr, y[train_idx])
    oof_ridge[val_idx] = m.predict(X_val)
    ridge_coefs_per_fold.append(m.coef_.tolist())
rmse_ridge = float(np.sqrt(mean_squared_error(y, oof_ridge)))
print(f"[Strategy 1] Ridge meta-learner:  RMSE = {rmse_ridge:.4f}")
mean_coefs = np.mean(ridge_coefs_per_fold, axis=0)
for i, name in enumerate(oof_files.keys()):
    print(f"  Ridge weight [{name}]: mean={mean_coefs[i]:.4f}")

# Strategy 3: LightGBM shallow stacker (max_depth=3)
oof_lgbm_stack = np.zeros(len(X_meta))
for train_idx, val_idx in kf.split(X_meta):
    m = lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                          num_leaves=7, random_state=42, verbose=-1)
    m.fit(X_meta[train_idx], y[train_idx],
          eval_set=[(X_meta[val_idx], y[val_idx])],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    oof_lgbm_stack[val_idx] = m.predict(X_meta[val_idx])
rmse_lgbm_stack = float(np.sqrt(mean_squared_error(y, oof_lgbm_stack)))
print(f"[Strategy 3] LightGBM stacker:    RMSE = {rmse_lgbm_stack:.4f}")

print()
print("=" * 55)
print("SUMMARY TABLE")
print("=" * 55)
print(f"{'Strategy':<30} {'RMSE':>8}  {'vs best':>8}")
print("-" * 55)
strategies = {
    'Best single (Iter 9)':     rmse_best_single,
    'Weighted average':          rmse_weighted,
    'Ridge meta-learner':        rmse_ridge,
    'LightGBM shallow stacker':  rmse_lgbm_stack,
}
for name, rmse in strategies.items():
    delta = rmse - rmse_best_single
    delta_str = f"{delta:+.4f}"
    print(f"  {name:<28} {rmse:>8.4f}  {delta_str:>8}")
print("=" * 55)

# Determine best strategy (excluding reference)
stack_strategies = {
    'weighted_average':       rmse_weighted,
    'ridge_meta_learner':     rmse_ridge,
    'lgbm_shallow_stacker':   rmse_lgbm_stack,
}
best_strategy = min(stack_strategies, key=stack_strategies.get)
best_rmse = stack_strategies[best_strategy]
improvement = rmse_best_single - best_rmse

print(f"\nBest stacking strategy: {best_strategy}  RMSE={best_rmse:.4f}")
print(f"Improvement over best single model: {improvement:+.4f}")

# Key insight
if best_rmse < rmse_best_single:
    key_insight = (
        f"{best_strategy} achieves RMSE {best_rmse:.4f}, "
        f"improving by {improvement:.4f} over the best single model (Iter 9, {rmse_best_single:.4f}), "
        "demonstrating that ensemble diversity from LightGBM, XGBoost, and Ridge base learners "
        "provides complementary signal even when individual gaps are modest."
    )
else:
    key_insight = (
        f"No stacking strategy beats the best single model ({rmse_best_single:.4f}); "
        f"the best stack ({best_strategy}) achieves {best_rmse:.4f} (+{abs(improvement):.4f}), "
        "suggesting the base model predictions are highly correlated and offer limited "
        "diversity for the meta-learner to exploit."
    )

print(f"\nKey insight: {key_insight}")

# Save result.json
result = {
    "status": "success",
    "strategies": {
        "best_single_iter9": rmse_best_single,
        "weighted_average": rmse_weighted,
        "ridge_meta_learner": rmse_ridge,
        "lgbm_shallow_stacker": rmse_lgbm_stack,
    },
    "best_strategy": best_strategy,
    "best_rmse": best_rmse,
    "best_single_rmse": rmse_best_single,
    "improvement": improvement,
    "ridge_weights_per_fold": {
        f"fold_{i+1}": dict(zip(list(oof_files.keys()), coefs))
        for i, coefs in enumerate(ridge_coefs_per_fold)
    },
    "ridge_mean_weights": dict(zip(list(oof_files.keys()), mean_coefs.tolist())),
    "weighted_avg_weights": weights,
    "key_insight": key_insight,
    "base_model_columns": list(oof_files.keys()),
    "n_samples": int(len(y)),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}

out_dir = os.path.join(BASE_DIR, 'logs/phase3_stacking')
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, 'result.json'), 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved result.json")

# Save stacked_preds.csv from best strategy
if best_strategy == 'weighted_average':
    best_preds = weighted_pred
elif best_strategy == 'ridge_meta_learner':
    best_preds = oof_ridge
else:
    best_preds = oof_lgbm_stack

stacked_df = pd.DataFrame({
    'tmdb_id': common_ids,
    'stacked_pred': best_preds,
    'actual': y,
})
stacked_df.to_csv(os.path.join(out_dir, 'stacked_preds.csv'), index=False)
print(f"Saved stacked_preds.csv  (strategy: {best_strategy}, n={len(stacked_df)})")
