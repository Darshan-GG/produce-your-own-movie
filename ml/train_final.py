import sys, os, json, pickle
sys.path.insert(0, '/Users/darshan/GaminGaza Webapps/produce_your_own_movie')
import pandas as pd, numpy as np
import lightgbm as lgb

from feature_template import (load_and_preprocess,
    CURATED_DIRECTORS, CURATED_HEROES, CURATED_HEROINES, CURATED_VILLAINS, CURATED_COMPOSERS,
    ALL_GENRES, CATEGORICAL_COLS)

# Load full dataset
X, y, _, ids, _, _ = load_and_preprocess(as_category=True)
CREW_COLS = ['director','hero','heroine','villain','music_composer']
GENRE_COLS = [c for c in X.columns if c.startswith('genre_')]

# Add curated flags
X['is_curated_director'] = (X['director'].astype(str).isin(CURATED_DIRECTORS)).astype(int)
X['is_curated_hero'] = (X['hero'].astype(str).isin(CURATED_HEROES)).astype(int)
X['is_curated_heroine'] = (X['heroine'].astype(str).isin(CURATED_HEROINES)).astype(int)
X['is_curated_villain'] = (X['villain'].astype(str).isin(CURATED_VILLAINS)).astype(int)
X['is_curated_composer'] = (X['music_composer'].astype(str).isin(CURATED_COMPOSERS)).astype(int)
CURATED_FLAG_COLS = ['is_curated_director','is_curated_hero','is_curated_heroine','is_curated_villain','is_curated_composer']
FEATURE_COLS = CREW_COLS + GENRE_COLS + CURATED_FLAG_COLS

# Ensure category dtype and record all category mappings
category_maps = {}
for col in CREW_COLS:
    X[col] = X[col].astype('category')
    category_maps[col] = X[col].cat.categories.tolist()

print(f"Training on {len(X)} rows, {len(FEATURE_COLS)} features")
print(f"Feature cols: {FEATURE_COLS}")

# Train on full dataset (no early stopping — use fixed n_estimators=60)
model = lgb.LGBMRegressor(
    n_estimators=60, learning_rate=0.05, num_leaves=31,
    min_child_samples=3, random_state=42, verbose=-1
)
model.fit(X[FEATURE_COLS], y, categorical_feature=CREW_COLS)

print("Model trained successfully.")

# Save model
ml_dir = '/Users/darshan/GaminGaza Webapps/produce_your_own_movie/ml'
os.makedirs(ml_dir, exist_ok=True)
model_path = os.path.join(ml_dir, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to: {model_path}")

# Save metadata
metadata = {
    'feature_cols': FEATURE_COLS,
    'crew_cols': CREW_COLS,
    'genre_cols': GENRE_COLS,
    'curated_flag_cols': CURATED_FLAG_COLS,
    'category_maps': category_maps,
    'all_genres': ALL_GENRES,
    'curated_directors': CURATED_DIRECTORS,
    'curated_heroes': CURATED_HEROES,
    'curated_heroines': CURATED_HEROINES,
    'curated_villains': CURATED_VILLAINS,
    'curated_composers': CURATED_COMPOSERS,
    'n_estimators': 60,
    'cv_rmse_5fold': 0.8097,
    'train_rows': len(X),
    'game_display_min': 5.0,  # display = max(prediction, 5)
    'target_col': 'tmdb_rating',
    'scale': '0-10',
}
metadata_path = os.path.join(ml_dir, 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")
print("Done.")
