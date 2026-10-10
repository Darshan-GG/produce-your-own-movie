"""
Standalone prediction script for the PYOM web app.
Usage: from ml.predict import predict_rating
result = predict_rating(
    director="Sanjay Leela Bhansali",
    hero="Ranveer Singh",
    heroine="Deepika Padukone",
    villain="Unknown",  # or actual villain name
    music_composer="Pritam",
    genres=["Romance", "Drama"]
)
# Returns: {"raw_score": 7.2, "display_score": 7.2, "stars": 3.6}
"""
import os, pickle, json
import pandas as pd
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_MODEL = None
_META = None

def _load():
    global _MODEL, _META
    if _MODEL is None:
        with open(os.path.join(_BASE, 'model.pkl'), 'rb') as f:
            _MODEL = pickle.load(f)
        with open(os.path.join(_BASE, 'model_metadata.json')) as f:
            _META = json.load(f)

def predict_rating(director: str, hero: str, heroine: str,
                   villain: str, music_composer: str,
                   genres: list) -> dict:
    """
    Predict TMDB rating for a crew + genre combination.

    Parameters
    ----------
    director, hero, heroine, villain, music_composer : str
        Exact names as in the game's curated lists. Unknown/not selected = "Unknown".
    genres : list of str
        List of genre strings, e.g. ["Action", "Drama"]. 1-3 typical.

    Returns
    -------
    dict with keys:
        raw_score   : float, model's prediction (0-10 scale)
        display_score : float, max(raw_score, 5.0)  — game's "never below 5" rule
        stars       : float, display_score / 2  — 0-5 star scale
    """
    _load()
    meta = _META

    # Build feature row
    row = {}
    row['director'] = director or 'Unknown'
    row['hero'] = hero or 'Unknown'
    row['heroine'] = heroine or 'Unknown'
    row['villain'] = villain or 'Unknown'
    row['music_composer'] = music_composer or 'Unknown'

    # Genre binary cols
    for genre in meta['all_genres']:
        col = f"genre_{genre.lower()}"
        row[col] = 1 if genre in genres else 0

    # Curated flags
    row['is_curated_director'] = int(row['director'] in meta['curated_directors'])
    row['is_curated_hero'] = int(row['hero'] in meta['curated_heroes'])
    row['is_curated_heroine'] = int(row['heroine'] in meta['curated_heroines'])
    row['is_curated_villain'] = int(row['villain'] in meta['curated_villains'])
    row['is_curated_composer'] = int(row['music_composer'] in meta['curated_composers'])

    df = pd.DataFrame([row])[meta['feature_cols']]

    # Cast crew cols as category (use training categories)
    for col in meta['crew_cols']:
        known_cats = meta['category_maps'][col]
        val = df[col].iloc[0]
        if val not in known_cats:
            df[col] = 'Unknown'  # unseen person → Unknown
        df[col] = pd.Categorical(df[col], categories=known_cats)

    raw = float(_MODEL.predict(df)[0])
    raw = max(0.0, min(10.0, raw))  # clamp to valid range
    display = max(raw, meta['game_display_min'])
    return {
        'raw_score': round(raw, 2),
        'display_score': round(display, 2),
        'stars': round(display / 2, 2),
    }

if __name__ == '__main__':
    # Quick test
    result = predict_rating(
        director='Sanjay Leela Bhansali',
        hero='Ranveer Singh',
        heroine='Deepika Padukone',
        villain='Unknown',
        music_composer='Pritam',
        genres=['Romance', 'Drama']
    )
    print('Test prediction:', result)
    result2 = predict_rating(
        director='Rohit Shetty',
        hero='Akshay Kumar',
        heroine='Katrina Kaif',
        villain='Prakash Raj',
        music_composer='Pritam',
        genres=['Action', 'Comedy']
    )
    print('Test prediction 2:', result2)
