"""
Feature preprocessing pipeline for the Produce Your Own Movie ML model.
Import this in experiment scripts to get a consistent preprocessing baseline.

Key decisions (see ml/eda_findings.md for full reasoning):
- vote_count EXCLUDED (unavailable at inference)
- year EXCLUDED (user decision)
- matched_roles EXCLUDED (data leakage)
- villain null (65%) → filled "Unknown"
- heroine null (40%) → filled "Unknown"
- hero null (16%) → filled "Unknown"
- music_composer: multi-value → first value only (matches game's single-slot)
- genres: multi-value → 16 binary columns
"""

import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_PATH = str(Path(__file__).parent / "data_scraping/output/bollywood_crew_data.csv")

CURATED_DIRECTORS = [
    "Yash Chopra", "Sanjay Leela Bhansali", "Rajkumar Hirani", "Karan Johar",
    "Aditya Chopra", "Rohit Shetty", "Farah Khan", "Anurag Basu", "Imtiaz Ali",
    "Zoya Akhtar", "Ashutosh Gowariker", "Sandeep Reddy Vanga", "Ayan Mukerji",
    "Anurag Kashyap", "Amar Kaushik", "Atlee", "Aditya Dhar", "S. S. Rajamouli",
]

CURATED_HEROES = [
    "Ranbir Kapoor", "Ranveer Singh", "Vicky Kaushal", "Shahid Kapoor",
    "Kartik Aaryan", "Varun Dhawan", "Ayushmann Khurrana", "Sidharth Malhotra",
    "Emraan Hashmi", "R. Madhavan", "Riteish Deshmukh", "Sushant Singh Rajput",
    "Shah Rukh Khan", "Salman Khan", "Aamir Khan", "Akshay Kumar", "Ajay Devgn",
    "Hrithik Roshan", "Sanjay Dutt", "Saif Ali Khan", "Prabhas", "Allu Arjun",
    "Ram Charan", "Jr. NTR", "Mahesh Babu",
]

CURATED_HEROINES = [
    "Deepika Padukone", "Alia Bhatt", "Kareena Kapoor Khan", "Katrina Kaif",
    "Priyanka Chopra Jonas", "Shraddha Kapoor", "Kiara Advani", "Kriti Sanon",
    "Rashmika Mandanna", "Triptii Dimri", "Madhuri Dixit", "Kajol",
    "Aishwarya Rai Bachchan", "Yami Gautam", "Mrunal Thakur", "Sharvari",
    "Disha Patani", "Rani Mukerji", "Preity Zinta", "Karisma Kapoor",
    "Anushka Sharma", "Janhvi Kapoor", "Sara Ali Khan", "Kangana Ranaut", "Ananya Panday",
]

CURATED_VILLAINS = [
    "Amrish Puri", "Arjun Kapoor", "Rana Daggubati", "Vidya Balan", "Tabu",
    "John Abraham", "Saif Ali Khan", "Prakash Raj", "Urmila Matondkar",
    "Akshaye Khanna", "Vijay Sethupathi", "Pankaj Tripathi", "Bobby Deol",
    "Arjun Rampal", "Raghav Juyal", "Richa Chadha", "Ranjeet", "Nawazuddin Siddiqui",
    "Sonu Sood", "Danny Denzongpa", "Jackie Shroff", "Sushmita Sen",
]

CURATED_COMPOSERS = [
    "A. R. Rahman", "Pritam", "Shankar-Ehsaan-Loy", "Vishal-Shekhar",
    "Amit Trivedi", "Anu Malik", "Jatin-Lalit", "Bappi Lahiri", "Rajesh Roshan",
    "Sachin-Jigar", "Ajay-Atul", "Amaal Mallik", "Tanishk Bagchi", "Mithoon",
    "Salim-Sulaiman", "Shantanu Moitra", "Ram Sampath", "Jeet Gannguli",
    "Sachet-Parampara", "Darshan Raval", "Himesh Reshammiya", "Jasleen Royal",
]

ALL_GENRES = [
    "Drama", "Comedy", "Romance", "Action", "Thriller", "Crime",
    "Adventure", "History", "Family", "Horror", "Mystery", "Fantasy",
    "Animation", "Music", "War", "Biography",
]

TARGET_COL = "tmdb_rating"
ID_COL = "tmdb_id"
CATEGORICAL_COLS = ["director", "hero", "heroine", "villain", "music_composer"]
GENRE_COLS = [f"genre_{g.lower()}" for g in ALL_GENRES]
EXCLUDED_COLS = ["tmdb_id", "movie_title", "year", "original_language",
                 "vote_count", "matched_roles", "source", "genres"]


def _preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing decisions in order."""
    df = df.copy()

    # music_composer: first value only (game has single slot)
    df["music_composer"] = (
        df["music_composer"]
        .fillna("Unknown")
        .apply(lambda x: x.split("|")[0].strip())
    )

    # Fill null crew roles
    for col in ["director", "hero", "heroine", "villain"]:
        df[col] = df[col].fillna("Unknown")

    # genres → 16 binary columns
    df["genres"] = df["genres"].fillna("Drama")
    for genre in ALL_GENRES:
        df[f"genre_{genre.lower()}"] = (
            df["genres"].str.contains(genre, regex=False).astype(int)
        )

    return df


def load_and_preprocess(train_path=None, test_path=None,
                        target_col=TARGET_COL, id_col=ID_COL,
                        as_category=True):
    """
    Load and preprocess dataset. Returns (X, y, X_test, id_train, id_test, feature_names).

    No separate test set exists — X_test and id_test are always None.
    All experiments should use K-fold CV on X / y.

    Parameters
    ----------
    as_category : bool
        If True, cast CATEGORICAL_COLS to pd.Categorical (for LightGBM native handling).
        Set False for sklearn models that need numeric input.
    """
    path = train_path or TRAIN_PATH
    df = pd.read_csv(path)
    df = _preprocess_raw(df)

    feature_cols = [c for c in df.columns
                    if c not in EXCLUDED_COLS + [target_col]]

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    ids = df[id_col].copy()

    if as_category:
        for col in CATEGORICAL_COLS:
            if col in X.columns:
                X[col] = X[col].astype("category")

    return X, y, None, ids, None, feature_cols


def frequency_encode(X_train: pd.DataFrame, X_val: pd.DataFrame = None,
                     cols: list = None) -> tuple:
    """
    Frequency encode categorical columns using train frequencies only (no leakage).
    Returns (X_train_enc, X_val_enc). X_val_enc is None if X_val is None.
    """
    cols = cols or CATEGORICAL_COLS
    X_tr = X_train.copy()
    X_v = X_val.copy() if X_val is not None else None

    for col in cols:
        if col not in X_tr.columns:
            continue
        freq_map = X_tr[col].astype(str).value_counts(normalize=True).to_dict()
        X_tr[col] = X_tr[col].astype(str).map(freq_map).fillna(0.0)
        if X_v is not None:
            X_v[col] = X_v[col].astype(str).map(freq_map).fillna(0.0)

    return X_tr, X_v


def target_encode_cv(X: pd.DataFrame, y: pd.Series, cv_folds,
                     cols: list = None, smoothing: float = 10.0) -> pd.DataFrame:
    """
    Target encode categorical columns using out-of-fold estimates (no leakage).
    Must be called separately for each CV fold.

    Parameters
    ----------
    smoothing : float
        Bayesian smoothing factor — higher = more shrinkage toward global mean.
    """
    cols = cols or CATEGORICAL_COLS
    X_out = X.copy()
    global_mean = y.mean()

    for col in cols:
        if col not in X_out.columns:
            continue
        oof_enc = pd.Series(index=X.index, dtype=float)
        col_str = X[col].astype(str)

        for _, val_idx in cv_folds:
            train_mask = ~X.index.isin(val_idx)
            stats = y[train_mask].groupby(col_str[train_mask]).agg(["sum", "count"])
            smoothed = (stats["sum"] + smoothing * global_mean) / (stats["count"] + smoothing)
            oof_enc.iloc[val_idx] = col_str.iloc[val_idx].map(smoothed).fillna(global_mean)

        X_out[col] = oof_enc

    return X_out


def add_curated_flags(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary flags: is the crew member from the game's curated list?
    Useful signal — curated members are well-known, which correlates with quality.
    """
    X = X.copy()
    if "director" in X.columns:
        X["is_curated_director"] = X["director"].astype(str).isin(CURATED_DIRECTORS).astype(int)
    if "hero" in X.columns:
        X["is_curated_hero"] = X["hero"].astype(str).isin(CURATED_HEROES).astype(int)
    if "heroine" in X.columns:
        X["is_curated_heroine"] = X["heroine"].astype(str).isin(CURATED_HEROINES).astype(int)
    if "villain" in X.columns:
        X["is_curated_villain"] = X["villain"].astype(str).isin(CURATED_VILLAINS).astype(int)
    if "music_composer" in X.columns:
        X["is_curated_composer"] = X["music_composer"].astype(str).isin(CURATED_COMPOSERS).astype(int)
    return X


def get_data_info() -> dict:
    """Return dataset metadata without reading the full file."""
    return {
        "train_path": TRAIN_PATH,
        "test_path": None,
        "target_col": TARGET_COL,
        "id_col": ID_COL,
        "task_type": "regression",
        "metric": "rmse",
        "n_rows": 675,
        "n_folds": 5,
        "feature_names": CATEGORICAL_COLS + GENRE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "genre_cols": GENRE_COLS,
        "notes": [
            "No separate test set — use K-fold CV on full dataset",
            "vote_count excluded (unavailable at inference)",
            "year excluded (user decision)",
            "matched_roles excluded (data leakage)",
            "villain: 65% null -> 'Unknown'",
            "heroine: 40% null -> 'Unknown'",
            "music_composer: multi-value -> first value only",
            "genres: multi-value -> 16 binary columns",
            "Realistic RMSE ceiling ~0.65-0.80 due to noisy low-vote-count ratings",
        ],
    }
