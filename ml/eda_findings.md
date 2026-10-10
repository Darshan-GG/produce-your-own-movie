# EDA Findings — Produce Your Own Movie (ML Dataset)

**Dataset**: `data_scraping/output/bollywood_crew_data.csv`  
**Date**: 2026-10-10  
**Analyst**: Claude Code (auto-research)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Rows | 675 |
| Columns | 14 |
| Target | `tmdb_rating` (regression) |
| Rating range | 2.3 – 8.5 |
| Rating mean | 6.21 ± 0.90 |
| Source: seed | 157 rows (23%) |
| Source: expanded | 518 rows (77%) |

---

## 2. Target Distribution

| Bucket | Count | % |
|---|---|---|
| Below 5 | 56 | 8.3% |
| 5 – 6 | 173 | 25.6% |
| 6 – 7 | 303 | 44.9% |
| 7 – 8 | 137 | 20.3% |
| 8+ | 6 | 0.9% |

**Key insight**: Target is narrow and right-skewed toward the 6–7 range. Only 6 movies score 8+. RMSE baseline (predict mean) ≈ 0.90.

---

## 3. Missing Values (Feature Columns)

| Column | Null % | Action |
|---|---|---|
| director | 0% | ✓ None needed |
| hero | 16.2% | Fill → "Unknown" |
| heroine | 40.4% | Fill → "Unknown" |
| villain | 65.2% | Fill → "Unknown" (user confirmed) |
| music_composer | 0% | ✓ None needed |
| genres | 0.15% | Fill → "Drama" (mode) |

---

## 4. Feature Cardinality

| Column | Unique Values | Problem |
|---|---|---|
| director | 279 | Very sparse vs 675 rows |
| villain | 123 | High, but 65% are "Unknown" after fill |
| music_composer (raw) | 183 (multi-value) → 197 after split | Multi-value |
| genres (raw) | 246 combos | Multi-value → 16 base genres |
| heroine | ~80 | Moderate |
| hero | 59 | Moderate |

---

## 5. Numeric Feature Correlations with Target

| Column | Correlation | Note |
|---|---|---|
| vote_count | +0.34 | **Excluded** — unavailable at inference (game context) |
| year | -0.07 | Weak negative (older films rated slightly lower) |
| matched_roles | +0.065 | **Excluded** — data leakage (not available at inference) |

**Key insight**: After excluding vote_count and matched_roles, NO numeric feature has meaningful correlation with target. All signal comes from categorical features.

---

## 6. Multi-Value Columns

### Genres (16 unique base genres)
| Genre | Count |
|---|---|
| Drama | 408 |
| Comedy | 265 |
| Romance | 263 |
| Action | 258 |
| Thriller | 162 |
| Crime | 117 |
| Adventure | 46 |
| History | 42 |
| Family | 37 |
| Horror | 25 |

→ Will be split into 16 binary columns.

### Music Composer (multi-value, | separated)
- 197 unique composer names across all values
- Average 1.3 composers per movie
- Top composers: A.R. Rahman (57 movies), Pritam (46), Anu Malik (44)
- **Game only has 22 curated composers** — others will map to "Unknown"
- **Pending decision**: use first composer only OR create binary flags for all 22 curated names

---

## 7. Top Entities

### Directors (top 5 by movie count)
Mani Ratnam (14), Rohit Shetty (14), David Dhawan (13), Anurag Kashyap (13), Priyadarshan (12)

### Heroes (top 5)
Akshay Kumar (67), Shah Rukh Khan (54), Salman Khan (45), Ajay Devgn (41), Aamir Khan (26)

### Heroines (top 5)
Kareena Kapoor Khan (36), Katrina Kaif (31), Rani Mukerji (25), Deepika Padukone (25), Aishwarya Rai Bachchan (24)

### Villains (top 5)
Prakash Raj (20), Jackie Shroff (13), Sonu Sood (10), Amrish Puri (8), Nawazuddin Siddiqui (8)

---

## 8. Vote Count Quality

| Metric | Value |
|---|---|
| Min votes | 25 |
| Median votes | 72 |
| Mean votes | 129 |
| Max votes | 4,625 |
| % under 50 votes | 32.3% |
| % under 100 votes | 62.1% |

**Key concern**: 62% of ratings are based on <100 votes. Low vote counts = noisy/unreliable ratings. This adds inherent noise to the target — the model ceiling may be lower than expected.

---

## 9. Columns to EXCLUDE from Features

| Column | Reason |
|---|---|
| `tmdb_id` | ID column |
| `movie_title` | Near-unique identifier (673/675) |
| `original_language` | Near-constant (only hi/te/ta, low variance) |
| `vote_count` | Strong predictor but unavailable at inference (**user confirmed exclude**) |
| `matched_roles` | Data leakage — artifact of scraping, not available at inference |
| `source` | Seed/expanded label — not a real feature |

---

## 10. Key Challenges & Risks

1. **Narrow target range** (std=0.9): Model will find it hard to distinguish a 6.5 from a 7.0 film. Expect RMSE ≈ 0.6–0.8 as realistic ceiling.

2. **Signal entirely in categoricals**: No numeric feature (after exclusions) correlates meaningfully with rating. Success depends entirely on learning quality signals from director/hero/genre etc.

3. **High cardinality with small dataset**: 279 unique directors in 675 rows. Many appear only once. Target encoding with CV is essential to avoid overfitting.

4. **Noisy target**: 62% of ratings from <100 votes — inherently noisy ground truth. This caps model performance independent of algorithm choice.

5. **Villain sparsity**: 65% null. After "Unknown" fill, model may learn "no villain" as a signal rather than true villain quality.

6. **Out-of-game crew in training data**: Dataset includes directors/actors outside our curated game lists. At inference, these map to "Unknown". This is intentional but means the model must generalize the "Unknown" category correctly.

---

## 11. Proposed ML Strategy

### Feature Set
- `director` → frequency encode + target encode (CV-safe)
- `hero` → frequency encode + target encode (null → "Unknown")
- `heroine` → frequency encode + target encode (null → "Unknown")
- `villain` → frequency encode + target encode (null → "Unknown")
- `music_composer` → **TBD** (first-only vs multi-hot binary flags for 22 curated composers)
- `genres` → 16 binary columns (one per base genre)
- `year` → numeric as-is (or omit — **pending user decision**)

### Model Plan
1. **Baseline**: LightGBM default, frequency-encoded categoricals only
2. **Add genre binary flags** + target encoding for hero/director
3. **XGBoost** with same features for diversity
4. **Feature engineering**: era buckets from year, genre combos, composer tier
5. **Stacking**: Ridge meta-learner on top OOF predictions

---

*Pending decisions (see strategy discussion):*
- Music composer encoding method
- Whether to include `year` as feature
