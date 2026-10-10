# Research State

## Config
- **Target**: tmdb_rating
- **Task**: regression
- **Metric**: rmse (lower_is_better)
- **Time Budget**: 14400s (4h)
- **Elapsed Compute**: 152s
- **N Folds**: 5
- **Data**:
  - Train: data_scraping/output/bollywood_crew_data.csv (675 rows x 14 cols)
  - Test: none (no held-out test set — all experiments use K-fold CV)
  - Enriched: none
- **ID Column**: tmdb_id
- **Positive Rate**: N/A (regression)
- **Kaggle Dataset Slug**: not configured — Kaggle disabled
- **Created**: 2026-10-10T00:00:00

## EDA Summary
- 675 rows, all categorical signal — no meaningful numeric predictors after excluding vote_count/year/matched_roles
- Target range: 2.3–8.5, mean 6.21, std 0.90; baseline RMSE (predict mean) = 0.90
- LightGBM native categorical handling is definitively the best single model approach
- Phase 3 stacking (Ridge meta-learner) gave best overall RMSE: 0.8004

## Strategy & Reflections
- BEST SINGLE: LightGBM native cat, num_leaves=31, min_child_samples=3, lr=0.05 → RMSE 0.8097
- BEST ENSEMBLE: Ridge meta-learner on [Iter9+Iter12+Iter6+Iter5] → RMSE 0.8004
- XGBoost native cat earns 0.16 weight in ensemble despite 0.8404 solo RMSE — different error patterns
- Ridge linear model earns 0.057 weight — useful diversity despite 0.8522 solo RMSE
- For production: train LightGBM native cat on full 675 rows; stacking impractical at inference
- Improvement over mean-baseline: 11.1% (0.90 → 0.8004)

## Global Best
- **Score**: 0.8004 (RMSE) — Ridge meta-stacker (Phase 3)
- **Best single model**: 0.8097 — LightGBM native cat (Iter 9)
- **Consecutive Failures**: 0
- **Total Experiments**: 13 (12 base + 1 Phase 3)
- **Successful Experiments**: 13

## Hypothesis Queue

### Phase 1 (Bold Bets)
{all completed}

### Phase 2 (Fine-Tuning)
{all completed}

### Phase 3 (Meta-Stacking)
{COMPLETED — Ridge meta-learner RMSE 0.8004}

## Experiment History (Rolling: last 10 + top 5)

### Top 5 by Score
| Iter | Phase | Model | CV Score | CV Std | Features | Duration | Key Insight |
|------|-------|-------|----------|--------|----------|----------|-------------|
| P3 | 3 | Ridge meta-stacker | 0.8004 | — | 6 OOF preds | <1s | Stacking gains 0.009 over best single |
| 9 | 2 | LightGBM native cat (nl=31, mcs=3) | 0.8097 | 0.0300 | 26 | 24s | Best single model |
| 8 | 2 | LightGBM native cat (nl=31, mcs=5) | 0.8179 | 0.0327 | 26 | 2s | Native categorical breakthrough |
| 12 | 2 | XGBoost native cat | 0.8404 | 0.0378 | 26 | 0.5s | Different errors from LGBM; key for stacking |
| 5 | 1 | XGBoost target-enc+curated flags | 0.8440 | 0.0404 | 26 | 1s | Best non-native-cat model |

### Recent 10
| Iter | Phase | Model | CV Score | Status | Duration | Key Insight |
|------|-------|-------|----------|--------|----------|-------------|
| 4 | 1 | LightGBM target-enc+curated | 0.8572 | success | 1s | Curated flags proved effective |
| 5 | 1 | XGBoost target-enc+curated | 0.8440 | success | 1s | XGBoost > LightGBM on target-enc |
| 6 | 1 | Ridge freq-enc (alpha=10) | 0.8522 | success | 0.1s | Lowest variance; useful stacking base |
| 7 | 2 | XGBoost regularized (depth=3) | 0.8467 | success | 0.4s | Underfits; depth=4 optimal |
| 8 | 2 | LightGBM native cat (mcs=5) | 0.8179 | success | 2s | Native cat breakthrough |
| 9 | 2 | LightGBM native cat grid | 0.8097 | success | 24s | Best single; nl=31,mcs=3 |
| 10 | 2 | LightGBM native cat LR sweep | 0.8097 | success | 11s | LR tuning no gain; feature ceiling |
| 11 | 2 | LightGBM native cat + film counts | 0.8229 | success | 2s | Film counts overfit; collinear with cats |
| 12 | 2 | XGBoost native cat | 0.8404 | success | 0.5s | Different error pattern; key for stacking |
| P3 | 3 | Ridge meta-stacker | 0.8004 | success | <1s | Best overall; 0.009 gain over Iter 9 |

## Completed Experiments (one-line summaries)
- Iter 1: LightGBM genre-only → 0.8679.
- Iter 2: LightGBM freq-encoded crew → 0.8689. Freq enc collapses rare entities.
- Iter 3: LightGBM OOF target-enc → 0.8725. Director #1; high fold variance.
- Iter 4: LightGBM target-enc+curated → 0.8572. Curated flags proved genuine signal.
- Iter 5: XGBoost target-enc+curated → 0.8440. XGBoost > LightGBM on target-enc.
- Iter 6: Ridge freq-enc (alpha=10) → 0.8522. Lowest fold variance (std=0.015).
- Iter 7: XGBoost regularized (depth=3) → 0.8467. Shallower underfits.
- Iter 8: LightGBM native cat (mcs=5) → 0.8179. Breakthrough — native cat >> manual encoding.
- Iter 9: LightGBM native cat grid (nl=31, mcs=3) → 0.8097. BEST SINGLE MODEL.
- Iter 10: LightGBM native cat LR sweep → 0.8097 still best. Feature ceiling confirmed.
- Iter 11: LightGBM native cat + film counts → 0.8229. Film counts hurt; collinear.
- Iter 12: XGBoost native cat → 0.8404. Different error patterns; key stacking component.
- Phase 3: Ridge meta-stacker on [Iter9,8,12,5,6,4] → 0.8004. BEST OVERALL.

## Learnings
- LightGBM native categorical >> all manual encoding (0.8097 vs 0.8440 best target-enc)
- Director is the #1 feature; curated binary flags consistently rank 3rd-5th
- min_child_samples=3 better than 5/10 — allows rare crew combination splits
- num_leaves=31 optimal; wider trees overfit on 675 rows
- lr=0.05 is optimal; lower LR with more trees converges to same error
- Film count features hurt — collinear with native categorical encoding
- XGBoost native cat earns 0.16 stacking weight despite 0.84 solo RMSE — different error patterns
- Ridge linear model earns 0.057 stacking weight — useful diversity
- LightGBM shallow stacker overfit (only 675 OOF samples, 6 meta-features)
- Improvement trajectory: 0.90 (baseline) → 0.8679 (genres) → 0.8097 (native cat) → 0.8004 (stacking)
- Production model: train LightGBM native cat (Iter 9 config) on full 675 rows
