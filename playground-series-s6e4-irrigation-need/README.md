# Playground Series S6E4 — Predicting Irrigation Need

Kaggle Tabular Playground Series, Season 6 Episode 4. Multiclass classification predicting `Irrigation_Need` (Low / Medium / High) from 19 agronomic features. Metric: **balanced accuracy**.

- **Best public LB:** `0.97785`
- **Final rank:** 775/4315 (Top 18%, I could do better :/)
- **Submissions made:** 6 scoring (best kept; rest pruned during exploration)
- **Compute:** local CPU (M-series MacBook), no GPU
- **Total runtime across all rounds:** ~120 hours of model training

This repo documents the approach end-to-end, including the dead-ends, because the dead-ends are where the real learning happened on this competition.

---

## TL;DR

1. The competition organisers released a small noise-free `irrigation_prediction.csv` (10K rows) alongside the synthetic 630K-row train set. A community member identified that the original data is **fully deterministic** under a simple linear scoring formula on 7 features.
2. Standard target encoding from the noisy 630K train set was actively harmful (later confirmed in R46). The clean 10K original is the only reliable encoding source.
3. The breakthrough was switching from a **scalar** mean-target encoding (`E[y|feat=x]`) to a **per-class probability** target encoding (`P(y=c|feat=x)` for each class c) computed against the clean original. This 3× richer signal moved the LB from 0.97655 → 0.97785.
4. Beyond R48, a sustained ceiling appeared. Five distinct attack vectors all failed to improve LB despite 30+ hours of additional compute, because the MTE features had already absorbed essentially everything the formula encoded.

---

## The data

| File | Rows | Notes |
|------|------|-------|
| `train.csv` | 630,000 | Synthetic, target balance ≈ Low 58.7%, Med 38.0%, High 3.3% |
| `test.csv` | 270,000 | Synthetic, same distribution |
| `irrigation_prediction.csv` | 10,000 | "Original" data, noise-free, fully deterministic |

Features (19 total):
- 8 categoricals: `Soil_Type`, `Crop_Type`, `Crop_Growth_Stage`, `Season`, `Irrigation_Type`, `Water_Source`, `Mulching_Used`, `Region`
- 11 numerics: `Soil_pH`, `Soil_Moisture`, `Organic_Carbon`, `Electrical_Conductivity`, `Temperature_C`, `Humidity`, `Rainfall_mm`, `Sunlight_Hours`, `Wind_Speed_kmh`, `Field_Area_hectare`, `Previous_Irrigation_mm`

---

## The deterministic formula on the original data

On the clean 10K, the target is a perfect function of 7 features:

```
score = 2*(Soil_Moisture < 25) + 2*(Rainfall_mm < 300)
      +   (Temperature_C > 30) +   (Wind_Speed_kmh > 10)
      - 2*(Crop_Growth_Stage == "Harvest")
      - 2*(Crop_Growth_Stage == "Sowing")
      -   (Mulching_Used == "Yes")
```

Mapping (verified row-by-row on `irrigation_prediction.csv`):

| `score` | n in orig | Class |
|---------|-----------|-------|
| -3 | 410 | Low (100%) |
| -2 | 1293 | Low (100%) |
| -1 | 1846 | Low (100%) |
| 0 | 2315 | Low (100%) |
| +1 | 1890 | Medium (100%) |
| +2 | 1244 | Medium (100%) |
| +3 | 666 | Medium (100%) |
| +4 | 255 | High (100%) |
| +5 | 61 | High (100%) |
| +6 | 20 | High (100%) |

So on the original: `score ≤ 0 → Low`, `1 ≤ score ≤ 3 → Medium`, `score ≥ 4 → High`.

On the noisy synthetic train/test, the same formula is no longer deterministic — added noise on the numeric features pushes some rows across the cut points, so a model is required.

---

## Final architecture

The submitted model is round 48 (`R48_multiclass_te.py`).

### Features (540 total)

1. **Raw originals** — the 11 numerics + 8 factorised categoricals (19 cols).
2. **Formula score** — the integer score from above, shifted to `[0, 9]` for safe categorical-style use.
3. **Rounded numerics** — each numeric rounded to a sensible precision based on the orig range (`abs.max < 10 → 3 dp`, `< 100 → 2 dp`, `≥ 100 → 1 dp`). 11 cols.
4. **Digit features** — for each numeric and each digit position `k ∈ [-3, 3]`, the digit at that position. 77 cols. These let the model see noise structure separately from magnitude.
5. **Digit pair features** — for the 4 formula-critical numerics (`Soil_Moisture`, `Rainfall_mm`, `Temperature_C`, `Wind_Speed_kmh`), every ordered pair of digits combined as a 2-digit integer. 84 cols.
6. **Multiclass target encoding (MTE)** from the original 10K — the breakthrough. For each of 108 base features (formula score + cats + nums + rounded nums + all digit cols) and each of the 3 classes, compute `P(class=c | feature=x)` as an unbiased estimate from `irrigation_prediction.csv`. 324 cols.
7. **Pairwise MTE** — `formula_score × each categorical`, again multiclass. 24 cols.

### Model

XGBoost classifier with "competition-grade" hyperparameters:

```python
XGBClassifier(
    max_depth=4, max_leaves=30,
    n_estimators=5000, learning_rate=0.1,
    reg_alpha=5, reg_lambda=5, min_child_weight=2,
    subsample=0.9, colsample_bytree=0.9,
    max_bin=10000,                      # critical
    eval_metric='mlogloss',
    early_stopping_rounds=500,
    tree_method='hist',
    num_class=3, random_state=42,
)
```

Trained with **balanced sample weights** (`len(y) / (n_classes * class_count)[y]`) so the optimiser doesn't ignore the 3.3% High class.

5-fold StratifiedKFold, single seed (multi-seed ensembling was tried and consistently *hurt* — see R47/R49 below).

### Post-processing

Threshold optimisation on OOF probabilities. 200 random Dirichlet starts, Nelder-Mead minimising negative balanced accuracy, applied as multiplicative weights on the (normalised) class probabilities before argmax. Final weights: `[High=0.8241, Low=0.0848, Medium=0.1109]`.

This step was non-negotiable. Submitting raw argmax (no threshold weights) collapsed the LB from **0.97785 → 0.97067** because High recall fell off a cliff (R56 in the rounds log).

### Validation strategy

| Metric | Value |
|--------|-------|
| OOF raw argmax | 0.971869 |
| OOF threshold-optimised | 0.977048 |
| Public LB | 0.97785 |

CV-LB gap was tight and consistent throughout — about +0.0008 LB above OOF-thresh. No leakage detected.

### Runtime

- Feature engineering: ~10 minutes
- 5 folds × ~1.5 hours = ~7.5 hours XGB training
- Threshold opt: ~2 minutes
- **Total: ~7.6 hours**, single seed, single CPU machine.

---

## What didn't work — and why

We ran 50+ rounds of experiments. The interesting failures and what they taught us:

### Seed ensembling consistently hurt LB

| Round | Strategy | OOF | LB |
|-------|----------|-----|----|
| R42 | XGB compgrade, single seed | 0.976560 | 0.97655 |
| R47 | 4-seed average of R42 | 0.97653 | 0.97635 |
| R48 | XGB compgrade + MTE, single seed | 0.977048 | **0.97785** |
| R49 | 4-seed average of R48 | 0.977367 | 0.97765 |

Both 4-seed ensembles improved OOF but lost LB. Cause: the threshold optimiser fits 20 model-fold OOF combinations and finds weights that subtly overfit the OOF distribution. Single-seed weights generalise better. **Lesson: when post-processing weights matter as much as the model probs, ensemble noise hurts.**

### Pseudo-labelling at high keep-rate (R50b)

Took R48's test predictions, kept 267K of 270K rows where top1 prob exceeded per-class thresholds (Low 0.80, Medium 0.75, High 0.60), appended to training, retrained the full pipeline.

Result: OOF 0.977153 (+0.0001), LB **0.97707** (-0.00078).

Cause: at 99% keep rate the pseudo-labels are essentially R48's own predictions, so we trained the new model to memorise R48's mistakes. Self-distillation with no signal injection.

### Probability-blend with CatBoost (R44 + R48 → R51)

We had a separately-trained CatBoost model (R44, LB 0.97655) and tried blending its probabilities with R48's. OOF agreement was 99.57%. Best raw OOF blend at α=0.925 gave +0.00037 over R48 alone, but the threshold-optimised blend was strictly worse (0.97596 vs 0.97705). Not submitted.

### Binary High-vs-rest specialist (R53)

Trained a dedicated binary XGB for High class with `scale_pos_weight≈29`. AUC=0.999 across folds. Replaced R48's High probability column with the specialist's. Threshold-optimised blend OOF was 0.97689 — slightly worse than R48 alone. Not submitted.

The High class is too small for a specialist to find new signal that the multiclass model with balanced weights hasn't already extracted.

### Rainfall denoising (R54 / R55)

Hypothesis: the synthetic noise lives on the formula-critical numerics, particularly `Rainfall_mm`. Diagnostic confirmed it — `Rainfall_mm` has 19,308 unique values in train vs 9,813 in orig (~2× more, indicating injected noise). Soil_Moisture, Temperature, and Wind_Speed live on essentially the same grid as orig (99%+ snap rate).

So we tried snapping each Rainfall value to the nearest orig-grid value. **Zero rows crossed the `< 300` cut** because the noise amplitude is much smaller than the distance from any test row to the cut point. The noise exists but doesn't corrupt the formula.

### LGBM-compgrade (R57)

Same 540 features, same fold split, LGBM instead of XGB. Per-fold raw OOF was a hair below XGB. Best blend α with R48 at the raw-argmax level was **exactly 0** — pure R48 won. Threshold-optimised blend OOF was 0.977049 vs R48 alone 0.977048 (+0.000001). Not submitted.

LGBM and XGB on the same feature set with the same regularisation produce nearly identical predictions on this task.

### Prior correction (R50a)

Hypothesis: train and test class priors might differ from each other or from orig. Diagnostic: train and orig priors are essentially identical (Low 0.587 / 0.586, Medium 0.380 / 0.380, High 0.033 / 0.034). No room to exploit. Not submitted.

### Formula override (R52)

Hypothesis: where the noisy formula is unambiguous, we could override the model with the deterministic class. Diagnostic showed R48 already beats raw-formula in **every** score bucket — particularly at score=4 (R48 0.97 vs formula 0.91). The MTE features have absorbed everything the formula encoded.

---

## Rounds timeline

The 50+ rounds explored a lot. The ones that mattered:

| # | What | OOF | LB | Status |
|---|------|-----|------|---------|
| R31-R33 | Scalar TE_ORIG (mean of ordinal target) | ~0.9745 | ~0.974 | Foundation |
| R38 | + formula score feature | 0.97506 | 0.9750 | |
| R41 | LGBM with default params | ~0.9740 | — | LGBM defaults too weak |
| R42 | XGB with compgrade params + max_bin=10000 | 0.97656 | 0.97655 | First strong baseline |
| R43 | Confidence-gated label blend with anchor | — | 0.97722 | First clean blend win |
| R44 | CatBoost with same features | 0.97574 | 0.97655 | Diversity attempt |
| R47 | R42 with 4-seed average | 0.97653 | 0.97635 | Worse — seed ensembling hurts |
| R48 | **R42 + multiclass TE_ORIG (MTE)** | 0.97705 | **0.97785** | **Final submission** |
| R49 | R48 with 4-seed average | 0.97737 | 0.97765 | Worse — same lesson |
| R50b | Loose pseudo-labelling | 0.97715 | 0.97707 | Worse |
| R51 | R48+R44 prob blend | — | — | Skip (CV regression) |
| R53 | High specialist | — | — | Skip (CV regression) |
| R56 | R48 raw argmax (no threshold) | 0.97187 | 0.97067 | Confirmed threshold critical |
| R57 | LGBM-compgrade blend | 0.977049 | — | Skip (no diversity) |

---

## Repo layout

```
playground-series-s6e4-irrigation-need/
├── README.md                    ← you are here
├── src/
│   ├── R42_xgb_compgrade.py     ← XGB baseline with formula + scalar TE
│   ├── R44_catboost_stack.py    ← CatBoost variant for diversity tests
│   ├── R48_multiclass_te.py     ← FINAL SUBMISSION pipeline
│   ├── R52_formula_diagnostic.py← Confirms MTE absorbs the formula
│   ├── R54_noise_diag.py        ← Identifies Rainfall_mm as the only noisy num
│   ├── R55_rain_denoise_diag.py ← Confirms denoising can't change the answer
│   └── R57_lgbm_compgrade.py    ← LGBM alternative (didn't add diversity)
└── notes/
    ├── lessons.md               ← Compact list of what generalises to other PS tasks
    └── feature_inventory.md     ← The 540 features, grouped by purpose
```

## Reproducing the final submission

1. Place `train.csv`, `test.csv`, `irrigation_prediction.csv`, `sample_submission.csv` in `competition/data/` (paths in the script). Update `DATA_DIR` and `ROUNDS_DIR` at the top of the script.
2. `python src/R48_multiclass_te.py` — runs feature engineering and 5-fold XGB. Resumes from saved `*_folds_done.json` if interrupted.
3. Output: `R48_submission.csv` matching the LB 0.97785 result.

Dependencies: `numpy`, `pandas`, `xgboost>=2.0`, `scikit-learn`, `scipy`.

---

## Acknowledgements

- Kaggle community member who reverse-engineered the deterministic formula on the original data — that observation reframed the entire task.
- The `yunsuxiaozi` notebook for the digit-feature trick that gave gradient-boosted trees a way to see noise structure separately from feature magnitude.
