# Feature inventory — R48 final pipeline

540 features total, broken down by purpose.

| Group | n | Description |
|-------|---|-------------|
| Raw numerics | 11 | `Soil_pH`, `Soil_Moisture`, `Organic_Carbon`, `Electrical_Conductivity`, `Temperature_C`, `Humidity`, `Rainfall_mm`, `Sunlight_Hours`, `Wind_Speed_kmh`, `Field_Area_hectare`, `Previous_Irrigation_mm` |
| Factorised categoricals | 8 | `Soil_Type`, `Crop_Type`, `Crop_Growth_Stage`, `Season`, `Irrigation_Type`, `Water_Source`, `Mulching_Used`, `Region` |
| Formula score (shifted) | 1 | Chris-formula score + 3, range [0,9] |
| Rounded numerics | 11 | Each numeric rounded based on orig granularity (`abs.max<10 → 3dp`, `<100 → 2dp`, `else 1dp`) |
| Digit features | 77 | For each numeric, digit at positions k ∈ {-3, -2, -1, 0, 1, 2, 3} |
| Digit pair features | 84 | For each of {SM, Rain, Temp, Wind}, every ordered pair of digits combined as 2-digit ints (C(7,2)=21 per feature × 4) |
| MTE on base features | 324 | For each of 108 base features (formula + 8 cats + 11 nums + 11 rounded + 77 digits) and each of 3 classes, P(class=c \| feat=x) computed from clean orig 10K |
| Pairwise MTE | 24 | `formula_score × CAT` (8 cats × 3 classes), again multiclass TE |
| **Total** | **540** | |

The MTE features (348 total = 324 base + 24 pairwise) make up 64% of the feature set and carry most of the predictive signal. The XGB feature importances confirmed this: top-20 features by total gain were dominated by `MTE_formula_score_c*`, `MTE_fxcat_*`, and a handful of digit features on the formula-critical numerics.

## Why digit features matter

The synthetic noise on numeric features is concentrated at specific decimal positions. A digit at position `-2` (hundredths) carries different information than a digit at `+1` (tens). Splitting them into separate columns lets the tree splitter find boundaries on the noise dimension itself, rather than smearing the noise into the magnitude axis.

Empirically: removing the digit features (kept only raw + rounded + MTE) dropped CV by ~0.0008.

## Why pairwise MTE matters

`formula_score × CAT` captures cases where the formula's deterministic mapping breaks down conditional on a categorical. E.g. on the noisy train, `formula_score=4` is High most of the time, but with `Crop_Growth_Stage='Sowing'` it might shift toward Medium. The pairwise MTE encodes this conditional probability directly.

Removing the pairwise MTE dropped CV by ~0.0003.
