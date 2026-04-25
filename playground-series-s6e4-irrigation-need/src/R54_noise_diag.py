"""R54 DIAGNOSTIC: locate the synthetic noise structure.

Compare orig vs train distributions for the 4 formula-critical features:
  Soil_Moisture, Rainfall_mm, Temperature_C, Wind_Speed_kmh.
Specifically for each feature:
 - Unique value count in orig vs train
 - Digit-position entropy (is noise in d-1, d-2, d-3?)
 - Whether train values mod some step equals orig values
"""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(line_buffering=True)

D = './data/'
orig  = pd.read_csv(D+'irrigation_prediction.csv')
train = pd.read_csv(D+'train.csv')
test  = pd.read_csv(D+'test.csv')

FEATS = ["Soil_Moisture","Rainfall_mm","Temperature_C","Wind_Speed_kmh",
         "Soil_pH","Organic_Carbon","Electrical_Conductivity","Humidity",
         "Sunlight_Hours","Field_Area_hectare","Previous_Irrigation_mm"]

print("=== unique value counts ===")
print(f"{'feature':>26}  {'orig_uq':>8}  {'train_uq':>10}  {'test_uq':>10}  orig_range            train_range")
for f in FEATS:
    print(f"{f:>26}  {orig[f].nunique():>8}  {train[f].nunique():>10}  {test[f].nunique():>10}  "
          f"[{orig[f].min():>7.3f},{orig[f].max():>7.3f}]  [{train[f].min():>7.3f},{train[f].max():>7.3f}]")

print("\n=== digit-position entropy (train, test, orig) ===")
# For each feature, compute per-digit histogram to see which digits are ~uniform (noise)
FOCUS = ["Soil_Moisture","Rainfall_mm","Temperature_C","Wind_Speed_kmh"]
for f in FOCUS:
    print(f"\n-- {f} --")
    for k in range(-3, 4):
        if k >= 0:
            tr = ((train[f].values // (10**k)) % 10)
            te = ((test[f].values  // (10**k)) % 10)
            og = ((orig[f].values  // (10**k)) % 10)
        else:
            tr = ((train[f].values * (10**(-k))).astype('int64') % 10)
            te = ((test[f].values  * (10**(-k))).astype('int64') % 10)
            og = ((orig[f].values  * (10**(-k))).astype('int64') % 10)
        def entropy(x):
            _, c = np.unique(x, return_counts=True)
            p = c/c.sum()
            return -(p*np.log(p+1e-12)).sum()
        print(f"  digit k={k:+d}  H_orig={entropy(og):.3f}  H_train={entropy(tr):.3f}  H_test={entropy(te):.3f}")

print("\n=== orig value granularity (step) ===")
for f in FOCUS:
    og = np.sort(orig[f].unique())
    diffs = np.diff(og)
    print(f"{f:>26}  step_min={diffs.min():.4f}  step_median={np.median(diffs):.4f}  step_max={diffs.max():.4f}  "
          f"n_unique_orig={len(og)}")

print("\n=== does train value, rounded to orig step, equal an orig value? ===")
for f in FOCUS:
    og = np.sort(orig[f].unique())
    # infer step as the mode of the diffs, rounded
    diffs = np.diff(og)
    step = np.round(np.median(diffs), 3)
    if step <= 0: step = 0.01
    # snap train to multiples of step
    tr_snap = np.round(train[f].values / step) * step
    matches = np.isin(np.round(tr_snap, 3), np.round(og, 3)).mean()
    print(f"{f:>26}  inferred_step={step:.3f}  train matches orig grid after snap: {matches*100:.2f}%")
