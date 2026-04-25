"""R55 DIAG: Test whether Rainfall denoising (snap to orig grid) flips formula
score enough to justify a full retrain.
"""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(line_buffering=True)
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

R = './output/'
D = './data/'

orig  = pd.read_csv(D+'irrigation_prediction.csv')
train = pd.read_csv(D+'train.csv')
test  = pd.read_csv(D+'test.csv')
TARGET = 'Irrigation_Need'
le = LabelEncoder(); y_sk = le.fit_transform(train[TARGET])

orig_rain_sorted = np.sort(orig['Rainfall_mm'].unique())

def snap_to_grid(vals, grid):
    idx = np.searchsorted(grid, vals)
    idx = np.clip(idx, 0, len(grid)-1)
    left = grid[np.maximum(idx-1, 0)]
    right = grid[idx]
    # choose whichever is closer
    out = np.where(np.abs(vals-left) <= np.abs(vals-right), left, right)
    return out

train_rain_snap = snap_to_grid(train['Rainfall_mm'].values.astype(np.float64), orig_rain_sorted)
test_rain_snap  = snap_to_grid(test['Rainfall_mm'].values.astype(np.float64),  orig_rain_sorted)

print(f"Train: {(train['Rainfall_mm'].values != train_rain_snap).sum()} rows snapped ({(train['Rainfall_mm'].values != train_rain_snap).mean()*100:.2f}%)")
print(f"Test:  {(test['Rainfall_mm'].values != test_rain_snap).sum()} rows snapped ({(test['Rainfall_mm'].values != test_rain_snap).mean()*100:.2f}%)")

# How many cross the <300 boundary?
tr_old_lt300 = (train['Rainfall_mm'].values < 300)
tr_new_lt300 = (train_rain_snap < 300)
flips_tr = tr_old_lt300 != tr_new_lt300
print(f"Train: <300 bool flipped on {flips_tr.sum()} rows ({flips_tr.mean()*100:.3f}%)")

te_old_lt300 = (test['Rainfall_mm'].values < 300)
te_new_lt300 = (test_rain_snap < 300)
flips_te = te_old_lt300 != te_new_lt300
print(f"Test:  <300 bool flipped on {flips_te.sum()} rows ({flips_te.mean()*100:.3f}%)")

# Re-compute formula score using snapped rainfall on train
def formula_raw(rain_vals, df):
    hp = (2*(df['Soil_Moisture']<25).astype(int) + 2*(rain_vals<300).astype(int)
          + (df['Temperature_C']>30).astype(int) + (df['Wind_Speed_kmh']>10).astype(int))
    lp = (2*(df['Crop_Growth_Stage']=='Harvest').astype(int)
          + 2*(df['Crop_Growth_Stage']=='Sowing').astype(int)
          + (df['Mulching_Used']=='Yes').astype(int))
    return (hp - lp).values

score_raw    = formula_raw(train['Rainfall_mm'].values, train)
score_snap   = formula_raw(train_rain_snap,              train)

def score_to_class(s):
    return np.where(s <= 0, 1, np.where(s >= 4, 0, 2)).astype(np.int64)  # sk: Low=1, Med=2, High=0

raw_pred  = score_to_class(score_raw)
snap_pred = score_to_class(score_snap)

raw_acc  = balanced_accuracy_score(y_sk, raw_pred)
snap_acc = balanced_accuracy_score(y_sk, snap_pred)
print(f"\nFormula-only on RAW  Rainfall: balanced_acc = {raw_acc:.6f}")
print(f"Formula-only on SNAP Rainfall: balanced_acc = {snap_acc:.6f}")
print(f"Gain from snap: {snap_acc - raw_acc:+.6f}")

# Per-score-bucket: does R48 improve on denoised score?
r48_oof = np.load(R+'R48_oof_sk.npy')
r48_pred = r48_oof.argmax(1)
r48_acc = balanced_accuracy_score(y_sk, r48_pred)
print(f"\nR48 OOF raw: {r48_acc:.6f}")

print("\n=== Per-snap-score-bucket R48 accuracy ===")
for s in sorted(np.unique(score_snap)):
    m = score_snap == s
    if m.sum() < 50: continue
    r48_c = (r48_pred[m] == y_sk[m]).mean()
    snap_c = (snap_pred[m] == y_sk[m]).mean()
    print(f"  snap_score={s:+d}  n={m.sum():>7d}  R48_acc={r48_c:.4f}  snap_formula_acc={snap_c:.4f}  delta={snap_c-r48_c:+.4f}")

# Does denoising help the score=3 and score=4 buckets where R48 is weak?
# Check: for rows where SNAP score says High (≥4) and R48 says NOT High, who's right?
snap_high = snap_pred == 0  # sk High=0
r48_high  = r48_pred  == 0
mask = snap_high & ~r48_high
if mask.sum() > 0:
    snap_right = (y_sk[mask] == 0).mean()
    r48_right  = (y_sk[mask] == r48_pred[mask]).mean()
    print(f"\nRows where SNAP=High, R48!=High: n={mask.sum()}  snap_correct={snap_right:.4f}  r48_correct={r48_right:.4f}")

mask = ~snap_high & r48_high
if mask.sum() > 0:
    snap_right = (y_sk[mask] == snap_pred[mask]).mean()
    r48_right  = (y_sk[mask] == 0).mean()
    print(f"Rows where R48=High, SNAP!=High: n={mask.sum()}  snap_correct={snap_right:.4f}  r48_correct={r48_right:.4f}")

# Save denoised rainfall for use in R55 retrain
np.save(R+'train_rain_snap.npy', train_rain_snap)
np.save(R+'test_rain_snap.npy',  test_rain_snap)
print("\nSaved train_rain_snap.npy and test_rain_snap.npy")
