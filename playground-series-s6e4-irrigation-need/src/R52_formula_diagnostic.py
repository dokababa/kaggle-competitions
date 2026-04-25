"""R52 DIAGNOSTIC: Compare R48 preds vs direct-formula preds on OOF.

The Chris formula on CLEAN orig: deterministic. On NOISY train/test: gives approximately
right answer but noise shifts some rows across cut points.

If formula > R48 on deep rows (score ≤ -1 or ≥ 5), we have a hard-override lever.
"""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(line_buffering=True)
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

R = './output/'
D = './data/'

train = pd.read_csv(D+'train.csv')
orig  = pd.read_csv(D+'irrigation_prediction.csv')
test  = pd.read_csv(D+'test.csv')
TARGET = 'Irrigation_Need'
le = LabelEncoder(); y_sk = le.fit_transform(train[TARGET])
print('sk classes:', list(le.classes_))  # [High, Low, Medium]

# Formula (raw, not shifted)
def formula_raw(df):
    hp = (2*(df['Soil_Moisture']<25).astype(int) + 2*(df['Rainfall_mm']<300).astype(int)
          + (df['Temperature_C']>30).astype(int) + (df['Wind_Speed_kmh']>10).astype(int))
    lp = (2*(df['Crop_Growth_Stage']=='Harvest').astype(int)
          + 2*(df['Crop_Growth_Stage']=='Sowing').astype(int)
          + (df['Mulching_Used']=='Yes').astype(int))
    return (hp - lp).values

score_orig  = formula_raw(orig)
score_train = formula_raw(train)
score_test  = formula_raw(test)

# Orig mapping
y_orig = orig[TARGET].values
print('\n=== CLEAN orig mapping: score → class ===')
for s in sorted(np.unique(score_orig)):
    m = score_orig == s
    dist = pd.Series(y_orig[m]).value_counts().to_dict()
    print(f'  score={s:+d}  n={m.sum():5d}  {dist}')

# Apply orig mapping to train: score ≤ 0 → Low, 1≤s≤3 → Medium, s ≥ 4 → High
def score_to_class(s):
    # returns sk-encoded: High=0, Low=1, Medium=2
    out = np.where(s <= 0, 1, np.where(s >= 4, 0, 2)).astype(np.int64)
    return out

formula_pred_train = score_to_class(score_train)
acc_formula = balanced_accuracy_score(y_sk, formula_pred_train)
print(f'\nFormula-only on train: balanced_acc = {acc_formula:.6f}')

# R48 OOF
r48_oof = np.load(R+'R48_oof_sk.npy')
r48_pred = r48_oof.argmax(1)
r48_acc  = balanced_accuracy_score(y_sk, r48_pred)
print(f'R48 OOF raw argmax: balanced_acc = {r48_acc:.6f}')

# Where does formula disagree with R48? Who wins?
disagree = formula_pred_train != r48_pred
print(f'\nFormula vs R48 disagreements: {disagree.sum()} ({disagree.mean()*100:.3f}%)')
for label, mask in [('Formula right, R48 wrong', (formula_pred_train==y_sk) & (r48_pred!=y_sk)),
                    ('R48 right, Formula wrong', (formula_pred_train!=y_sk) & (r48_pred==y_sk)),
                    ('Both wrong (differently)', (formula_pred_train!=y_sk) & (r48_pred!=y_sk) & (formula_pred_train!=r48_pred)),
                    ('Both right',               (formula_pred_train==y_sk) & (r48_pred==y_sk))]:
    print(f'  {label}: {mask.sum()}')

# Bucket by score: where is formula reliable?
print('\n=== Per-score-bucket analysis (train OOF) ===')
print(f'{"score":>6} {"n":>8} {"form_acc":>10} {"r48_acc":>10} {"override_gain":>14}')
for s in sorted(np.unique(score_train)):
    m = score_train == s
    if m.sum() == 0: continue
    form_correct = (formula_pred_train[m] == y_sk[m]).mean()
    r48_correct  = (r48_pred[m] == y_sk[m]).mean()
    gain = form_correct - r48_correct
    marker = ' ◀ override' if gain > 0.001 else ''
    print(f'{s:>+6d} {m.sum():>8d} {form_correct:>10.4f} {r48_correct:>10.4f} {gain:>+14.5f}{marker}')

# Now: if we override R48 with formula on score buckets where formula is ≥ R48,
# what's the combined OOF accuracy?
override_buckets = []
for s in sorted(np.unique(score_train)):
    m = score_train == s
    if m.sum() < 50: continue
    form_correct = (formula_pred_train[m] == y_sk[m]).mean()
    r48_correct  = (r48_pred[m] == y_sk[m]).mean()
    if form_correct > r48_correct:
        override_buckets.append(s)

print(f'\nOverride buckets (form > R48): {override_buckets}')
hybrid = r48_pred.copy()
override_mask = np.isin(score_train, override_buckets)
hybrid[override_mask] = formula_pred_train[override_mask]
hyb_acc = balanced_accuracy_score(y_sk, hybrid)
print(f'Hybrid OOF (R48 + formula-override on {override_mask.sum()} rows): {hyb_acc:.6f}')
print(f'R48 alone: {r48_acc:.6f}   →   delta={hyb_acc - r48_acc:+.6f}')

# R48 threshold-weighted baseline
bw = np.array([0.8241, 0.0848, 0.1109])
r48_th_pred = (r48_oof * bw).argmax(1)
r48_th_acc = balanced_accuracy_score(y_sk, r48_th_pred)
print(f'\nR48 OOF thresh-weighted: {r48_th_acc:.6f}')

# Try override on thresh-weighted preds
hybrid_th = r48_th_pred.copy()
for s in sorted(np.unique(score_train)):
    m = score_train == s
    if m.sum() < 50: continue
    form_correct = (formula_pred_train[m] == y_sk[m]).mean()
    r48_correct  = (r48_th_pred[m] == y_sk[m]).mean()
    if form_correct > r48_correct:
        hybrid_th[m] = formula_pred_train[m]
hyb_th_acc = balanced_accuracy_score(y_sk, hybrid_th)
ov = (hybrid_th != r48_th_pred).sum()
print(f'Hybrid OOF (R48_thresh + formula-override on {ov} rows): {hyb_th_acc:.6f}')
print(f'Delta vs R48_thresh: {hyb_th_acc - r48_th_acc:+.6f}')
