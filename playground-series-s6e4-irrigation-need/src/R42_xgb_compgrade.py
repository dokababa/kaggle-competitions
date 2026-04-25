"""
R42: XGBoost with competition-grade params + R41 feature stack.

R41 (LGBM defaults) gave CV=0.974 only. The critical missing piece is
max_bin=10000 which lets XGB use digit-like quantile splits — this is
what yunsuxiaozi-style notebooks exploit.

XGB params from competition top-scorer discussions:
  max_depth=4, max_leaves=30, max_bin=10000,
  reg_alpha=5, reg_lambda=5, min_child_weight=2,
  lr=0.1, n_estimators=5000, early_stopping=500

Features: same as R41 (formula + digits + digit pairs + TE_ORIG)
Fix: shift formula_score by +3 so it's in [0, 9] (no negative cat values).

Target: CV 0.977+ single model; runtime ~3-5 hours (5 folds, max_bin=10000
is slower than default 256).
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import json, os, time, gc

DATA_DIR   = "./data"
ROUNDS_DIR = "./output"
LOG        = f"{ROUNDS_DIR}/R42_run.log"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f: f.write(msg + "\n")

open(LOG, "w").close()
T_START = time.time()

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
orig  = pd.read_csv(f"{DATA_DIR}/irrigation_prediction.csv")
log(f"Train: {len(train)}, Test: {len(test)}, Orig: {len(orig)}")

TARGET = "Irrigation_Need"
CATS   = ["Soil_Type","Crop_Type","Crop_Growth_Stage","Season",
          "Irrigation_Type","Water_Source","Mulching_Used","Region"]
NUMS   = ["Soil_pH","Soil_Moisture","Organic_Carbon","Electrical_Conductivity",
          "Temperature_C","Humidity","Rainfall_mm","Sunlight_Hours",
          "Wind_Speed_kmh","Field_Area_hectare","Previous_Irrigation_mm"]

label_map_fwd = {'Low':0,'Medium':1,'High':2}
train[TARGET] = train[TARGET].map(label_map_fwd)
orig[TARGET]  = orig[TARGET].map(label_map_fwd)

le_sk = LabelEncoder()
y_sk = le_sk.fit_transform(pd.read_csv(f"{DATA_DIR}/train.csv")["Irrigation_Need"])

# -------- Deotte formula (shifted to [0..9] so no negative cat) --------
def compute_formula_shifted(df):
    high_part = (2*(df['Soil_Moisture']<25).astype(int)
                 + 2*(df['Rainfall_mm']<300).astype(int)
                 + (df['Temperature_C']>30).astype(int)
                 + (df['Wind_Speed_kmh']>10).astype(int))
    low_part  = (2*(df['Crop_Growth_Stage']=='Harvest').astype(int)
                 + 2*(df['Crop_Growth_Stage']=='Sowing').astype(int)
                 + (df['Mulching_Used']=='Yes').astype(int))
    return (high_part - low_part + 3).astype('int8')   # +3 shift → [0..9]

for df in (train, test, orig):
    df['formula_score'] = compute_formula_shifted(df)

log(f"Formula score (shifted) dist orig:\n{orig.groupby('formula_score')[TARGET].agg(['mean','count']).to_string()}")

# -------- Unsupervised rounding --------
round_map = {}
for c in NUMS:
    M = orig[c].abs().max()
    round_map[c] = 3 if M<10 else (2 if M<100 else 1)
for df in (train, test, orig):
    for c, r in round_map.items():
        df[f"{c}_round"] = df[c].round(r).astype('float32')

# -------- Digit features --------
DIGIT_K = list(range(-3, 4))   # -3..3
digit_cols = []
for c in NUMS:
    for k in DIGIT_K:
        col = f"{c}_digit{k}"
        digit_cols.append(col)
        if k >= 0:
            for df in (train, test, orig):
                df[col] = ((df[c] // (10**k)) % 10).astype('int8')
        else:
            for df in (train, test, orig):
                df[col] = ((df[c] * (10**(-k))).astype('int64') % 10).astype('int8')
log(f"Digit features: {len(digit_cols)}")

# -------- Digit pairs on the 4 critical continuous features --------
DIGIT_PAIR_FEATS = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh"]
digit_pair_cols = []
for c in DIGIT_PAIR_FEATS:
    ks = list(DIGIT_K)
    for i in range(len(ks)):
        for j in range(i+1, len(ks)):
            k1, k2 = ks[i], ks[j]
            col = f"{c}_dpair_{k1}_{k2}"
            digit_pair_cols.append(col)
            for df in (train, test, orig):
                df[col] = (df[f"{c}_digit{k1}"].astype('int32')*10
                           + df[f"{c}_digit{k2}"].astype('int32')).astype('int16')
log(f"Digit pair features: {len(digit_pair_cols)}")

# -------- Factorize categoricals --------
combined_cats = pd.concat([train[CATS], test[CATS], orig[CATS]], ignore_index=True)
for c in CATS:
    codes, _ = pd.factorize(combined_cats[c])
    n_tr = len(train); n_te = len(test)
    train[c] = codes[:n_tr].astype('int32')
    test[c]  = codes[n_tr:n_tr+n_te].astype('int32')
    orig[c]  = codes[n_tr+n_te:].astype('int32')

# -------- TE_ORIG on all engineered features --------
global_mean = orig[TARGET].mean()
TE_TARGETS = (
    ['formula_score'] + CATS + NUMS +
    [f"{c}_round" for c in NUMS] +
    digit_cols
)
log(f"Computing TE_ORIG on {len(TE_TARGETS)} features...")
te_cols_tr = {}
te_cols_te = {}
for c in TE_TARGETS:
    te_map = orig.groupby(c, observed=True)[TARGET].mean().astype('float32')
    name = f"TE_ORIG_{c}"
    te_cols_tr[name] = train[c].map(te_map).fillna(global_mean).astype('float32').values
    te_cols_te[name] = test[c].map(te_map).fillna(global_mean).astype('float32').values
train = pd.concat([train, pd.DataFrame(te_cols_tr, index=train.index)], axis=1)
test  = pd.concat([test,  pd.DataFrame(te_cols_te, index=test.index)],  axis=1)

# -------- Pairwise TE_ORIG on formula_score × each CAT (adds interaction signal) --------
log("Adding pairwise TE_ORIG(formula_score × cat)...")
pair_tr = {}; pair_te = {}
for c in CATS:
    orig_pair  = orig['formula_score'].astype(str) + '_' + orig[c].astype(str)
    train_pair = train['formula_score'].astype(str) + '_' + train[c].astype(str)
    test_pair  = test['formula_score'].astype(str)  + '_' + test[c].astype(str)
    te_map = orig.assign(_p=orig_pair).groupby('_p')[TARGET].mean().astype('float32')
    nm = f"TE_ORIG_formulaX{c}"
    pair_tr[nm] = train_pair.map(te_map).fillna(global_mean).astype('float32').values
    pair_te[nm] = test_pair.map(te_map).fillna(global_mean).astype('float32').values
train = pd.concat([train, pd.DataFrame(pair_tr, index=train.index)], axis=1)
test  = pd.concat([test,  pd.DataFrame(pair_te, index=test.index)],  axis=1)

# Free orig
del orig; gc.collect()

# -------- Final feature set --------
FEATURE_COLS = (
    NUMS + CATS +
    ['formula_score'] +
    [f"{c}_round" for c in NUMS] +
    digit_cols +
    digit_pair_cols +
    [c for c in train.columns if c.startswith('TE_ORIG_')]
)
FEATURE_COLS = [c for c in FEATURE_COLS if c in test.columns and c in train.columns]
FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))
log(f"Total features: {len(FEATURE_COLS)}")

X_train = train[FEATURE_COLS]
y_train = train[TARGET].values
X_test  = test[FEATURE_COLS]
log(f"X_train mem: {X_train.memory_usage(deep=True).sum()/1e9:.2f} GB")

del train, test; gc.collect()

def compute_sw(y, n_classes=3):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return (len(y) / (n_classes * counts))[y].astype(np.float32)

# -------- 5-fold XGB CV with competition-grade params --------
log(f"\n5-fold XGB CV (max_bin=10000, depth=4, alpha=5, lambda=5)...")
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_probs  = np.zeros((len(X_train), 3), dtype=np.float32)
test_accum = np.zeros((len(X_test), 3),  dtype=np.float32)
scores = []

folds_file = f"{ROUNDS_DIR}/R42_folds_done.json"
done = 0
if os.path.exists(folds_file):
    d = json.load(open(folds_file))
    done = d.get("done", 0); scores = d.get("scores", [])
    if done > 0:
        oof_probs  = np.load(f"{ROUNDS_DIR}/R42_oof.npy")
        test_accum = np.load(f"{ROUNDS_DIR}/R42_test_accum.npy")
        log(f"Resuming from fold {done+1}")

for fi, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    if fi < done:
        log(f"  Fold {fi+1}: skipped"); continue
    t0 = time.time()
    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]
    sw = compute_sw(y_tr)

    model = XGBClassifier(
        max_depth=4, max_leaves=30,
        n_estimators=5000, learning_rate=0.1,
        reg_alpha=5, reg_lambda=5, min_child_weight=2,
        subsample=0.9, colsample_bytree=0.9,
        max_bin=10000,
        eval_metric='mlogloss', early_stopping_rounds=500,
        tree_method='hist', device='cpu', n_jobs=-1,
        num_class=3, random_state=42, verbosity=1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], sample_weight=sw, verbose=250)

    va_probs = model.predict_proba(X_va).astype(np.float32)
    te_probs = model.predict_proba(X_test).astype(np.float32)
    oof_probs[va_idx] = va_probs
    test_accum += te_probs

    va_sk = va_probs[:, [2,0,1]]
    sc = balanced_accuracy_score(y_sk[va_idx], va_sk.argmax(1))
    scores.append(sc)
    np.save(f"{ROUNDS_DIR}/R42_oof.npy", oof_probs)
    np.save(f"{ROUNDS_DIR}/R42_test_accum.npy", test_accum)
    json.dump({"done": fi+1, "scores": scores}, open(folds_file, "w"))
    log(f"  Fold {fi+1}: {sc:.6f}  iter={model.best_iteration}  ({(time.time()-t0)/60:.1f}m)")
    del model, X_tr, X_va; gc.collect()

oof_sk = oof_probs[:, [2,0,1]]
raw_cv = balanced_accuracy_score(y_sk, oof_sk.argmax(1))
log(f"\nOOF CV (raw): {raw_cv:.6f}")

def thresh_opt(probs, y, n_trials=200):
    def neg_ba(w):
        return -balanced_accuracy_score(y, (probs * np.array(w)).argmax(1))
    best_val, best_w = 1, [1,1,1]
    np.random.seed(0)
    for _ in range(n_trials):
        w0 = np.random.dirichlet([2,2,2])*3
        res = minimize(neg_ba, w0, method='Nelder-Mead', options={'maxiter':2000,'xatol':1e-6})
        if res.fun < best_val:
            best_val = res.fun; best_w = res.x
    return -best_val, best_w

cv_opt, bw = thresh_opt(oof_sk, y_sk)
log(f"OOF CV (thresh): {cv_opt:.6f}  w={np.round(bw,4)}")

np.save(f"{ROUNDS_DIR}/R42_oof_sk.npy", oof_sk)
test_avg = test_accum / N_FOLDS
test_sk  = test_avg[:, [2,0,1]]
pred_labels = le_sk.inverse_transform((test_sk * np.array(bw)).argmax(1))
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub["Irrigation_Need"] = pred_labels
sub.to_csv(f"{ROUNDS_DIR}/R42_submission.csv", index=False)
log("Submission saved.")
log(sub["Irrigation_Need"].value_counts().to_string())
log(f"\nCV_SCORE={cv_opt:.6f}")
log(f"Total runtime: {(time.time()-T_START)/60:.1f} min")
log("DONE")
