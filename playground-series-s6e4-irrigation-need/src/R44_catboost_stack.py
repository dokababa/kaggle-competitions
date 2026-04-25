"""
R44: CatBoost with R42's feature stack.

Goal: generate diversity vs R42 (XGB) and R50 (XGB ensemble) for 3-way blend.
CatBoost handles categoricals natively → often finds different decision
boundaries on the digit features than XGB's histogram splits.

Feature stack (same as R42):
  - Raw nums + cats
  - Deotte formula_score (shifted to [0,9])
  - Unsupervised rounded nums
  - Digit features (x // 10**k) % 10 for k in [-3..3]
  - Digit pairs on 4 critical features (SM, Rain, Temp, Wind)
  - TE_ORIG on all cats/nums/rounds/digits/formula
  - Pairwise TE_ORIG(formula × cat)

CatBoost params:
  depth=7, iter=5000, lr=0.05
  auto_class_weights=Balanced (handles class imbalance)
  l2_leaf_reg=5, random_strength=1

Target: CV 0.975+ single-model, test_accum saved for 3-way blend w/ R42+R50.
Runtime: ~4-5 hrs (CPU, 5 folds).
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import json, os, time, gc

DATA_DIR   = "./data"
ROUNDS_DIR = "./output"
LOG        = f"{ROUNDS_DIR}/R44_run.log"

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
log(f"sk classes: {list(le_sk.classes_)}")   # ['High','Low','Medium']

# ---- Deotte formula (shifted) ----
def compute_formula_shifted(df):
    high_part = (2*(df['Soil_Moisture']<25).astype(int)
                 + 2*(df['Rainfall_mm']<300).astype(int)
                 + (df['Temperature_C']>30).astype(int)
                 + (df['Wind_Speed_kmh']>10).astype(int))
    low_part  = (2*(df['Crop_Growth_Stage']=='Harvest').astype(int)
                 + 2*(df['Crop_Growth_Stage']=='Sowing').astype(int)
                 + (df['Mulching_Used']=='Yes').astype(int))
    return (high_part - low_part + 3).astype('int8')

for df in (train, test, orig):
    df['formula_score'] = compute_formula_shifted(df)

# ---- Unsupervised rounding ----
round_map = {}
for c in NUMS:
    M = orig[c].abs().max()
    round_map[c] = 3 if M<10 else (2 if M<100 else 1)
for df in (train, test, orig):
    for c, r in round_map.items():
        df[f"{c}_round"] = df[c].round(r).astype('float32')

# ---- Digit features ----
DIGIT_K = list(range(-3, 4))
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

# ---- Digit pairs on 4 critical features ----
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

# ---- Factorize categoricals ----
combined_cats = pd.concat([train[CATS], test[CATS], orig[CATS]], ignore_index=True)
for c in CATS:
    codes, _ = pd.factorize(combined_cats[c])
    n_tr = len(train); n_te = len(test)
    train[c] = codes[:n_tr].astype('int32')
    test[c]  = codes[n_tr:n_tr+n_te].astype('int32')
    orig[c]  = codes[n_tr+n_te:].astype('int32')

# ---- TE_ORIG ----
global_mean = orig[TARGET].mean()
TE_TARGETS = (
    ['formula_score'] + CATS + NUMS +
    [f"{c}_round" for c in NUMS] +
    digit_cols
)
log(f"Computing TE_ORIG on {len(TE_TARGETS)} features...")
te_cols_tr, te_cols_te = {}, {}
for c in TE_TARGETS:
    te_map = orig.groupby(c, observed=True)[TARGET].mean().astype('float32')
    name = f"TE_ORIG_{c}"
    te_cols_tr[name] = train[c].map(te_map).fillna(global_mean).astype('float32').values
    te_cols_te[name] = test[c].map(te_map).fillna(global_mean).astype('float32').values
train = pd.concat([train, pd.DataFrame(te_cols_tr, index=train.index)], axis=1)
test  = pd.concat([test,  pd.DataFrame(te_cols_te, index=test.index)],  axis=1)

# ---- Pairwise TE_ORIG(formula × cat) ----
pair_tr, pair_te = {}, {}
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

del orig; gc.collect()

# ---- Feature set ----
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

# CatBoost needs categorical indices — ONLY orig cats + formula.
# Digits and digit_pairs are kept as NUMERIC (we already have TE_ORIG on them,
# passing as cat is redundant and 10x slower).
CAT_COLS = CATS + ['formula_score']
CAT_COLS = [c for c in CAT_COLS if c in FEATURE_COLS]
CAT_IDX = [FEATURE_COLS.index(c) for c in CAT_COLS]
# CatBoost requires categoricals as int or str — ensure they're int
for c in CAT_COLS:
    X_train.loc[:, c] = X_train[c].astype('int32')
    X_test.loc[:, c]  = X_test[c].astype('int32')

log(f"Categorical cols: {len(CAT_COLS)} ({len(CATS)} orig cats + formula); "
    f"digits/pairs kept as numeric")
log(f"X_train mem: {X_train.memory_usage(deep=True).sum()/1e9:.2f} GB")

del train, test; gc.collect()

def compute_sw(y, n_classes=3):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return (len(y) / (n_classes * counts))[y].astype(np.float32)

# ---- 5-fold CatBoost CV ----
log(f"\n5-fold CatBoost CV (depth=7, iter=5000, lr=0.05)...")
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_probs  = np.zeros((len(X_train), 3), dtype=np.float32)
test_accum = np.zeros((len(X_test), 3),  dtype=np.float32)
scores = []

folds_file = f"{ROUNDS_DIR}/R44_folds_done.json"
done = 0
if os.path.exists(folds_file):
    d = json.load(open(folds_file))
    done = d.get("done", 0); scores = d.get("scores", [])
    if done > 0:
        oof_probs  = np.load(f"{ROUNDS_DIR}/R44_oof.npy")
        test_accum = np.load(f"{ROUNDS_DIR}/R44_test_accum.npy")
        log(f"Resuming from fold {done+1}")

for fi, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    if fi < done:
        log(f"  Fold {fi+1}: skipped"); continue
    t0 = time.time()
    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]
    sw = compute_sw(y_tr)

    tr_pool = Pool(X_tr, y_tr, cat_features=CAT_IDX, weight=sw)
    va_pool = Pool(X_va, y_va, cat_features=CAT_IDX)

    model = CatBoostClassifier(
        iterations=3000,
        depth=6,                   # reduced from 7 — faster; digit features are shallow-informative
        learning_rate=0.08,
        l2_leaf_reg=5,
        random_strength=1.0,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        bootstrap_type='Bernoulli',
        subsample=0.85,
        random_seed=42,
        early_stopping_rounds=150,
        task_type='CPU',
        thread_count=-1,
        verbose=200,
    )
    model.fit(tr_pool, eval_set=va_pool)

    va_probs = model.predict_proba(X_va).astype(np.float32)
    te_probs = model.predict_proba(X_test).astype(np.float32)
    oof_probs[va_idx] = va_probs
    test_accum += te_probs

    # CatBoost preserves class order from training labels (0=Low, 1=Medium, 2=High)
    # Convert to sk order [High=0, Low=1, Medium=2]: columns [2,0,1]
    va_sk = va_probs[:, [2,0,1]]
    sc = balanced_accuracy_score(y_sk[va_idx], va_sk.argmax(1))
    scores.append(sc)
    np.save(f"{ROUNDS_DIR}/R44_oof.npy", oof_probs)
    np.save(f"{ROUNDS_DIR}/R44_test_accum.npy", test_accum)
    json.dump({"done": fi+1, "scores": scores}, open(folds_file, "w"))
    log(f"  Fold {fi+1}: {sc:.6f}  iter={model.best_iteration_}  ({(time.time()-t0)/60:.1f}m)")
    del model, X_tr, X_va, tr_pool, va_pool; gc.collect()

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

np.save(f"{ROUNDS_DIR}/R44_oof_sk.npy", oof_sk)
test_avg = test_accum / N_FOLDS
test_sk  = test_avg[:, [2,0,1]]
pred_labels = le_sk.inverse_transform((test_sk * np.array(bw)).argmax(1))
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub["Irrigation_Need"] = pred_labels
sub.to_csv(f"{ROUNDS_DIR}/R44_submission.csv", index=False)
log("Submission saved.")
log(sub["Irrigation_Need"].value_counts().to_string())
log(f"\nCV_SCORE={cv_opt:.6f}")
log(f"Total runtime: {(time.time()-T_START)/60:.1f} min")
log("DONE")
