"""R57: LGBM compgrade counterpart to R48's XGB.
Same 540 features, different algorithm. Prob-level blend with R48 + threshold opt.
"""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(line_buffering=True)
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import json, os, time, gc

DATA_DIR   = "./data"
ROUNDS_DIR = "./output"
LOG        = f"{ROUNDS_DIR}/R57_run.log"

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
N_CLASSES = 3

label_map = {'Low':0, 'Medium':1, 'High':2}
train[TARGET] = train[TARGET].map(label_map)
orig[TARGET]  = orig[TARGET].map(label_map)
le_sk = LabelEncoder(); y_sk = le_sk.fit_transform(pd.read_csv(f"{DATA_DIR}/train.csv")[TARGET])
y_train = train[TARGET].values

def formula_shifted(df):
    hp = (2*(df['Soil_Moisture']<25).astype(int) + 2*(df['Rainfall_mm']<300).astype(int)
          + (df['Temperature_C']>30).astype(int) + (df['Wind_Speed_kmh']>10).astype(int))
    lp = (2*(df['Crop_Growth_Stage']=='Harvest').astype(int)
          + 2*(df['Crop_Growth_Stage']=='Sowing').astype(int)
          + (df['Mulching_Used']=='Yes').astype(int))
    return (hp - lp + 3).astype('int8')

for df in (train, test, orig):
    df['formula_score'] = formula_shifted(df)

round_map = {c: (3 if orig[c].abs().max()<10 else (2 if orig[c].abs().max()<100 else 1)) for c in NUMS}
for df in (train, test, orig):
    for c, r in round_map.items():
        df[f"{c}_round"] = df[c].round(r).astype('float32')

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

DIGIT_PAIR_FEATS = ["Soil_Moisture","Rainfall_mm","Temperature_C","Wind_Speed_kmh"]
digit_pair_cols = []
for c in DIGIT_PAIR_FEATS:
    for i, k1 in enumerate(DIGIT_K):
        for k2 in DIGIT_K[i+1:]:
            col = f"{c}_dpair_{k1}_{k2}"
            digit_pair_cols.append(col)
            for df in (train, test, orig):
                df[col] = (df[f"{c}_digit{k1}"].astype('int32')*10
                           + df[f"{c}_digit{k2}"].astype('int32')).astype('int16')

combined_cats = pd.concat([train[CATS], test[CATS], orig[CATS]], ignore_index=True)
for c in CATS:
    codes, _ = pd.factorize(combined_cats[c])
    n_tr = len(train); n_te = len(test)
    train[c] = codes[:n_tr].astype('int32')
    test[c]  = codes[n_tr:n_tr+n_te].astype('int32')
    orig[c]  = codes[n_tr+n_te:].astype('int32')

global_frac = np.array([(orig[TARGET]==c).mean() for c in range(N_CLASSES)], dtype='float32')
TE_TARGETS = ['formula_score'] + CATS + NUMS + [f"{c}_round" for c in NUMS] + digit_cols
log(f"Computing MTE on {len(TE_TARGETS)} features → {len(TE_TARGETS)*N_CLASSES} cols")

te_tr, te_te = {}, {}
for feat in TE_TARGETS:
    for cls in range(N_CLASSES):
        te_map = (orig.groupby(feat, observed=True)[TARGET]
                  .apply(lambda y: (y == cls).mean()).astype('float32'))
        nm = f"MTE_{feat}_c{cls}"
        te_tr[nm] = train[feat].map(te_map).fillna(global_frac[cls]).astype('float32').values
        te_te[nm] = test[feat].map(te_map).fillna(global_frac[cls]).astype('float32').values

train = pd.concat([train, pd.DataFrame(te_tr, index=train.index)], axis=1)
test  = pd.concat([test,  pd.DataFrame(te_te, index=test.index)],  axis=1)

pair_tr, pair_te = {}, {}
for c in CATS:
    orig_pair  = orig['formula_score'].astype(str)+'_'+orig[c].astype(str)
    train_pair = train['formula_score'].astype(str)+'_'+train[c].astype(str)
    test_pair  = test['formula_score'].astype(str)+'_'+test[c].astype(str)
    for cls in range(N_CLASSES):
        te_map = (orig.assign(_p=orig_pair).groupby('_p')[TARGET]
                  .apply(lambda y: (y==cls).mean()).astype('float32'))
        nm = f"MTE_fxcat_{c}_c{cls}"
        pair_tr[nm] = train_pair.map(te_map).fillna(global_frac[cls]).astype('float32').values
        pair_te[nm] = test_pair.map(te_map).fillna(global_frac[cls]).astype('float32').values
train = pd.concat([train, pd.DataFrame(pair_tr, index=train.index)], axis=1)
test  = pd.concat([test,  pd.DataFrame(pair_te, index=test.index)],  axis=1)

del orig; gc.collect()

FEATURE_COLS = (NUMS + CATS + ['formula_score']
                + [f"{c}_round" for c in NUMS]
                + digit_cols + digit_pair_cols
                + [c for c in train.columns if c.startswith('MTE_')])
FEATURE_COLS = list(dict.fromkeys([c for c in FEATURE_COLS if c in test.columns]))
log(f"Total features: {len(FEATURE_COLS)}")

X_train = train[FEATURE_COLS]; X_test = test[FEATURE_COLS]
log(f"X_train mem: {X_train.memory_usage(deep=True).sum()/1e9:.2f} GB")
del train, test; gc.collect()

def compute_sw(y, n=3):
    c = np.bincount(y, minlength=n).astype(float)
    return (len(y)/(n*c))[y].astype(np.float32)

# ---- 5-fold LGBM CV ----
log(f"\n5-fold LGBM (num_leaves=30, lr=0.05, feature_fraction=0.9, bagging=0.9, seed=42)...")
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_probs  = np.zeros((len(X_train), 3), dtype=np.float32)
test_accum = np.zeros((len(X_test),  3), dtype=np.float32)
scores = []

folds_file = f"{ROUNDS_DIR}/R57_folds_done.json"
done = 0
if os.path.exists(folds_file):
    d = json.load(open(folds_file))
    done = d.get("done", 0); scores = d.get("scores", [])
    if done > 0:
        oof_probs  = np.load(f"{ROUNDS_DIR}/R57_oof.npy")
        test_accum = np.load(f"{ROUNDS_DIR}/R57_test_accum.npy")
        log(f"Resuming from fold {done+1}")

lgb_params = dict(
    objective='multiclass', num_class=3,
    learning_rate=0.05,
    num_leaves=30,
    max_depth=-1,
    min_child_samples=20,
    reg_alpha=5.0, reg_lambda=5.0,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    max_bin=1023,
    verbose=-1,
    metric='multi_logloss',
    seed=42,
    n_jobs=-1,
)

for fi, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    if fi < done:
        log(f"  Fold {fi+1}: skipped"); continue
    t0 = time.time()
    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]
    sw = compute_sw(y_tr)

    dtr = lgb.Dataset(X_tr, label=y_tr, weight=sw)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    model = lgb.train(lgb_params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(300), lgb.log_evaluation(250)])

    va_probs = model.predict(X_va, num_iteration=model.best_iteration).astype(np.float32)
    te_probs = model.predict(X_test, num_iteration=model.best_iteration).astype(np.float32)
    oof_probs[va_idx] = va_probs
    test_accum += te_probs

    va_sk = va_probs[:, [2,0,1]]
    sc = balanced_accuracy_score(y_sk[va_idx], va_sk.argmax(1))
    scores.append(sc)
    np.save(f"{ROUNDS_DIR}/R57_oof.npy",  oof_probs)
    np.save(f"{ROUNDS_DIR}/R57_test_accum.npy", test_accum)
    json.dump({"done": fi+1, "scores": scores}, open(folds_file, "w"))
    log(f"  Fold {fi+1}: {sc:.6f}  iter={model.best_iteration}  ({(time.time()-t0)/60:.1f}m)")
    del model, X_tr, X_va, dtr, dva; gc.collect()

oof_sk = oof_probs[:, [2,0,1]]
raw_cv = balanced_accuracy_score(y_sk, oof_sk.argmax(1))
log(f"\nR57 OOF raw: {raw_cv:.6f}")

np.save(f"{ROUNDS_DIR}/R57_oof_sk.npy", oof_sk)

# ---- Blend with R48 ----
r48_oof_sk = np.load(f"{ROUNDS_DIR}/R48_oof_sk.npy")
r48_test_sk = (np.load(f"{ROUNDS_DIR}/R48_test_accum.npy")/5.0)[:, [2,0,1]]
r57_test_sk = (test_accum/N_FOLDS)[:, [2,0,1]]

log(f"\nα grid (blend = α*R57 + (1-α)*R48) on OOF raw argmax:")
best_raw = (balanced_accuracy_score(y_sk, r48_oof_sk.argmax(1)), 0.0)
for a in np.arange(0, 1.01, 0.025):
    b = a*oof_sk + (1-a)*r48_oof_sk
    sc = balanced_accuracy_score(y_sk, b.argmax(1))
    if sc > best_raw[0]: best_raw = (sc, a)
log(f"Best raw α={best_raw[1]:.3f}  OOF={best_raw[0]:.6f}")

# Threshold-optimize best blend
best_blend_oof = best_raw[1]*oof_sk + (1-best_raw[1])*r48_oof_sk
def neg_ba(w, p, y):
    return -balanced_accuracy_score(y, (p*np.array(w)).argmax(1))
rng = np.random.default_rng(0)
best_th = (best_raw[0], np.ones(3))
for _ in range(200):
    w0 = rng.dirichlet([2,2,2])*3
    res = minimize(neg_ba, w0, args=(best_blend_oof, y_sk), method='Nelder-Mead',
                   options={'maxiter':2000,'xatol':1e-6})
    if -res.fun > best_th[0]:
        w = np.abs(res.x)
        best_th = (-res.fun, w)
log(f"Blend thresh OOF: {best_th[0]:.6f}  w={best_th[1].round(4)}")
log(f"R48 alone thresh baseline: 0.977048")

# Apply to test
blend_test = best_raw[1]*r57_test_sk + (1-best_raw[1])*r48_test_sk
pred = le_sk.inverse_transform((blend_test*best_th[1]).argmax(1))
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub[TARGET] = pred
sub.to_csv(f"{ROUNDS_DIR}/R57_submission.csv", index=False)
log(f"R57 dist: {sub[TARGET].value_counts().to_dict()}")

# Also save R57-only (no blend) for reference
r57_only_pred = le_sk.inverse_transform((r57_test_sk*best_th[1]).argmax(1))
sub2 = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub2[TARGET] = r57_only_pred
sub2.to_csv(f"{ROUNDS_DIR}/R57_lgbm_only.csv", index=False)

# Compare vs R48
r48 = pd.read_csv(f"{ROUNDS_DIR}/R48_submission.csv")
for f in ['R57_submission.csv', 'R57_lgbm_only.csv']:
    s = pd.read_csv(f"{ROUNDS_DIR}/{f}")
    dis = (s[TARGET].values != r48[TARGET].values).sum()
    log(f"{f}: {dis} diffs vs R48 ({dis/len(s)*100:.3f}%)")

log(f'\nCV_SCORE={best_th[0]:.6f}')
log(f"Total runtime: {(time.time()-T_START)/60:.1f} min")
log("DONE")
