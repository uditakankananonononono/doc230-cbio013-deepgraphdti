"""CBIO013-G runner. Implements PREREG.md (tag prereg-lock-v2). Run once. Verbatim results to results/."""
import json, hashlib, os, sys, urllib.request, itertools
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFingerprintGenerator as rfg
import lightgbm as lgb
from scipy.stats import pearsonr
RDLogger.DisableLog('rdApp.*')
OUT = 'results'; os.makedirs(OUT, exist_ok=True); os.makedirs('data', exist_ok=True)
BASE = 'https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/'
AA = 'ACDEFGHIKLMNPQRSTVWY'

def fetch(ds, name):
    p = f'data/{ds}_{name}'
    if not os.path.exists(p):
        urllib.request.urlretrieve(BASE + f'{ds}/{name}', p)
    return p

def load(ds):
    import pickle
    lig = json.load(open(fetch(ds, 'ligands_can.txt')))
    prot = json.load(open(fetch(ds, 'proteins.txt')))
    Y = pickle.load(open(fetch(ds, 'Y'), 'rb'), encoding='latin1')
    Y = np.array(Y, dtype=float)
    dk, pk = list(lig.keys()), list(prot.keys())
    if ds == 'davis': Y = -np.log10(Y / 1e9)
    r, c = np.where(~np.isnan(Y))
    return dk, pk, [lig[k] for k in dk], [prot[k][:1000] for k in pk], r, c, Y[r, c], sum(len(prot[k]) > 1000 for k in pk)

def sha(ds):
    return {n: hashlib.sha256(open(f'data/{ds}_{n}', 'rb').read()).hexdigest() for n in ['ligands_can.txt', 'proteins.txt', 'Y']}

def lig_feats(smiles):
    g = rfg.GetMorganGenerator(radius=2, fpSize=1024)
    X = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None: X.append(np.zeros(1024 + len(Descriptors._descList))); continue
        fp = g.GetCountFingerprintAsNumPy(m).astype(float)
        d = []
        for n, f in Descriptors._descList:
            try: d.append(f(m))
            except Exception: d.append(0.0)
        X.append(np.concatenate([fp, np.nan_to_num(np.array(d, float), posinf=0, neginf=0)]))
    return np.array(X)

def p0(seqs):
    X = []
    for s in seqs:
        a = np.zeros(20); d = np.zeros(400)
        for ch in s:
            if ch in AA: a[AA.index(ch)] += 1
        for i in range(len(s) - 1):
            if s[i] in AA and s[i + 1] in AA: d[AA.index(s[i]) * 20 + AA.index(s[i + 1])] += 1
        X.append(np.concatenate([a / max(len(s), 1), d / max(len(s) - 1, 1)]))
    return np.array(X)

def p1(seqs):
    X = np.zeros((len(seqs), 1024))
    for i, s in enumerate(seqs):
        for j in range(len(s) - 2): X[i, hash(s[j:j + 3]) % 1024 if False else int(hashlib.md5(s[j:j + 3].encode()).hexdigest(), 16) % 1024] += 1
    return X

def p2(seqs):
    g = rfg.GetMorganGenerator(radius=2, fpSize=1024)
    X = []
    for s in seqs:
        m = Chem.MolFromSequence(''.join(ch if ch in AA else 'G' for ch in s))
        fp = g.GetCountFingerprintAsNumPy(m).astype(float)
        deg = np.bincount([a.GetDegree() for a in m.GetAtoms()], minlength=5)[:5]
        ri = m.GetRingInfo()
        X.append(np.concatenate([fp, [m.GetNumAtoms(), m.GetNumBonds(), ri.NumRings()], deg / m.GetNumAtoms()]))
    return np.array(X)

def hs(XL, XP, ri, ci):
    out = np.empty((len(ri), XL.shape[1] + XP.shape[1]), dtype=np.float32)
    out[:, :XL.shape[1]] = XL[ri]; out[:, XL.shape[1]:] = XP[ci]
    return out

NTRAIN = 20000
def fit_pred_idx(XL, XP, r, c, y, tr_mask, te_mask, seed):
    import gc
    idx = np.where(tr_mask)[0]
    if len(idx) > NTRAIN: idx = np.sort(np.random.default_rng(seed + 7).choice(idx, NTRAIN, replace=False))
    m = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, colsample_bytree=0.3, random_state=0, n_jobs=2, verbose=-1, max_bin=255)
    X = hs(XL, XP, r[idx], c[idx]); m.fit(X, y[idx]); del X; gc.collect()
    te = np.where(te_mask)[0]; out = np.empty(len(te))
    for i in range(0, len(te), 20000):
        j = te[i:i + 20000]; out[i:i + 20000] = m.predict(hs(XL, XP, r[j], c[j]))
    return out

def fit_pred(Xtr, ytr, Xte):
    m = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, colsample_bytree=0.3, random_state=0, n_jobs=2, verbose=-1)
    m.fit(Xtr, ytr); return m.predict(Xte)

def cindex(y, p, n=200000, seed=0):
    rng = np.random.default_rng(seed); N = len(y)
    i = rng.integers(0, N, n); j = rng.integers(0, N, n)
    k = y[i] != y[j]; i, j = i[k], j[k]
    s = np.sign(y[i] - y[j]) * np.sign(p[i] - p[j])
    return float(((s > 0) + 0.5 * (s == 0)).mean())

def metrics(y, p):
    return dict(r=float(pearsonr(y, p)[0]), rmse=float(np.sqrt(np.mean((y - p) ** 2))), ci=cindex(y, p))

def folds_cold(n_ent, seed, k=5):
    rng = np.random.default_rng(seed); perm = rng.permutation(n_ent)
    return [perm[i::k] for i in range(k)]

def run_cv(ds_tuple, featfn, seed, mode='target', perm_prot=False):
    dk, pk, ls, ps, r, c, y, _ = ds_tuple
    XL = np.clip(ds_tuple_cache['XL'], -1e30, 1e30).astype(np.float32); XP = np.clip(featfn, -1e30, 1e30).astype(np.float32)
    if perm_prot: XP = XP[np.random.default_rng(seed).permutation(len(XP))]
    pred = np.full(len(y), np.nan)
    if mode == 'target':
        for f in folds_cold(len(pk), seed):
            te = np.isin(c, f); tr = ~te
            pred[te] = fit_pred_idx(XL, XP, r, c, y, tr, te, seed)
        return y, pred
    if mode == 'double':
        pf = folds_cold(len(pk), seed); lf = folds_cold(len(dk), seed + 100)
        ys, ps_ = [], []
        for a, b in zip(pf, lf):
            te = np.isin(c, a) & np.isin(r, b); tr = ~np.isin(c, a) & ~np.isin(r, b)
            if te.sum() == 0: continue
            pr = fit_pred_idx(XL, XP, r, c, y, tr, te, seed)
            ys.append(y[te]); ps_.append(pr)
        return np.concatenate(ys), np.concatenate(ps_)

def ctrl_ligmean(ds_tuple, seed):
    dk, pk, ls, ps, r, c, y, _ = ds_tuple
    pred = np.full(len(y), np.nan)
    for f in folds_cold(len(pk), seed):
        te = np.isin(c, f); tr = ~te
        gm = y[tr].mean(); sums = np.bincount(r[tr], y[tr], len(dk)); cnt = np.bincount(r[tr], minlength=len(dk))
        lm = np.where(cnt > 0, sums / np.maximum(cnt, 1), gm)
        pred[te] = lm[r[te]]
    return y, pred

def agg(ms):
    return {k: [float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms]))] for k in ms[0]}

ds_tuple_cache = {}
def evaluate(ds, directions):
    t = load(ds); ds_tuple_cache['XL'] = lig_feats(t[2])
    P = {'P0': p0(t[3]), 'P1': p1(t[3]), 'P2': p2(t[3])}; P['P3'] = np.hstack([P['P0'], P['P2']])
    res = {'dataset': ds, 'n_pairs': int(len(t[6])), 'n_ligands': len(t[0]), 'n_proteins': len(t[1]), 'truncated_proteins': int(t[7]), 'sha256': sha(ds)}
    for name, (feat, mode, perm) in directions.items():
        ms = []
        for sd in (0, 1, 2):
            y, p = run_cv(t, P[feat], sd, mode, perm); ms.append(metrics(y, p))
        res[name] = {'reps': ms, 'mean_sd': agg(ms)}; print(ds, name, res[name]['mean_sd'], flush=True)
    return res, t, P

if __name__ == '__main__':
    kiba_dirs = {'D1': ('P0', 'target', False), 'D2': ('P1', 'target', False), 'D3': ('P2', 'target', False), 'D4': ('P3', 'target', False),
                 'D6': ('P2', 'target', True), 'D8_P0': ('P0', 'double', False), 'D8_P2': ('P2', 'double', False)}
    res, t, P = evaluate('kiba', kiba_dirs)
    ms = [metrics(*ctrl_ligmean(t, sd)) for sd in (0, 1, 2)]; res['D5'] = {'reps': ms, 'mean_sd': agg(ms)}
    # D10 applicability domain: per test protein max 3-mer Jaccard to training proteins, seed-0 folds, D3 predictions
    sets = [set(s[i:i + 3] for i in range(len(s) - 2)) for s in t[3]]
    f0 = folds_cold(len(t[1]), 0); hi = []; ad = {}
    for f in f0:
        tr_p = [i for i in range(len(sets)) if i not in set(f)]
        for i in f: ad[i] = max(len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for j in tr_p)
    y3, p3 = run_cv(t, P['P2'], 0, 'target')
    hiidx = np.array([ad[c] > 0.5 for c in t[5]])
    res['D10'] = {'frac_proteins_gt_0.5': float(np.mean([v > 0.5 for v in ad.values()])),
                  'r_high': float(pearsonr(y3[hiidx], p3[hiidx])[0]) if hiidx.sum() > 10 else None,
                  'r_low': float(pearsonr(y3[~hiidx], p3[~hiidx])[0]) if (~hiidx).sum() > 10 else None, 'n_high': int(hiidx.sum()), 'n_low': int((~hiidx).sum())}
    json.dump(res, open(f'{OUT}/kiba_results.json', 'w'), indent=1)
    dav, *_ = evaluate('davis', {'D1': ('P0', 'target', False), 'D3': ('P2', 'target', False)})
    json.dump(dav, open(f'{OUT}/davis_results.json', 'w'), indent=1)
    # gates
    r = lambda d, k='mean_sd': d[k]['r'][0]
    g = {'D3': r(res['D3']) - max(r(res['D1']), r(res['D2'])) > 0.01,
         'D4': r(res['D4']) - r(res['D1']) > 0.01,
         'D5': r(res['D1']) - r(res['D5']) > 0.05,
         'D6': r(res['D6']) < r(res['D3']) - 0.02,
         'D7': np.sign(r(dav['D3']) - r(dav['D1'])) == np.sign(r(res['D3']) - r(res['D1'])),
         'D8': r(res['D8_P2']) > r(res['D8_P0'])}
    json.dump({k: bool(v) for k, v in g.items()}, open(f'{OUT}/gates.json', 'w'), indent=1); print('GATES', g)
