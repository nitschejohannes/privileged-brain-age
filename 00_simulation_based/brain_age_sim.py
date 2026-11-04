#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
AGE_LO, AGE_HI = 20.0, 80.0     # chronological age range (years)
D          = 8                  # number of standard features
SIG_U      = 4.0                # sd of the generic aging offset u (years)
SIG_V_HC   = 4.0                # sd of the privileged offset v in controls
V0         = 1.0                # baseline privileged offset in cases (years)
BETA_SV    = 0.8                # severity -> privileged offset slope
SIG_V_RES  = 1.5                # residual sd of v in cases
SEV_HI     = 10.0               # severity range in cases
V_SCALE    = 1.30               # loading scale of the symptom channel q=v in x
SIG_X      = 3.0                # standard-feature measurement noise
SIG_XS     = 1.0                # privileged-feature measurement noise (clean)

_W_RNG = np.random.default_rng(20240517)
W1 = _W_RNG.uniform(0.7, 1.3, D)              # age-channel loadings
W2 = _W_RNG.uniform(-1.0, 1.0, D) * V_SCALE   # symptom-channel loadings


def make_cohort(n, kind, seed):
    """
    Generate one cohort.

    Parameters
    ----------
    n    : number of subjects
    kind : "hc" (healthy controls) or "case" (patients)
    seed : RNG seed

    Returns
    -------
    dict with keys:
      x    (n, D)  standard features        -- available at train and test
      xs   (n, 1)  privileged feature       -- teacher only
      age  (n,)    chronological age        -- regression target
      bag  (n,)    latent true brain-age gap = u + v
      sev  (n,)    symptom severity (NaN for controls)
    """
    rng = RNG(seed)
    age = rng.uniform(AGE_LO, AGE_HI, n)
    u   = rng.normal(0.0, SIG_U, n)

    if kind == "hc":
        v   = rng.normal(0.0, SIG_V_HC, n)
        sev = np.full(n, np.nan)
    elif kind == "case":
        sev = rng.uniform(0.0, SEV_HI, n)
        v   = V0 + BETA_SV * sev + rng.normal(0.0, SIG_V_RES, n)
    else:
        raise ValueError(kind)

    ba  = age + u + v                 # true brain age
    bag = u + v                       # true (latent) brain-age gap
    p   = age + u                     # age channel
    q   = v                           # symptom channel

    # standard features: fixed linear mixture of the two channels + noise
    x   = p[:, None] * W1[None, :] + q[:, None] * W2[None, :] \
          + rng.normal(0.0, SIG_X, (n, D))
    xs  = ba[:, None] + rng.normal(0.0, SIG_XS, (n, 1))

    return dict(x=x, xs=xs, age=age, bag=bag, sev=sev)


# ---------------------------------------------------------------------------
# PyTorch MLP student
# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    """1-hidden-layer ReLU MLP, scalar regression output."""
    def __init__(self, d_in, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                nn.init.zeros_(mod.bias)

    def forward(self, x):
        return self.net(x).squeeze(1)

def _train_student(X, y, soft=None, lam=0.0,
                   hidden=32, lr=1e-2, epochs=400, seed=0):
    """
    Train the student MLP with the generalized-distillation regression loss.

        loss = (1 - lam) * MSE(pred, y) + lam * MSE(pred, soft)

    X, y, soft are standardized (see gen_distillation). soft is ignored if
    lam == 0. Full-batch Adam, fixed schedule, no early stopping / no tuning.
    """
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    st = torch.tensor(soft, dtype=torch.float32) if soft is not None else None

    model = _MLP(X.shape[1], hidden=hidden)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    mse   = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = (1.0 - lam) * mse(pred, yt)
        if st is not None and lam > 0.0:
            loss = loss + lam * mse(pred, st)
        loss.backward()
        opt.step()

    model.eval()
    return model

def _predict(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()

# ---------------------------------------------------------------------------
# One normative-modelling run
# ---------------------------------------------------------------------------
def gen_distillation(train_hc, test_hc, test_case,
                     lam=0.8, hidden=32, lr=1e-2, epochs=400, seed=0):
    """
    Fit teacher / baseline / student on TRAIN controls and score the held-out
    controls and cases. Returns per-subject brain-age gaps and the fitted
    training summary.

    Returns
    -------
    dict with, for each model in {teacher, baseline, student}:
        bag_hc    (n_test_hc,)   BAG on held-out controls
        bag_case  (n_test_case,) BAG on cases
    plus:
        sev       (n_test_case,) case symptom severity
        age_hc, age_case         chronological ages (for reference)
    """
    Xtr,  ytr        = train_hc["x"],  train_hc["age"]
    XStr             = train_hc["xs"]
    Xhc,  age_hc     = test_hc["x"],   test_hc["age"]
    XShc             = test_hc["xs"]
    Xca,  age_ca     = test_case["x"], test_case["age"]
    XSca             = test_case["xs"]
    sev              = test_case["sev"]

    # --- standardization fitted on TRAIN controls only ---------------------
    sc_x  = StandardScaler().fit(Xtr)
    sc_xs = StandardScaler().fit(XStr)
    y_mu, y_sd = ytr.mean(), ytr.std()

    Ztr,  Zhc,  Zca  = sc_x.transform(Xtr),  sc_x.transform(Xhc),  sc_x.transform(Xca)
    ZStr, ZShc, ZSca = sc_xs.transform(XStr), sc_xs.transform(XShc), sc_xs.transform(XSca)
    ytr_z = (ytr - y_mu) / y_sd

    def bag(pred_z, age):
        return (pred_z * y_sd + y_mu) - age      # de-standardize, then gap

    # --- TEACHER: privileged features, kept at test (upper anchor) ---------
    teacher = MLPRegressor(hidden_layer_sizes=(hidden,), max_iter=1000,
                           random_state=seed).fit(ZStr, ytr_z)
    t_hc   = bag(teacher.predict(ZShc), age_hc)
    t_ca   = bag(teacher.predict(ZSca), age_ca)
    soft_z = teacher.predict(ZStr)                # soft brain-age targets (std units)

    # --- BASELINE student: standard features, hard labels only (lam = 0) ---
    base = _train_student(Ztr, ytr_z, lam=0.0,
                          hidden=hidden, lr=lr, epochs=epochs, seed=seed)
    b_hc = bag(_predict(base, Zhc), age_hc)
    b_ca = bag(_predict(base, Zca), age_ca)

    # --- STUDENT: standard features, distilling teacher soft brain ages ----
    stud = _train_student(Ztr, ytr_z, soft=soft_z, lam=lam,
                          hidden=hidden, lr=lr, epochs=epochs, seed=seed)
    s_hc = bag(_predict(stud, Zhc), age_hc)
    s_ca = bag(_predict(stud, Zca), age_ca)

    return {
        "teacher":  {"bag_hc": t_hc, "bag_case": t_ca},
        "baseline": {"bag_hc": b_hc, "bag_case": b_ca},
        "student":  {"bag_hc": s_hc, "bag_case": s_ca},
        "sev": sev, "age_hc": age_hc, "age_case": age_ca,
    }

# ---------------------------------------------------------------------------
# Multi-split runner
# ---------------------------------------------------------------------------
def run(reps=50, n_train=200, n_test_hc=500, n_test_case=500,
        lam=0.8, **kw):
    """
    Repeat the whole normative protocol over `reps` independent cohorts
    (fresh data and a clean train/test split each time).

    Returns
    -------
    splits : list of dicts (one per rep) as returned by gen_distillation
    summary: dict of arrays, per model:
        r_case        (reps,)  Pearson r between case BAG and severity
        slope_case    (reps,)  OLS slope of case BAG on severity (years / sev-unit)
        mean_bag_hc   (reps,)  mean BAG on held-out controls
        mean_bag_case (reps,)  mean BAG on cases
    """
    models = ["teacher", "baseline", "student"]
    summary = {m: {k: [] for k in
                   ("r_case", "slope_case", "mean_bag_hc", "mean_bag_case")}
               for m in models}
    splits = []

    for r in range(reps):
        # distinct seeds so the three cohorts never share draws
        train_hc  = make_cohort(n_train,     "hc",   seed=3 * r + 0)
        test_hc   = make_cohort(n_test_hc,   "hc",   seed=3 * r + 1)
        test_case = make_cohort(n_test_case, "case", seed=3 * r + 2)

        res = gen_distillation(train_hc, test_hc, test_case,
                               lam=lam, seed=r, **kw)
        splits.append(res)

        sev = res["sev"]
        for m in models:
            bc = res[m]["bag_case"]
            bh = res[m]["bag_hc"]
            r_case = np.corrcoef(bc, sev)[0, 1]
            slope  = np.polyfit(sev, bc, 1)[0]
            summary[m]["r_case"].append(r_case)
            summary[m]["slope_case"].append(slope)
            summary[m]["mean_bag_hc"].append(bh.mean())
            summary[m]["mean_bag_case"].append(bc.mean())

    for m in models:
        for k in summary[m]:
            summary[m][k] = np.asarray(summary[m][k])
    return splits, summary

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    _, summ = run(reps=30)
    hdr = f"{'model':<10}{'r(BAG,sev)':>14}{'slope':>12}{'mean BAG HC':>14}{'mean BAG case':>16}"
    print(hdr)
    print("-" * len(hdr))
    for m in ["teacher", "baseline", "student"]:
        s = summ[m]
        print(f"{m:<10}"
              f"{s['r_case'].mean():>8.3f}±{s['r_case'].std():>4.3f}"
              f"{s['slope_case'].mean():>12.3f}"
              f"{s['mean_bag_hc'].mean():>14.2f}"
              f"{s['mean_bag_case'].mean():>16.2f}")