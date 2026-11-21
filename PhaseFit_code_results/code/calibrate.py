"""Fit (eta_c, eta_m, tau_l) on the training records; report errors on all records."""
import json
import math
import numpy as np
from scipy.optimize import minimize

from perfmodel import Plan, Params, max_throughput_aggregated, request_latency
from calib_data import THR, LAT, is_train


def predict_thr(rec, P):
    src, model, acc, prec, par, t, i, o, v = rec
    plan = Plan.make(acc, model, prec, t, par)
    # MLPerf DeepSeek-R1 permits up to 20K output tokens; schedulers reserve that maximum
    res = 20000 if src == "MLP" else None
    thr, b = max_throughput_aggregated(plan, i, o, P, osl_reserve=res)
    return thr, b


def predict_lat(rec, P):
    src, model, acc, prec, par, t, b, i, o, v = rec
    plan = Plan.make(acc, model, prec, t, par)
    return request_latency(plan, b, i, o, P)


ETA_M = 0.85   # achievable HBM bandwidth fraction (fixed, not fitted)


def mkP(x):
    return Params(eta_c=x[0], eta_c4=x[1], m_half=x[2], eta_m=ETA_M, tau_l=x[3] * 1e-6, kappa=x[4])


def loss(x, recs):
    if min(x) <= 0 or x[0] > 1 or x[1] > 1 or x[4] < 1:
        return 1e9
    P = mkP(x)
    err = 0.0
    for r in recs:
        if len(r) == 9:
            p, _ = predict_thr(r, P)
            y = r[-1]
        else:
            p = predict_lat(r, P)
            y = r[-1]
        if p <= 0:
            return 1e9
        err += math.log(p / y) ** 2
    return err / len(recs)


def fit():
    train = [r for r in THR if is_train(r)]
    best = None
    for x0 in ([0.6, 0.45, 64, 20, 1.3], [0.8, 0.6, 200, 10, 2.0], [0.5, 0.3, 32, 100, 3.0]):
        res = minimize(loss, x0, args=(train,), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    return mkP([float(v) for v in best.x]), best.fun


def report(P):
    rows = []
    for r in THR:
        p, b = predict_thr(r, P)
        rows.append(dict(src=r[0], model=r[1], acc=r[2], prec=r[3], par=f"{r[4].upper()}{r[5]}",
                         isl=r[6], osl=r[7], metric="tok/s/GPU", meas=r[8], pred=round(p, 1),
                         batch=b, ape=abs(p - r[8]) / r[8] * 100, split="train" if is_train(r) else "test"))
    for r in LAT:
        p = predict_lat(r, P)
        rows.append(dict(src=r[0], model=r[1], acc=r[2], prec=r[3], par=f"{r[4].upper()}{r[5]}",
                         isl=r[7], osl=r[8], metric=f"latency s (B={r[6]})", meas=r[9], pred=round(p, 2),
                         batch=r[6], ape=abs(p - r[9]) / r[9] * 100, split="test"))
    return rows


if __name__ == "__main__":
    P, l = fit()
    print("fitted:", P, "train mean sq log err", l)
    rows = report(P)
    for r in rows:
        print(f"{r['split']:5s} {r['src']} {r['model']:16s} {r['acc']:6s} {r['prec']} {r['par']:5s} "
              f"{r['isl']:>5}/{r['osl']:<5} {r['metric']:18s} meas={r['meas']:>9} pred={r['pred']:>9} "
              f"B={r['batch']:>5} APE={r['ape']:5.1f}%")
    for split in ("train", "test"):
        e = [r["ape"] for r in rows if r["split"] == split]
        print(split, "n=", len(e), "MAPE=%.1f%%" % np.mean(e), "median=%.1f%%" % np.median(e))
    groups = {}
    for r in rows:
        groups.setdefault((r["src"], r["model"]), []).append(r["ape"])
    for k, v in groups.items():
        print(k, "n=", len(v), "MAPE=%.1f%%" % np.mean(v))
    json.dump(dict(params=P.__dict__, rows=rows), open("../results/calibration.json", "w"), indent=1)
