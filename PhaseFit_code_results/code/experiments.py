"""Profile scenarios (per-phase and aggregated sizing) and store results.

usage: python3 experiments.py A B ...     (writes ../results/profile_<S>.pkl)
"""
import pickle
import sys
import time

from params import FITTED as P
from planner import profile, SLO
from workloads import azure, reasoning

STD = SLO(2.0, 0.100, "Standard")
INT = SLO(1.0, 0.040, "Interactive")
RSN = SLO(2.0, 0.080, "MLPerf-R1")

WINDOW = 1800.0   # first 30 min of each one-hour trace (original time base)


def workload(name, rate_x):
    if name == "reason":
        return reasoning(rate_x=rate_x, window=WINDOW / rate_x)
    return azure(name, rate_x=rate_x, window=WINDOW / rate_x)


SCEN = {
    "A": ("Llama-3.1-70B", "conv", 40, STD),
    "B": ("Llama-3.1-70B", "conv", 40, INT),
    "C": ("Llama-3.1-70B", "code", 40, STD),
    "D": ("Llama-3.1-70B", "code", 40, INT),
    "E": ("DeepSeek-R1", "reason", 4, RSN),
    "F": ("Qwen3-235B-A22B", "conv", 40, STD),
    "A10": ("Llama-3.1-70B", "conv", 10, STD),
    "A20": ("Llama-3.1-70B", "conv", 20, STD),
    "E8": ("DeepSeek-R1", "reason", 8, RSN),
}

PERTURB = {   # model-error robustness study (Section 6.4)
    "etac-20": dict(eta_c=0.8, eta_c4=0.8),
    "etam-12": dict(eta_m=0.75 / 0.85),
    "tau+100": dict(tau_l=2.0),
    "kappa+50": dict(kappa=1.5),
}


def strip(prof):
    for k in ("pre", "dec", "agg"):
        for r in prof[k]:
            r.pop("first", None)
            r["label"] = r["plan"].label()
            r["acc"] = r["plan"].acc.name
            r["prec"] = r["plan"].prec
            r["t"] = r["plan"].t
            r["vendor"] = r["plan"].acc.vendor
    return prof


if __name__ == "__main__":
    import os
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    phases = ("pre", "dec", "agg")
    tag = ""
    for a in sys.argv[1:]:
        if a.startswith("--phases="):
            phases = tuple(a.split("=")[1].split(","))
        if a.startswith("--perturb="):
            import copy
            tag = a.split("=")[1]
            P = copy.deepcopy(P)
            for k, f in PERTURB[tag].items():
                setattr(P, k, getattr(P, k) * f)
            print("perturbed params", P, flush=True)
    for s in args:
        model, wname, rx, slo = SCEN[s]
        wl = workload(wname, rx)
        t0 = time.time()
        print(f"[{s}] {model} {wname} x{rx} {slo.name} n={len(wl[0])} phases={phases}", flush=True)
        prof = profile(model, wl, slo, P, verbose=True, phases=phases)
        prof = strip(prof)
        fn = f"../results/profile_{s}{'_' + tag if tag else ''}.pkl"
        if os.path.exists(fn) and phases != ("pre", "dec", "agg"):
            old = pickle.load(open(fn, "rb"))
            for k in phases:
                old[k] = prof[k]
            prof = old
        prof["meta"] = dict(model=model, workload=wname, rate_x=rx, slo=slo.__dict__,
                            n=len(wl[0]), out_tokens=float(wl[2].sum()),
                            horizon=float(wl[0][-1] - wl[0][0]), secs=time.time() - t0)
        pickle.dump(prof, open(fn, "wb"))
        print(f"[{s}] done in {time.time() - t0:.0f}s", flush=True)
