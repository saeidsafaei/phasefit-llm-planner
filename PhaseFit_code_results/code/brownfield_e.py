"""Case study 2b: scarce B200 GPUs added to an H200 fleet serving DeepSeek-R1 (Reasoning x4,
MLPerf-R1 SLO). Writes ../results/brownfield_E.json"""
import json, sys, time, itertools
sys.path.insert(0, ".")
from params import FITTED as P
from perfmodel import Plan
from planner import decode_batch_cap
from brownfield import mixed_phase
from experiments import workload, RSN

PHI = 0.25
wl = workload("reason", 4)
slo = RSN
M = "DeepSeek-R1"
old = Plan.make("H200", M, "fp8", 8)
bc_old = decode_batch_cap(old, wl, slo, P)
NEW = {("fp4", 4): Plan.make("B200", M, "fp4", 4), ("fp4", 8): Plan.make("B200", M, "fp4", 8),
       ("fp8", 8): Plan.make("B200", M, "fp8", 8)}
bc = {k: decode_batch_cap(p, wl, slo, P) for k, p in NEW.items()}
tok = float(wl[2].sum())
out = dict(meta=dict(model=M, workload="reason x4", slo=slo.__dict__, phi=PHI, old=old.label(),
                     out_tokens=tok, n=len(wl[0])), runs=[])
cache = {}
t0 = time.time()

def phase(ph, key, n_new):
    k = (ph, key, n_new)
    if k not in cache:
        if ph == "pre":
            cache[k] = mixed_phase("pre", NEW[key] if key else old, n_new, old, wl, slo, P, PHI)
        else:
            cache[k] = mixed_phase("dec", NEW[key] if key else old, n_new, old, wl, slo, P, PHI,
                                   bcap_new=bc[key] if key else bc_old, bcap_old=bc_old)
        print(ph, key, n_new, round(cache[k]["energy"] / tok * 1e3, 1), cache[k]["n_new"],
              round(cache[k]["f_new"], 2), cache[k]["n_old"], f"{time.time()-t0:.0f}s", flush=True)
    return cache[k]

base_p, base_d = phase("pre", None, 0), phase("dec", None, 0)
out["old_only"] = dict(pre=base_p, dec=base_d, energy=base_p["energy"] + base_d["energy"])
for N in (8, 16, 32):
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        for kp, kd in itertools.product(NEW, NEW):
            g_p = int(round(beta * N))
            n_p = g_p // kp[1]
            n_d = (N - n_p * kp[1]) // kd[1]
            rp = phase("pre", kp, n_p) if n_p else base_p
            rd = phase("dec", kd, n_d) if n_d else base_d
            e = rp["energy"] + rd["energy"]
            out["runs"].append(dict(N=N, alpha=beta, prec_pre=f"{kp[0]}/DEP{kp[1]}", prec_dec=f"{kd[0]}/DEP{kd[1]}",
                                    pre=rp, dec=rd, energy=e, mJ_tok=e / tok * 1e3,
                                    gpus_old=rp["gpus_old"] + rd["gpus_old"], kw=rp["kw"] + rd["kw"]))
    json.dump(out, open("../results/brownfield_E.json", "w"), indent=1, default=str)
print("done", time.time() - t0)
