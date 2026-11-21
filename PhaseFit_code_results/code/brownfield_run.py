"""Case study 1: adding scarce Blackwell GPUs to an existing Hopper fleet
(Llama-3.1-70B, Azure-Conv x40, Standard SLO). Writes ../results/brownfield.json"""
import json, sys, time, itertools
sys.path.insert(0, ".")
from params import FITTED as P
from perfmodel import Plan
from planner import decode_batch_cap
from brownfield import mixed_phase, mixed_agg
from experiments import workload, STD

PHI = 0.25
wl = workload("conv", 40)
slo = STD
M = "Llama-3.1-70B"
old_pre = Plan.make("H200", M, "fp8", 1)
old_dec = Plan.make("H200", M, "fp8", 2)
bc_old = decode_batch_cap(old_dec, wl, slo, P)
new = {p: Plan.make("B200", M, p, 1) for p in ("fp4", "fp8")}
bc_new = {p: decode_batch_cap(new[p], wl, slo, P) for p in new}
tok = float(wl[2].sum())
out = dict(meta=dict(model=M, workload="conv x40", slo=slo.__dict__, phi=PHI, old_pre=old_pre.label(),
                     old_dec=old_dec.label(), out_tokens=tok, n=len(wl[0])), runs=[])
cache = {}

def phase(ph, prec, n_new):
    key = (ph, prec, n_new)
    if key not in cache:
        if ph == "pre":
            cache[key] = mixed_phase("pre", new[prec], n_new, old_pre, wl, slo, P, PHI)
        else:
            cache[key] = mixed_phase("dec", new[prec], n_new, old_dec, wl, slo, P, PHI,
                                     bcap_new=bc_new[prec], bcap_old=bc_old)
    return cache[key]

t0 = time.time()
base_p = phase("pre", "fp8", 0); base_d = phase("dec", "fp8", 0)
out["old_only"] = dict(pre=base_p, dec=base_d, energy=base_p["energy"] + base_d["energy"])
print("old only", out["old_only"]["energy"] / tok * 1e3, "mJ/tok", flush=True)
for N in (8, 16, 32):
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        n_p = int(round(a * N))
        n_d = N - n_p
        for pp, pd in itertools.product(("fp4", "fp8"), ("fp4", "fp8")):
            rp = phase("pre", pp, n_p) if n_p else base_p
            rd = phase("dec", pd, n_d) if n_d else base_d
            e = rp["energy"] + rd["energy"]
            out["runs"].append(dict(N=N, alpha=a, prec_pre=pp, prec_dec=pd, pre=rp, dec=rd,
                                    energy=e, mJ_tok=e / tok * 1e3,
                                    gpus_new_used=rp["gpus_new"] * (n_p > 0) + rd["gpus_new"] * (n_d > 0),
                                    gpus_old=rp["gpus_old"] + rd["gpus_old"],
                                    kw=rp["kw"] + rd["kw"]))
            print(N, a, pp, pd, round(e / tok * 1e3, 1), "mJ/tok", rp["n_new"], round(rp["f_new"], 2),
                  rd["n_new"], round(rd["f_new"], 2), f"{time.time()-t0:.0f}s", flush=True)
        json.dump(out, open("../results/brownfield.json", "w"), indent=1, default=str)
# aggregated mixed baseline: B200 FP4 TP2 aggregated + H200 FP8 TP2 aggregated
import pickle
prof = pickle.load(open("../results/profile_A.pkl", "rb"))
ag = {a["label"]: a for a in prof["agg"]}
old_lab = min((a for a in prof["agg"] if a["acc"] in ("H100", "H200")), key=lambda a: a["e_dyn"] + PHI * a["e_sta"])["label"]
new_lab = min((a for a in prof["agg"] if a["acc"] == "B200"), key=lambda a: a["e_dyn"] + PHI * a["e_sta"])["label"]
ao, an = ag[old_lab], ag[new_lab]
agg_old, agg_new = ao["plan"], an["plan"]
out["agg_cfg"] = dict(old=old_lab, new=new_lab)
out["agg"] = {}
for N in (0, 8, 16, 32):
    r = mixed_agg(agg_new, N // agg_new.t, an["bcap"], an["chunk"], agg_old, ao["bcap"], ao["chunk"],
                  wl, slo, P, PHI)
    out["agg"][N] = r
    print("agg", N, r and round(r["energy"] / tok * 1e3, 1), flush=True)
    json.dump(out, open("../results/brownfield.json", "w"), indent=1, default=str)
print("done", time.time() - t0)
