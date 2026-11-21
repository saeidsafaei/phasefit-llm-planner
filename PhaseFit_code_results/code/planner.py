"""PhaseFit: SLO-aware joint selection of accelerator type, numeric precision and
parallelism for each phase of disaggregated LLM serving, plus constrained baselines.

Key property used (Section 4): with prefill/decode disaggregation, the p99 TTFT
depends only on the prefill pool and the p99 TPOT only on the decode pool (plus a
small KV-transfer term), so the joint problem separates into two per-phase
problems coupled only through (i) KV transfer bandwidth/format and (ii) the
pairing constraints of a deployment policy (same vendor, same precision, ...).
Each per-phase problem is solved by analytic screening followed by
simulation-guided minimal sizing (binary search on the instance count).
"""
import math
from dataclasses import dataclass

import numpy as np

from hw import ACCELS, PLANNER_ACCELS
from models import MODELS
from perfmodel import Plan
from sim import simulate_prefill, simulate_decode, run_disagg, run_aggregated, _metrics


@dataclass
class SLO:
    ttft: float
    tpot: float
    name: str = ""


def candidate_plans(model, allow_fp4=True, accels=PLANNER_ACCELS):
    m = MODELS[model]
    ts = (4, 8) if m.moe else (1, 2, 4, 8)
    out = []
    for a in accels:
        acc = ACCELS[a]
        for prec in ("bf16", "fp8", "fp4"):
            if not acc.supports(prec) or (prec == "fp4" and not allow_fp4):
                continue
            for t in ts:
                p = Plan.make(a, model, prec, t)
                if p.kv_capacity_tokens() <= 0:
                    continue
                out.append(p)
    return out


def _min_count(feasible, lo, hi=96):
    """Smallest n in [lo, hi] with feasible(n) True (feasible monotone); None if none."""
    lo = max(1, lo)
    n = lo
    while n <= hi and not feasible(n):
        n = n * 2 if n < 8 else n + max(1, n // 4)
    if n > hi:
        if not feasible(hi):
            return None
        n = hi
    # binary search between previous probe and n
    a, b = lo, n
    while a < b:
        mid = (a + b) // 2
        if feasible(mid):
            b = mid
        else:
            a = mid + 1
    return a


# Energy bookkeeping: simulations run with phi=0, which yields the *dynamic*
# energy  E_dyn = sum_dt t*TDP*u  (Eq. 12). With the provisioned static energy
# E_sta = n*t*TDP*H, the total for any idle fraction phi is
#     E(phi) = phi * E_sta + (1 - phi) * E_dyn .
def energy_at(x, phi):
    return phi * x["e_sta"] + (1.0 - phi) * x["e_dyn"]


def fix_static(prof):
    """Charge provisioned (static) power over the arrival window H only, for every
    method alike; work done while draining the last requests is still counted in E_dyn."""
    H = prof["meta"]["horizon"]
    for k in ("pre", "dec", "agg"):
        for r in prof.get(k, []):
            p = r["plan"]
            r["e_sta"] = r["n"] * p.t * p.acc.tdp * H
    return prof


# ------------------------------------------------------------------ prefill phase
def size_prefill(plan, wl, slo, P, tok_budget=16384):
    arr, isl, osl = wl
    lam = len(arr) / (arr[-1] - arr[0])
    T, _, _ = plan.prefill_time(tok_budget, float(np.sum(isl ** 2) / np.sum(isl)), P)
    mu = tok_budget / T
    p99 = float(np.percentile(isl, 99))
    if plan.prefill_time(p99, p99, P)[0] > slo.ttft:
        return None
    lo = max(1, math.ceil(lam * isl.mean() / mu))
    lo_trim, hi_trim = int(0.05 * len(arr)), int(0.95 * len(arr))
    memo = {}

    def run(n):
        if n not in memo:
            first, e, busy = simulate_prefill(plan, n, arr, isl, P, tok_budget, 0.0)
            ttft = (first - arr)[lo_trim:hi_trim]
            memo[n] = (float(np.percentile(ttft, 99)), first, float(np.sum(e)), float(np.sum(busy)))
        return memo[n]

    n = _min_count(lambda k: run(k)[0] <= slo.ttft, lo)
    if n is None:
        return None
    p99t, first, e, busy = run(n)
    H = arr[-1] - arr[0]
    return dict(plan=plan, n=n, ttft_p99=p99t, e_dyn=e, e_sta=n * plan.t * plan.acc.tdp * H,
                util=busy / (n * H), gpus=n * plan.t, kw=n * plan.t * plan.acc.tdp / 1e3)


# ------------------------------------------------------------------ decode phase
def decode_batch_cap(plan, wl, slo, P, margin=0.85, bmax=None):
    arr, isl, osl = wl
    if bmax is None:   # same per-rank batch limit as in calibration (2048 per rank)
        bmax = 2048 * (plan.t if plan.par == "dep" else 1)
    c = float(isl.mean() + osl.mean() * 0.75)
    cap_mem = plan.kv_capacity_tokens() / float(isl.mean() + osl.mean())
    best = 0
    b = 1
    while b <= min(bmax, cap_mem):
        if plan.decode_step(b, c, P)[0] <= margin * slo.tpot:
            best = b
            b = b + 1 if b < 16 else int(b * 1.1) + 1
        else:
            break
    return int(min(best, cap_mem))


def size_decode(plan, wl, slo, P):
    arr, isl, osl = wl
    bcap = decode_batch_cap(plan, wl, slo, P)
    if bcap < 1:
        return None
    ready = arr.copy()
    lam = len(arr) / (arr[-1] - arr[0])
    T = plan.decode_step(bcap, isl.mean() + osl.mean() / 2, P)[0]
    lo = max(1, math.ceil(lam * osl.mean() * T / bcap))
    lo_trim, hi_trim = int(0.05 * len(arr)), int(0.95 * len(arr))
    memo = {}

    def run(n):
        if n not in memo:
            fin, e, busy = simulate_decode(plan, n, ready, isl, osl, P, bcap, 0.0)
            idx = np.arange(lo_trim, hi_trim)
            m = osl[idx] > 1
            tp = (fin[idx][m] - ready[idx][m]) / (osl[idx][m] - 1)
            memo[n] = (float(np.percentile(tp, 99)), fin, e, busy)
        return memo[n]

    n = _min_count(lambda k: run(k)[0] <= slo.tpot * 0.97, lo)
    if n is None:
        return None
    p99, fin, e, busy = run(n)
    H = arr[-1] - arr[0]
    return dict(plan=plan, n=n, bcap=bcap, tpot_p99=p99, e_dyn=e,
                e_sta=n * plan.t * plan.acc.tdp * H, util=busy / (n * H),
                gpus=n * plan.t, kw=n * plan.t * plan.acc.tdp / 1e3)


# ------------------------------------------------------------------ aggregated baseline
def size_aggregated(plan, wl, slo, P):
    arr, isl, osl = wl
    c = float(isl.mean() + osl.mean() * 0.75)
    lam = len(arr) / (arr[-1] - arr[0])
    cap_mem = plan.kv_capacity_tokens() / float(isl.mean() + osl.mean())
    # choose (chunk, batch cap) analytically: fewest instances meeting the TPOT target
    choice = None
    for chunk in (256, 512, 1024, 2048):
        bcap, b = 0, 1
        while b <= min(2048 * (plan.t if plan.par == "dep" else 1), cap_mem):
            if plan.iteration(b, c, chunk, 2 * 1024 + chunk, P)[0] <= 0.85 * slo.tpot:
                bcap = b
                b = b + 1 if b < 16 else int(b * 1.1) + 1
            else:
                break
        if bcap < 1:
            continue
        T = plan.iteration(bcap, c, chunk, 2 * 1024 + chunk, P)[0]
        n_est = max(lam * osl.mean() * T / bcap, lam * isl.mean() * T / chunk)
        if choice is None or n_est < choice[0]:
            choice = (n_est, chunk, bcap)
    if choice is None:
        return None
    n_est, chunk, bcap = choice
    memo = {}

    def run(n):
        if n not in memo:
            memo[n] = run_aggregated(plan, n, bcap, wl, P, 0.0, chunk)
        return memo[n]

    def ok(n):
        s = run(n).summary()
        return s["ttft_p99"] <= slo.ttft and s["tpot_p99"] <= slo.tpot

    n = _min_count(ok, max(1, math.ceil(n_est)), hi=64)
    if n is None:
        return None
    r = run(n)
    H = arr[-1] - arr[0]
    return dict(plan=plan, n=n, bcap=bcap, chunk=chunk, summary=r.summary(), e_dyn=r.energy_j,
                e_sta=n * plan.t * plan.acc.tdp * H, gpus=r.n_gpus, kw=r.prov_kw)


# ------------------------------------------------------------------ policies
POLICIES = {
    "Dis-Homo": dict(same_acc=True, same_prec=True),
    "Dis-HW": dict(same_acc=False, same_prec=True, prec="fp8"),
    "Dis-Prec": dict(same_acc=True, same_prec=False),
    "PhaseFit-V": dict(same_acc=False, same_prec=False, same_vendor=True),
    "PhaseFit-X": dict(same_acc=False, same_prec=False, same_vendor=False),
}


def pair_ok(pp, dp, pol):
    if pol.get("same_acc") and pp.acc.name != dp.acc.name:
        return False
    if pol.get("same_prec") and pp.prec != dp.prec:
        return False
    if pol.get("prec") and (pp.prec != pol["prec"] or dp.prec != pol["prec"]):
        return False
    if pol.get("same_vendor") and pp.acc.vendor != dp.acc.vendor:
        return False
    return True


def shortlist(cands, est, key=lambda p: (p.acc.name, p.prec), factor=None):
    """Best configuration (by analytic estimate) of every group; optionally also any
    configuration within `factor` of the overall best."""
    best_all = min(v for v in est.values())
    groups = {}
    for p in cands:
        groups.setdefault(key(p), []).append((est[p], p))
    keep = []
    for lst in groups.values():
        lst.sort(key=lambda x: x[0])
        if lst[0][0] < math.inf:
            keep.append(lst[0][1])
    if factor:
        for p in cands:
            if p not in keep and est[p] <= factor * best_all:
                keep.append(p)
    return keep


def profile(model, wl, slo, P, verbose=False, with_agg=True, phases=("pre", "dec", "agg")):
    """Size every short-listed configuration for each phase (and aggregated)."""
    cands = candidate_plans(model, True)
    arr, isl, osl = wl
    lam = len(arr) / (arr[-1] - arr[0])

    def est_p(p):
        T = p.prefill_time(16384, 2048, P)[0]
        return math.ceil(lam * isl.mean() / (0.7 * 16384 / T)) * p.t * p.acc.tdp

    def est_d(p):
        b = decode_batch_cap(p, wl, slo, P)
        if b < 1:
            return math.inf
        T = p.decode_step(b, isl.mean() + osl.mean() / 2, P)[0]
        return math.ceil(lam * osl.mean() * T / (0.85 * b)) * p.t * p.acc.tdp

    ep = {p: est_p(p) for p in cands}
    ed = {p: est_d(p) for p in cands}
    pre, dec, agg = [], [], []
    # prefill simulations are cheap: size every candidate (all TP degrees matter
    # when the TTFT target is tight)
    for p in ([c for c in cands if ep[c] < math.inf] if "pre" in phases else []):
        r = size_prefill(p, wl, slo, P)
        if r:
            pre.append(r)
            if verbose:
                print("  P", p.label(), r["n"], flush=True)
    for p in (shortlist(cands, ed) if "dec" in phases else []):
        r = size_decode(p, wl, slo, P)
        if r:
            dec.append(r)
            if verbose:
                print("  D", p.label(), r["n"], r["bcap"], flush=True)
    if with_agg and "agg" in phases:
        ea = {p: ep[p] + ed[p] for p in cands}
        # aggregated sizing is costly: best config per accelerator + near-best ones
        for p in shortlist(cands, ea, key=lambda p: p.acc.name, factor=1.5):
            r = size_aggregated(p, wl, slo, P)
            if r:
                agg.append(r)
                if verbose:
                    print("  A", p.label(), r["n"], r["chunk"], r["bcap"], flush=True)
    return dict(pre=pre, dec=dec, agg=agg)


def _inv_ok(uses, inv):
    if inv is None:
        return True
    tot = {}
    for acc, g in uses:
        tot[acc] = tot.get(acc, 0) + g
    return all(tot[a] <= inv.get(a, 0) for a in tot)


def select(prof, policy, obj="energy", phi=0.25, inv=None, allow_fp4=True, accels=None):
    def allowed(x):
        p = x["plan"]
        if not allow_fp4 and p.prec == "fp4":
            return False
        if accels is not None and p.acc.name not in accels:
            return False
        return True

    def score(e, kw):
        return e if obj == "energy" else kw

    if policy == "Agg-Homo":
        best = None
        for a in prof["agg"]:
            if not allowed(a) or not _inv_ok([(a["plan"].acc.name, a["gpus"])], inv):
                continue
            e = energy_at(a, phi)
            if best is None or score(e, a["kw"]) < score(best["energy"], best["kw"]):
                best = dict(agg=a, energy=e, kw=a["kw"], gpus=a["gpus"])
        return best
    pol = POLICIES[policy]
    best = None
    for rp in prof["pre"]:
        if not allowed(rp):
            continue
        for rd in prof["dec"]:
            if not allowed(rd) or not pair_ok(rp["plan"], rd["plan"], pol):
                continue
            if not _inv_ok([(rp["plan"].acc.name, rp["gpus"]), (rd["plan"].acc.name, rd["gpus"])], inv):
                continue
            e = energy_at(rp, phi) + energy_at(rd, phi)
            kw = rp["kw"] + rd["kw"]
            if best is None or score(e, kw) < score(best["energy"], best["kw"]):
                best = dict(pre=rp, dec=rd, energy=e, kw=kw, gpus=rp["gpus"] + rd["gpus"])
    return best


def validate(sel, wl, P, phi=0.25, nic=None):
    """End-to-end simulation (prefill -> KV transfer -> decode) of a selected pair;
    adds decode instances if the coupled pipeline misses the TPOT target."""
    from hw import NIC_BW
    rp, rd = sel["pre"], sel["dec"]
    n_d = rd["n"]
    for _ in range(4):
        r = run_disagg(rp["plan"], rp["n"], rd["plan"], n_d, rd["bcap"], wl, P, phi,
                       nic or NIC_BW)
        if r.summary()["tpot_p99"] <= r_tpot_target(sel):
            break
        n_d += 1
    return r, n_d


def r_tpot_target(sel):
    return sel.get("slo_tpot", math.inf)
