"""Fleet-constrained ("brownfield") planning: an existing fleet of older
accelerators plus a small number of new ones. Each phase may now be served by a
*mixed* pool: the scarce new instances take the largest request fraction they can
serve within the SLO; the remainder goes to the best configuration of the
existing fleet (Section 4.4). Requests are split deterministically with a
golden-ratio sequence, which preserves the burstiness of the trace.
"""
import math

import numpy as np

from sim import simulate_prefill, simulate_decode, run_aggregated
from planner import _min_count

GOLD = 0.6180339887498949


def split(wl, f):
    arr, isl, osl = wl
    u = (np.arange(len(arr)) * GOLD) % 1.0
    a = u < f
    return (arr[a], isl[a], osl[a]), (arr[~a], isl[~a], osl[~a])


def _H(wl_full):
    return wl_full[0][-1] - wl_full[0][0]


def _p99(x):
    return float(np.percentile(x, 99)) if len(x) else 0.0


def eval_prefill(plan, n, wl, slo, P, H):
    arr, isl, osl = wl
    if len(arr) == 0:
        return True, 0.0
    first, e, busy = simulate_prefill(plan, n, arr, isl, P, 16384, 0.0)
    k = len(arr)
    ok = _p99((first - arr)[int(.05 * k):int(.95 * k)]) <= slo.ttft
    return ok, float(np.sum(e))


def eval_decode(plan, n, bcap, wl, slo, P, H):
    arr, isl, osl = wl
    if len(arr) == 0:
        return True, 0.0
    fin, e, busy = simulate_decode(plan, n, arr.copy(), isl, osl, P, bcap, 0.0)
    k = len(arr)
    idx = np.arange(int(.05 * k), int(.95 * k))
    m = osl[idx] > 1
    tp = (fin[idx][m] - arr[idx][m]) / (osl[idx][m] - 1)
    return _p99(tp) <= slo.tpot * 0.97, e


def energy(dyn, plan, n, H, phi):
    return phi * n * plan.t * plan.acc.tdp * H + (1 - phi) * dyn


def mixed_phase(phase, new, n_new, old, wl, slo, P, phi=0.25, bcap_new=None, bcap_old=None):
    """Largest fraction f served by n_new instances of `new`; rest sized on `old`."""
    H = _H(wl)

    def ev(plan, n, sub, bcap):
        if phase == "pre":
            return eval_prefill(plan, n, sub, slo, P, H)
        return eval_decode(plan, n, bcap, sub, slo, P, H)

    f_best, dyn_new = 0.0, 0.0
    if n_new > 0:
        lo, hi = 0.0, 1.0
        ok1, d1 = ev(new, n_new, split(wl, 1.0)[0], bcap_new)
        if ok1:
            # the new instances alone suffice: deploy only as many as needed
            f_best, dyn_new = 1.0, d1
            memo_n = {n_new: (ok1, d1)}

            def feas_new(n):
                if n not in memo_n:
                    memo_n[n] = ev(new, n, split(wl, 1.0)[0], bcap_new)
                return memo_n[n][0]
            n_min = _min_count(feas_new, 1, hi=n_new)
            n_new, dyn_new = n_min, memo_n[n_min][1]
        else:
            for _ in range(7):
                mid = (lo + hi) / 2
                ok, d = ev(new, n_new, split(wl, mid)[0], bcap_new)
                if ok:
                    lo, f_best, dyn_new = mid, mid, d
                else:
                    hi = mid
    rest = split(wl, f_best)[1]
    n_old, dyn_old = 0, 0.0
    if len(rest[0]):
        lam = len(rest[0]) / H
        memo = {}

        def feas(n):
            if n not in memo:
                memo[n] = ev(old, n, rest, bcap_old)
            return memo[n][0]

        n_old = _min_count(feas, 1, hi=128)
        if n_old is None:
            return None
        dyn_old = memo[n_old][1]
    e = (energy(dyn_new, new, n_new, H, phi) if n_new else 0.0) + \
        (energy(dyn_old, old, n_old, H, phi) if n_old else 0.0)
    return dict(phase=phase, new=new.label(), n_new=n_new, f_new=f_best, old=old.label(),
                n_old=n_old, energy=e, gpus_new=n_new * new.t, gpus_old=n_old * old.t,
                kw=(n_new * new.t * new.acc.tdp + n_old * old.t * old.acc.tdp) / 1e3)


def mixed_agg(new, n_new, bcap_new, chunk_new, old, bcap_old, chunk_old, wl, slo, P, phi=0.25):
    """Aggregated (colocated) serving with a mixed fleet: new instances take the
    largest fraction they can serve within both SLOs; old instances serve the rest."""
    H = _H(wl)

    def ok_run(plan, n, sub, bcap, chunk):
        if len(sub[0]) == 0:
            return True, 0.0
        r = run_aggregated(plan, n, bcap, sub, P, 0.0, chunk)
        s = r.summary()
        return (s["ttft_p99"] <= slo.ttft and s["tpot_p99"] <= slo.tpot), r.energy_j

    f_best, dyn_new = 0.0, 0.0
    ok1, d1 = ok_run(new, n_new, split(wl, 1.0)[0], bcap_new, chunk_new) if n_new else (False, 0.0)
    if ok1:
        f_best, dyn_new = 1.0, d1
    elif n_new:
        lo, hi = 0.0, 1.0
        for _ in range(6):
            mid = (lo + hi) / 2
            ok, d = ok_run(new, n_new, split(wl, mid)[0], bcap_new, chunk_new)
            if ok:
                lo, f_best, dyn_new = mid, mid, d
            else:
                hi = mid
    rest = split(wl, f_best)[1]
    n_old, dyn_old = 0, 0.0
    if len(rest[0]):
        memo = {}

        def feas(n):
            if n not in memo:
                memo[n] = ok_run(old, n, rest, bcap_old, chunk_old)
            return memo[n][0]

        n_old = _min_count(feas, 1, hi=128)
        if n_old is None:
            return None
        dyn_old = memo[n_old][1]
    e = (energy(dyn_new, new, n_new, H, phi) if n_new else 0.0) + \
        (energy(dyn_old, old, n_old, H, phi) if n_old else 0.0)
    return dict(new=new.label(), n_new=n_new, f_new=f_best, old=old.label(), n_old=n_old,
                energy=e, gpus_new=n_new * new.t, gpus_old=n_old * old.t,
                kw=(n_new * new.t * new.acc.tdp + n_old * old.t * old.acc.tdp) / 1e3)
