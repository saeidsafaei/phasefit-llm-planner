"""Trace-driven discrete-event simulation of LLM serving.

Two deployment modes:
  * disaggregated : a prefill pool and a decode pool (possibly different accelerator
                    types / precisions / parallelism), KV cache moved over RDMA.
  * aggregated    : one pool; every iteration mixes all running decodes with a
                    prefill chunk (chunked prefill / stall-free batching).
Step durations and utilisations come from perfmodel.Plan.iteration. Between two
events (arrival, completion) the batch composition of an instance is constant, so
the simulator advances whole groups of identical steps at once ("macro steps").
Energy integrates P(u) over busy steps and the idle power over idle intervals.
"""
import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from perfmodel import Plan, Params, DEFAULT
from hw import NIC_BW, kv_bytes


@dataclass
class Result:
    ttft: np.ndarray
    tpot: np.ndarray
    energy_j: float
    out_tokens: int
    makespan: float
    n_gpus: int
    prov_kw: float
    detail: dict = field(default_factory=dict)

    def p99(self, x):
        return float(np.percentile(x, 99)) if len(x) else math.inf

    def summary(self):
        return dict(ttft_p50=float(np.median(self.ttft)), ttft_p99=self.p99(self.ttft),
                    tpot_p50=float(np.median(self.tpot)), tpot_p99=self.p99(self.tpot),
                    j_per_tok=self.energy_j / max(self.out_tokens, 1),
                    n_gpus=self.n_gpus, prov_kw=self.prov_kw, makespan=self.makespan)


# ---------------------------------------------------------------- prefill pool
def simulate_prefill(plan: Plan, n_inst: int, arr, isl, P: Params, tok_budget=16384,
                     phi=0.25):
    """FCFS batched prefill on n_inst instances, join-shortest-queue (by tokens).
    Returns per-request completion times and pool energy over [0, horizon]."""
    n = len(arr)
    done = np.zeros(n)
    free_at = np.zeros(n_inst)          # time each instance becomes idle
    queues = [[] for _ in range(n_inst)]
    qtok = np.zeros(n_inst)
    busy_energy = np.zeros(n_inst)
    busy_time = np.zeros(n_inst)
    # event-driven: process arrivals in order; each instance serves its queue
    # whenever idle. Because service is FCFS per instance, we can run each
    # instance lazily up to the current arrival time.
    heads = [0] * n_inst

    def run_until(k, t_limit):
        q = queues[k]
        while heads[k] < len(q):
            start = max(free_at[k], arr[q[heads[k]]])
            if start > t_limit:
                return
            # batch all queued requests that have arrived by `start`, within budget
            batch, toks, sq = [], 0, 0
            j = heads[k]
            while j < len(q) and arr[q[j]] <= start + 1e-12:
                s = isl[q[j]]
                if batch and toks + s > tok_budget:
                    break
                batch.append(q[j]); toks += s; sq += s * s
                j += 1
            T, uc, um = plan.iteration(0.0, 0.0, toks, sq / toks, P)
            end = start + T
            for r in batch:
                done[r] = end
                qtok[k] -= isl[r]
            busy_time[k] += T
            busy_energy[k] += T * plan.power(uc, um, phi)
            free_at[k] = end
            heads[k] = j

    t_tok = plan.prefill_time(tok_budget, 2048, P)[0] / tok_budget   # s per prompt token
    for r in range(n):
        t = arr[r]
        for k in range(n_inst):
            run_until(k, t)
        # join the instance with the smallest estimated backlog (seconds)
        k = int(np.argmin(np.maximum(free_at - t, 0) + qtok * t_tok))
        queues[k].append(r)
        qtok[k] += isl[r]
    for k in range(n_inst):
        run_until(k, math.inf)
    return done, busy_energy, busy_time


# ---------------------------------------------------------------- decode pool
class DecodeInst:
    __slots__ = ("t", "rem", "ctx", "rid", "res", "cap", "bcap", "energy", "busy", "pending")

    def __init__(self, cap, bcap):
        self.t = 0.0
        self.rem = []      # remaining tokens per active seq
        self.ctx = []
        self.rid = []
        self.res = 0.0     # reserved KV tokens
        self.cap = cap
        self.bcap = bcap
        self.energy = 0.0
        self.busy = 0.0
        self.pending = []  # (ready_time, rid) admitted to this instance, waiting for step boundary


def simulate_decode(plan: Plan, n_inst: int, ready, isl, osl, P: Params, bcap: int,
                    phi=0.25):
    """Continuous batching decode. `ready[r]` = time KV of request r is available.
    Returns finish times (time the last token is produced) and energy."""
    n = len(ready)
    order = np.argsort(ready, kind="stable")
    cap = plan.kv_capacity_tokens()
    insts = [DecodeInst(cap, bcap) for _ in range(n_inst)]
    finish = np.full(n, np.nan)
    waitq = []           # global FCFS queue of rids not yet placed
    first_tok = np.zeros(n)

    def step_time(inst):
        B = len(inst.rem)
        c = sum(inst.ctx) / B
        return plan.decode_step(B, c, P)

    def advance(inst, t_lim):
        """Run whole steps while next step ends <= t_lim, handling completions."""
        while inst.rem:
            T, uc, um = step_time(inst)
            k = min(inst.rem)                     # steps to next completion
            k_fit = int((t_lim - inst.t) // T) if t_lim < math.inf else k
            ksteps = min(k, k_fit)
            if ksteps <= 0:
                return
            # refine step time with mid-interval context
            B = len(inst.rem)
            c_mid = sum(inst.ctx) / B + ksteps / 2.0
            T, uc, um = plan.decode_step(B, c_mid, P)
            dt = ksteps * T
            inst.t += dt
            inst.busy += dt
            inst.energy += dt * plan.power(uc, um, phi)
            inst.rem = [x - ksteps for x in inst.rem]
            inst.ctx = [x + ksteps for x in inst.ctx]
            keep = [i for i, x in enumerate(inst.rem) if x > 0]
            for i, x in enumerate(inst.rem):
                if x <= 0:
                    r = inst.rid[i]
                    finish[r] = inst.t
                    inst.res -= isl[r] + osl[r]
            inst.rem = [inst.rem[i] for i in keep]
            inst.ctx = [inst.ctx[i] for i in keep]
            inst.rid = [inst.rid[i] for i in keep]
            if ksteps < k:   # stopped because of t_lim
                return
            if t_lim == math.inf:
                return

    def try_place(t_now):
        # place waiting requests FCFS onto least-loaded instance with room
        while waitq:
            r = waitq[0]
            need = isl[r] + osl[r]
            best, bestB = None, None
            for inst in insts:
                if len(inst.rem) < inst.bcap and inst.res + need <= inst.cap:
                    if best is None or len(inst.rem) < bestB:
                        best, bestB = inst, len(inst.rem)
            if best is None:
                return
            waitq.pop(0)
            # join at the instance's next step boundary
            if not best.rem:
                best.t = max(best.t, t_now)
            best.rem.append(osl[r] - 1)        # first token was produced by prefill
            best.ctx.append(isl[r] + 1)
            best.rid.append(r)
            best.res += need
            if osl[r] - 1 <= 0:
                finish[r] = best.t
                best.res -= need
                best.rem.pop(); best.ctx.pop(); best.rid.pop()

    for r in order:
        t = ready[r]
        for inst in insts:
            if inst.rem:
                advance(inst, t)
            else:
                inst.t = max(inst.t, t) if inst.t < t else inst.t
        waitq.append(r)
        try_place(t)
        # if nothing could be placed, let instances run until they free memory
        while waitq and all((len(i.rem) >= i.bcap or i.res + isl[waitq[0]] + osl[waitq[0]] > i.cap)
                            for i in insts):
            # advance the instance whose next completion is earliest
            cand = [i for i in insts if i.rem]
            nxt = min(cand, key=lambda i: i.t + min(i.rem) * step_time(i)[0])
            before = len(nxt.rem)
            advance(nxt, nxt.t + (min(nxt.rem) + 0.5) * step_time(nxt)[0])
            if len(nxt.rem) == before:   # numerical safety
                advance(nxt, math.inf)
            try_place(nxt.t)
    # drain
    while True:
        try_place(max(i.t for i in insts))
        active = [i for i in insts if i.rem]
        if not active and not waitq:
            break
        for inst in active:
            advance(inst, math.inf)
    energy = sum(i.energy for i in insts)
    busy = sum(i.busy for i in insts)
    return finish, energy, busy


# ---------------------------------------------------------------- aggregated
def simulate_aggregated(plan: Plan, n_inst: int, arr, isl, osl, P: Params, bcap: int,
                        chunk: int = 512, phi=0.25):
    """Chunked-prefill (Sarathi-style) colocated serving, JSQ dispatch by active count.
    Each iteration: all running decodes + up to `chunk` prompt tokens (FCFS)."""
    n = len(arr)
    cap = plan.kv_capacity_tokens()
    first = np.full(n, np.nan)
    finish = np.full(n, np.nan)

    class Inst:
        def __init__(s):
            s.t = 0.0; s.rem = []; s.ctx = []; s.rid = []; s.res = 0.0
            s.pq = []      # [rid, remaining_prompt_tokens, processed]
            s.energy = 0.0; s.busy = 0.0

    insts = [Inst() for _ in range(n_inst)]

    def one_macro(s, t_lim):
        """Advance by a group of identical iterations; return False if nothing to do."""
        has_pre = bool(s.pq)
        if not s.rem and not has_pre:
            return False
        B = len(s.rem)
        c = (sum(s.ctx) / B) if B else 0.0
        if has_pre:
            rid, remp, done_p = s.pq[0]
            C = min(chunk, remp)
            T, uc, um = plan.iteration(B, c, C, 2 * done_p + C, P)
            k_pre = math.ceil(remp / chunk)
            kd = min(s.rem) if B else math.inf
            k = min(k_pre, kd)
            k = 1 if k_pre == 1 else min(k, k_pre - 1)   # stop right before final chunk
            if k > 1:
                T, uc, um = plan.iteration(B, c + k / 2.0, C, 2 * (done_p + (k - 1) * chunk / 2.0) + C, P)
        else:
            T, uc, um = plan.iteration(B, c, 0, 0, P)
            k = min(s.rem)
        if t_lim < math.inf:
            k = min(k, int((t_lim - s.t) // T))
            if k <= 0:
                return False
        dt = k * T
        s.t += dt; s.busy += dt; s.energy += dt * plan.power(uc, um, phi)
        if B:
            s.rem = [x - k for x in s.rem]
            s.ctx = [x + k for x in s.ctx]
            keep = []
            for i, x in enumerate(s.rem):
                if x <= 0:
                    r = s.rid[i]; finish[r] = s.t; s.res -= isl[r] + osl[r]
                else:
                    keep.append(i)
            s.rem = [s.rem[i] for i in keep]; s.ctx = [s.ctx[i] for i in keep]
            s.rid = [s.rid[i] for i in keep]
        if has_pre:
            s.pq[0][1] -= min(remp, k * chunk)
            s.pq[0][2] += min(remp, k * chunk)
            if s.pq[0][1] <= 0:
                rid = s.pq.pop(0)[0]
                first[rid] = s.t
                if osl[rid] - 1 > 0:
                    s.rem.append(osl[rid] - 1); s.ctx.append(isl[rid] + 1); s.rid.append(rid)
                else:
                    finish[rid] = s.t; s.res -= isl[rid] + osl[rid]
        return True

    waitq = []

    def place(t_now):
        while waitq:
            r = waitq[0]; need = isl[r] + osl[r]
            best = None
            for s in insts:
                load = len(s.rem) + len(s.pq)
                if load < bcap and s.res + need <= cap:
                    if best is None or load < len(best.rem) + len(best.pq):
                        best = s
            if best is None:
                return
            waitq.pop(0)
            if not best.rem and not best.pq:
                best.t = max(best.t, t_now)
            best.pq.append([r, isl[r], 0]); best.res += need

    for r in range(n):
        t = arr[r]
        for s in insts:
            while one_macro(s, t):
                pass
            if not s.rem and not s.pq:
                s.t = max(s.t, t)
        waitq.append(r)
        place(t)
        while waitq and not any(len(s.rem) + len(s.pq) < bcap and
                                s.res + isl[waitq[0]] + osl[waitq[0]] <= cap for s in insts):
            cand = [s for s in insts if s.rem or s.pq]
            nxt = min(cand, key=lambda s: s.t)
            one_macro(nxt, math.inf)
            place(nxt.t)
    while True:
        place(max(s.t for s in insts))
        act = [s for s in insts if s.rem or s.pq]
        if not act and not waitq:
            break
        for s in act:
            one_macro(s, math.inf)
    energy = sum(s.energy for s in insts)
    busy = sum(s.busy for s in insts)
    return first, finish, energy, busy


# ---------------------------------------------------------------- drivers
def _metrics(arr, first, finish, osl, trim=0.05):
    n = len(arr)
    lo, hi = int(n * trim), int(n * (1 - trim))
    idx = np.arange(lo, hi)
    ttft = first[idx] - arr[idx]
    m = osl[idx] > 1
    tpot = (finish[idx][m] - first[idx][m]) / (osl[idx][m] - 1)
    return ttft, tpot


def run_disagg(pplan: Plan, n_p: int, dplan: Plan, n_d: int, bcap: int, wl, P: Params = DEFAULT,
               phi=0.25, nic=NIC_BW, tok_budget=16384):
    arr, isl, osl = wl
    first, ep, busy_p = simulate_prefill(pplan, n_p, arr, isl, P, tok_budget, phi)
    # KV transfer (decode-side format) over the NICs of the smaller instance
    kvb = dplan.model.kv_elems_per_token * kv_bytes(dplan.prec)
    xfer = isl * kvb / (nic * min(pplan.t, dplan.t))
    ready = first + xfer
    finish, ed, busy_d = simulate_decode(dplan, n_d, ready, isl, osl, P, bcap, phi)
    horizon = max(np.nanmax(finish), np.nanmax(first)) - arr[0]
    idle_p = max(n_p * horizon - float(np.sum(busy_p)), 0.0)
    idle_d = max(n_d * horizon - busy_d, 0.0)
    energy = (float(np.sum(ep)) + ed + idle_p * pplan.t * pplan.acc.tdp * phi
              + idle_d * dplan.t * dplan.acc.tdp * phi)
    ttft, tpot = _metrics(arr, first, finish, osl)
    ng = n_p * pplan.t + n_d * dplan.t
    kw = (n_p * pplan.t * pplan.acc.tdp + n_d * dplan.t * dplan.acc.tdp) / 1e3
    return Result(ttft, tpot, energy, int(osl.sum()), horizon, ng, kw,
                  dict(xfer_p50=float(np.median(xfer)), e_prefill=float(np.sum(ep)) + idle_p * pplan.t * pplan.acc.tdp * phi,
                       e_decode=ed + idle_d * dplan.t * dplan.acc.tdp * phi,
                       util_p=float(np.sum(busy_p)) / (n_p * horizon), util_d=busy_d / (n_d * horizon)))


def run_aggregated(plan: Plan, n: int, bcap: int, wl, P: Params = DEFAULT, phi=0.25, chunk=512):
    arr, isl, osl = wl
    first, finish, e_busy, busy = simulate_aggregated(plan, n, arr, isl, osl, P, bcap, chunk, phi)
    horizon = np.nanmax(finish) - arr[0]
    idle = max(n * horizon - busy, 0.0)
    energy = e_busy + idle * plan.t * plan.acc.tdp * phi
    ttft, tpot = _metrics(arr, first, finish, osl)
    return Result(ttft, tpot, energy, int(osl.sum()), horizon, n * plan.t,
                  n * plan.t * plan.acc.tdp / 1e3)
