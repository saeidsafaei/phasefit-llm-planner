"""Phase-aware analytical latency / power model for LLM inference iterations.

An *instance* is one model replica spread over `t` accelerators of one type,
using tensor parallelism ('tp', dense models) or attention data parallelism +
expert parallelism ('dep', MoE models). One *iteration* (forward pass) may
contain decode tokens and/or prefill tokens. Its duration is

    T = max(T_comp, T_mem) + T_comm + L * tau_l                      (Eq. 6)

with T_comp = F_lin/(eta_c P_prec) + F_attn/(eta_c P_bf16) and
T_mem = bytes/(eta_m BW). Efficiencies eta_c, eta_m and the per-layer
overhead tau_l are fitted once (calibrate.py) and then held fixed.
"""
from dataclasses import dataclass
import math

from hw import ACCELS, Accel, weight_bytes, kv_bytes
from models import MODELS, Model

ALPHA = 8e-6          # latency of one collective (s), fixed
MEM_FRAC = 0.90       # fraction of HBM usable by weights + KV cache


@dataclass
class Params:
    eta_c: float = 0.60     # peak GEMM efficiency, BF16/FP8
    eta_c4: float = 0.45    # peak GEMM efficiency, FP4
    m_half: float = 64.0    # GEMM rows at which efficiency reaches half of its peak
    eta_m: float = 0.80     # achievable fraction of HBM bandwidth
    tau_l: float = 20e-6    # per-layer fixed overhead (s): kernel launch, sync, scheduling
    kappa: float = 1.3      # expert-parallel load-imbalance factor (MoE expert time multiplier)


DEFAULT = Params()


@dataclass(frozen=True, eq=False)
class Plan:
    acc: Accel
    model: Model
    prec: str      # 'bf16' | 'fp8' | 'fp4'
    t: int         # accelerators per instance
    par: str       # 'tp' | 'dep'

    @staticmethod
    def make(acc: str, model: str, prec: str, t: int, par: str | None = None):
        m = MODELS[model]
        if par is None:
            par = "dep" if m.moe else "tp"
        return Plan(ACCELS[acc], m, prec, t, par)

    def label(self):
        return f"{self.acc.name}/{self.prec.upper()}/{self.par.upper()}{self.t}"

    # ---------------- memory ----------------
    def wb_main(self) -> float:
        return weight_bytes(self.acc, self.prec)

    def wb_rep(self) -> float:
        """Non-expert weights of MoE models stay at FP8 when experts are FP4."""
        if self.model.moe and self.prec == "fp4":
            return weight_bytes(self.acc, "fp8")
        return self.wb_main()

    def weights_per_gpu(self) -> float:
        m = self.model
        if self.par == "tp":
            return m.params_total * self.wb_main() / self.t
        return m.replicated_params() * self.wb_rep() + m.routed_params() * self.wb_main() / self.t

    def kv_shard(self) -> int:
        m = self.model
        if self.par == "dep" or m.mla:
            return 1
        return min(self.t, m.n_kv)

    def kv_token_bytes(self) -> float:
        """KV bytes for one token of one sequence, on the GPU(s) that hold it."""
        return self.model.kv_elems_per_token * kv_bytes(self.prec)

    def kv_capacity_tokens(self) -> float:
        """Total tokens (summed over sequences) the instance can keep in KV cache."""
        free = MEM_FRAC * self.acc.mem - self.weights_per_gpu()
        if free <= 0:
            return 0.0
        per_gpu_tok = self.kv_token_bytes() / self.kv_shard()
        if self.par == "tp" and not self.model.mla:
            return free / per_gpu_tok            # each GPU holds a 1/shard slice of every token
        if self.par == "tp":                     # MLA latent cache is replicated across TP ranks
            return free / per_gpu_tok
        return self.t * free / per_gpu_tok       # DEP: sequences are spread across GPUs

    def fits(self) -> bool:
        return self.acc.supports(self.prec) and self.kv_capacity_tokens() > 0

    # ---------------- one iteration ----------------
    def iteration(self, n_dec: float, ctx_dec: float, n_pre: float = 0.0,
                  pre_len: float = 0.0, P: Params = DEFAULT):
        """Return (seconds, util_compute, util_memory) for one forward pass.

        n_dec decode tokens whose mean context is ctx_dec; n_pre prefill tokens
        belonging to prompts of mean length pre_len.
        """
        m, a, t = self.model, self.acc, self.t
        tok = n_dec + n_pre
        # FLOPs per GPU
        f_lin = m.lin_flops_per_token * tok / t
        f_att = n_dec * m.attn_flops_decode(ctx_dec)
        if n_pre > 0 and pre_len > 0:
            f_att += (n_pre / pre_len) * m.attn_flops_prefill(pre_len)
        f_att /= t
        # GEMM efficiency rises with the number of rows M (tokens) of each GEMM:
        # eta(M) = eta_peak * M / (M + m_half)                              (Eq. 7)
        def eff(prec, M):
            peak = P.eta_c4 if prec == "fp4" else P.eta_c
            return peak * M / (M + P.m_half) if M > 0 else peak
        if m.moe:
            f_exp = 2.0 * m.n_moe_layers * m.top_k * m.expert_params() * tok / t
            f_rep = f_lin - f_exp
            rows_rep = tok / t if self.par == "dep" else tok
            rows_exp = tok * m.top_k / m.n_exp
            rep_prec = "fp8" if self.prec == "fp4" else self.prec
            t_lin = (f_rep / (eff(rep_prec, rows_rep) * a.peak[rep_prec])
                     + P.kappa * f_exp / (eff(self.prec, rows_exp) * a.peak[self.prec]))
        else:
            t_lin = f_lin / (eff(self.prec, tok) * a.peak[self.prec])
        t_comp = t_lin + f_att / (P.eta_c * a.peak["bf16"])
        # bytes per GPU
        if m.moe:
            frac = 1.0 - math.exp(-tok * m.top_k / m.n_exp) if tok > 0 else 0.0
            if self.par == "dep":
                w = (m.replicated_params() * self.wb_rep()
                     + m.routed_params() * self.wb_main() / t * frac)
            else:
                w = (m.replicated_params() * self.wb_rep()
                     + m.routed_params() * self.wb_main() * frac) / t
        else:
            w = self.weights_per_gpu()
        kv_read = n_dec * ctx_dec * self.kv_token_bytes() / self.kv_shard()
        kv_write = tok * self.kv_token_bytes() / self.kv_shard()
        if self.par == "dep":
            kv_read /= t
            kv_write /= t
        t_mem = (w + kv_read + kv_write) / (P.eta_m * a.bw)
        # communication
        t_comm = 0.0
        if t > 1:
            if self.par == "tp":
                msg = tok * m.d * 2.0
                t_comm = m.L * 2 * (ALPHA + 2.0 * (t - 1) / t * msg / a.link)
            else:
                per_gpu = tok / t * m.top_k * m.d * 3.0 * (t - 1) / t   # FP8 dispatch + BF16 combine
                t_comm = m.n_moe_layers * (2 * ALPHA + per_gpu / a.link)
        busy = max(t_comp, t_mem)
        T = busy + t_comm + m.L * P.tau_l
        return T, t_comp / T, t_mem / T

    # ---------------- convenience ----------------
    def decode_step(self, batch: float, ctx: float, P: Params = DEFAULT):
        return self.iteration(batch, ctx, 0.0, 0.0, P)

    def prefill_time(self, n_tokens: float, prompt_len: float, P: Params = DEFAULT):
        return self.iteration(0.0, 0.0, n_tokens, prompt_len, P)

    def power(self, uc: float, um: float, phi_idle: float = 0.25) -> float:
        """Instance power (W): P = t * TDP * (phi + (1 - phi) * max(uc, um))."""
        u = min(1.0, max(uc, um))
        return self.t * self.acc.tdp * (phi_idle + (1.0 - phi_idle) * u)


def max_throughput_aggregated(plan: Plan, isl: float, osl: float, P: Params = DEFAULT,
                              max_batch_rank: int = 2048, max_tokens_rank: int = 8192,
                              osl_reserve: float | None = None):
    """Steady-state output tokens/s per GPU of an aggregated (in-flight batching)
    instance driven at maximum load with fixed ISL/OSL (Eq. 9).

    Each iteration carries B decode tokens and, on average, B*ISL/OSL prefill
    tokens (one new request enters whenever one finishes). B is limited by the
    KV capacity (sequences reserve ISL+OSL tokens, i.e. no eviction), by the
    per-rank batch limit and by the per-rank token budget.
    """
    ranks = plan.t if plan.par == "dep" else 1
    cap = plan.kv_capacity_tokens()
    if cap <= 0:
        return 0.0, 0
    b_mem = cap / (isl + (osl_reserve or osl))
    b_max = min(b_mem, max_batch_rank * ranks, max_tokens_rank * ranks / (1.0 + isl / osl))
    best, best_b = 0.0, 0
    b = 1.0
    while b <= b_max + 1e-9:
        T, _, _ = plan.iteration(b, isl + osl / 2.0, b * isl / osl, isl, P)
        thr = b / T / plan.t
        if thr > best:
            best, best_b = thr, int(b)
        b = b * 1.05 + 1
    T, _, _ = plan.iteration(b_max, isl + osl / 2.0, b_max * isl / osl, isl, P)
    if b_max / T / plan.t > best:
        best, best_b = b_max / T / plan.t, int(b_max)
    return best, best_b


def request_latency(plan: Plan, batch: int, isl: int, osl: int, P: Params = DEFAULT,
                    stride: int = 16):
    """End-to-end latency of `batch` identical requests processed together
    (prefill of all prompts, then osl decode steps) - vLLM `benchmark_latency`."""
    T_pre, _, _ = plan.prefill_time(batch * isl, isl, P)
    total = T_pre
    for i in range(0, osl - 1, stride):
        n = min(stride, osl - 1 - i)
        T, _, _ = plan.decode_step(batch, isl + i + n / 2.0, P)
        total += n * T
    return total
