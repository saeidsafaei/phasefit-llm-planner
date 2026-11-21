"""Workloads: real Azure LLM inference traces (2023) and a reasoning workload.

Azure traces (Patel et al., Splitwise, ISCA'24; Azure Public Dataset):
  conv : 19,366 requests in ~58 min (conversation service)
  code : 8,819 requests in ~57 min (coding assistant)
`rate_x` compresses inter-arrival times (load multiplier); lengths are unchanged.
Reasoning: arrival process of the conv trace, lengths drawn from log-normal laws
whose means match the MLPerf v5.1 DeepSeek-R1 dataset (ISL 800, OSL 3,880;
OSL capped at 20,000). The dispersion parameters are our assumption.
"""
import numpy as np
import pandas as pd

RAW = "/home/claude/paper/data/raw/AzureLLMInferenceTrace_{}.csv"


def _load(kind):
    d = pd.read_csv(RAW.format(kind))
    t = pd.to_datetime(d["TIMESTAMP"]).astype("int64").to_numpy() / 1e9
    t = t - t.min()
    o = np.argsort(t, kind="stable")
    return t[o], d["ContextTokens"].to_numpy()[o].astype(float), \
        np.maximum(d["GeneratedTokens"].to_numpy()[o], 1).astype(float)


def azure(kind, rate_x=1.0, window=None):
    t, i, o = _load(kind)
    t = t / rate_x
    if window is not None:
        m = t <= window
        t, i, o = t[m], i[m], o[m]
    return t, i, o


def _lognorm(rng, mean, sigma, n, lo, hi):
    mu = np.log(mean) - sigma ** 2 / 2
    x = np.clip(rng.lognormal(mu, sigma, n), lo, hi)
    return np.round(x * mean / x.mean()).clip(lo, hi)


def reasoning(rate_x=1.0, window=None, seed=7, isl_mean=800, osl_mean=3880,
              isl_sigma=0.8, osl_sigma=0.9):
    t, _, _ = _load("conv")
    rng = np.random.default_rng(seed)
    n = len(t)
    i = _lognorm(rng, isl_mean, isl_sigma, n, 16, 8192)
    o = _lognorm(rng, osl_mean, osl_sigma, n, 64, 20000)
    t = t / rate_x
    if window is not None:
        m = t <= window
        t, i, o = t[m], i[m], o[m]
    return t, i, o


def describe(wl):
    t, i, o = wl
    dur = t.max() - t.min()
    return dict(n=len(t), duration_s=dur, rate=len(t) / dur,
                isl_mean=i.mean(), isl_p50=np.median(i), isl_p99=np.percentile(i, 99),
                osl_mean=o.mean(), osl_p50=np.median(o), osl_p99=np.percentile(o, 99))


if __name__ == "__main__":
    for name, wl in [("conv", azure("conv")), ("code", azure("code")), ("reason", reasoning())]:
        print(name, {k: round(float(v), 1) for k, v in describe(wl).items()})
