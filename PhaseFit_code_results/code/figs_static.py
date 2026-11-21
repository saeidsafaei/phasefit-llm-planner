"""Figures that do not depend on the planner runs: workload characterisation,
model validation, and per-phase efficiency of each accelerator/precision."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from params import FITTED as P
from planner import candidate_plans, decode_batch_cap, SLO
from workloads import azure, reasoning

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150})
OUT = "../figs/"


def fig_workloads():
    wls = {"Azure-Conv": azure("conv"), "Azure-Code": azure("code"), "Reasoning (R1-like)": reasoning()}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for k, (name, (t, i, o)) in enumerate(wls.items()):
        for arr, a in ((i, ax[0]), (o, ax[1])):
            x = np.sort(arr)
            a.plot(x, np.arange(1, len(x) + 1) / len(x), color=C[k], lw=1.6, label=name)
        if name != "Reasoning (R1-like)":
            per_min = np.bincount((t // 60).astype(int))
            ax[2].plot(np.arange(len(per_min)), per_min / 60.0, color=C[k], lw=1.6, label=name)
    ax[0].set_xscale("log"); ax[1].set_xscale("log")
    ax[0].set_xlabel("Input (prompt) tokens"); ax[1].set_xlabel("Output tokens")
    ax[0].set_ylabel("CDF"); ax[1].set_ylabel("CDF")
    ax[2].set_xlabel("Minute of trace"); ax[2].set_ylabel("Requests / s (1-min mean)")
    for a, l in zip(ax, "abc"):
        a.set_title(f"({l})", loc="left", fontsize=9, color=INK)
    ax[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT + "fig_workloads.png"); fig.savefig(OUT + "fig_workloads.pdf")


def fig_validation():
    d = json.load(open("../results/calibration.json"))
    rows = d["rows"]
    groups = [("TRT", "Llama-3.3-70B", "TRT-LLM Llama-3.3-70B (fit)", "o", C[0]),
              ("TRT", "DeepSeek-R1", "TRT-LLM DeepSeek-R1 (fit)", "s", C[1]),
              ("TRT", "Qwen3-235B-A22B", "TRT-LLM Qwen3-235B (held out)", "^", C[2]),
              ("MLP", "DeepSeek-R1", "MLPerf v5.1 DeepSeek-R1 (held out)", "D", C[6]),
              ("AMD", "Llama-3.1-70B", "vLLM/ROCm MI300X latency (held out)", "v", C[7])]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for src, model, lab, mk, col in groups:
        g = [r for r in rows if r["src"] == src and r["model"] == model]
        a = ax[1] if src == "AMD" else ax[0]
        a.scatter([r["meas"] for r in g], [r["pred"] for r in g], marker=mk, s=28, color=col,
                  edgecolor="white", linewidth=0.8, label=lab, zorder=3)
    for k, (a, (lo, hi)) in enumerate(zip(ax, ((80, 2e4), (10, 70)))):
        xs = np.array([lo, hi])
        a.plot(xs, xs, color=INK2, lw=1, zorder=1)
        a.fill_between(xs, xs * 0.8, xs * 1.2, color=GRID, alpha=0.6, zorder=0, label="±20 %")
        if k == 0:
            a.set_xscale("log"); a.set_yscale("log")
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    ax[0].set_xlabel("Measured output tokens/s per GPU"); ax[0].set_ylabel("Predicted")
    ax[1].set_xlabel("Measured batch latency (s)"); ax[1].set_ylabel("Predicted")
    ax[0].legend(fontsize=6.5, loc="upper left")
    ax[1].legend(fontsize=6.5, loc="upper left")
    ax[0].set_title("(a) Max-load throughput", loc="left", fontsize=9)
    ax[1].set_title("(b) Batch latency, 8×MI300X", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT + "fig_validation.png"); fig.savefig(OUT + "fig_validation.pdf")


def phase_efficiency(model="Llama-3.1-70B"):
    wl = azure("conv")
    slo = SLO(2.0, 0.1)
    best = {}
    for p in candidate_plans(model):
        T, uc, um = p.prefill_time(16384, 1500, P)
        pre = p.power(uc, um, 0.25) * T / 16384 * 1000
        b = decode_batch_cap(p, wl, slo, P)
        if b < 1:
            continue
        T2, uc2, um2 = p.decode_step(b, 1155 + 211 * 0.75, P)
        dec = p.power(uc2, um2, 0.25) * T2 / b
        k = (p.acc.name, p.prec)
        cur = best.get(k, {})
        if pre < cur.get("pre", 1e9):
            cur["pre"], cur["pre_t"] = pre, p.t
        if dec < cur.get("dec", 1e9):
            cur["dec"], cur["dec_t"], cur["dec_b"] = dec, p.t, b
        best[k] = cur
    return best


def fig_phase_eff():
    best = phase_efficiency()
    accs = ["H100", "H200", "B200", "MI300X", "MI325X", "MI355X", "Gaudi3"]
    precs = ["bf16", "fp8", "fp4"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    w = 0.26
    hatch = {"bf16": "", "fp8": "", "fp4": "////"}
    for j, prec in enumerate(precs):
        for a, key, scale in ((ax[0], "pre", 1.0), (ax[1], "dec", 1000.0)):
            xs, ys = [], []
            for i, acc in enumerate(accs):
                v = best.get((acc, prec), {}).get(key)
                if v is not None:
                    xs.append(i + (j - 1) * w); ys.append(v * scale)
            a.bar(xs, ys, width=w - 0.03, color=C[j], hatch=hatch[prec], edgecolor="white",
                  linewidth=0.5, label=prec.upper(), zorder=3)
    for a in ax:
        a.set_xticks(range(len(accs))); a.set_xticklabels(accs, rotation=30, ha="right", fontsize=7.5)
        a.grid(axis="x", visible=False)
        a.set_ylim(0, a.get_ylim()[1] * 1.18)
    ax[0].set_ylabel("J per 1k prompt tokens")
    ax[1].set_ylabel("mJ per output token")
    ax[0].set_title("(a) Prefill, 16k-token batches", loc="left", fontsize=9)
    ax[1].set_title("(b) Decode, batch with step ≤ 85 ms", loc="left", fontsize=9)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT + "fig_phase_eff.png"); fig.savefig(OUT + "fig_phase_eff.pdf")
    json.dump({f"{k[0]}/{k[1]}": v for k, v in best.items()}, open("../results/phase_eff.json", "w"), indent=1)


if __name__ == "__main__":
    fig_workloads()
    fig_validation()
    fig_phase_eff()
    print("ok")
