"""Turn the profiles into the paper's tables, figures and numbers (results/summary.json)."""
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from params import FITTED as P
from planner import select, POLICIES, energy_at, fix_static
from sim import run_disagg
from experiments import SCEN, workload
from hw import NIC_BW

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300})
RES = "../results/"
FIG = "../figs/"
POLS = ["Agg-Homo", "Dis-Homo", "Dis-HW", "Dis-Prec", "PhaseFit-V", "PhaseFit-X"]


def load(S):
    fn = f"{RES}profile_{S}.pkl"
    return fix_static(pickle.load(open(fn, "rb"))) if os.path.exists(fn) else None


def desc(sel, pol):
    if sel is None:
        return None
    if pol == "Agg-Homo":
        a = sel["agg"]
        return dict(kind="agg", cfg=a["label"], n=a["n"], chunk=a["chunk"], bcap=a["bcap"],
                    summary=a["summary"])
    return dict(kind="dis", pre=sel["pre"]["label"], n_pre=sel["pre"]["n"],
                dec=sel["dec"]["label"], n_dec=sel["dec"]["n"], bcap=sel["dec"]["bcap"],
                e_pre=energy_at(sel["pre"], 0.25), e_dec=energy_at(sel["dec"], 0.25))


def main_results(scens, phi=0.25):
    out = {}
    for S in scens:
        prof = load(S)
        if prof is None:
            continue
        m = prof["meta"]
        nreq, tok = m["n"], m["out_tokens"]
        res = {}
        for fp4 in (True, False):
            for pol in POLS:
                sel = select(prof, pol, phi=phi, allow_fp4=fp4)
                key = f"{pol}|{'fp4' if fp4 else 'nofp4'}"
                if sel is None:
                    res[key] = None
                    continue
                res[key] = dict(J_req=sel["energy"] / nreq, mJ_tok=sel["energy"] / tok * 1e3,
                                gpus=sel["gpus"], kw=sel["kw"], **desc(sel, pol))
        out[S] = dict(meta=m, res=res)
    return out


def validate_choices(summary, scens):
    """End-to-end re-simulation of the selected disaggregated deployments."""
    for S in scens:
        if S not in summary:
            continue
        prof = load(S)
        model, wname, rx, slo = SCEN[S]
        wl = workload(wname, rx)
        for key in ("PhaseFit-X|fp4", "Dis-Homo|fp4", "Dis-HW|fp4", "PhaseFit-X|nofp4"):
            pol, fp = key.split("|")
            sel = select(prof, pol, phi=0.25, allow_fp4=(fp == "fp4"))
            if sel is None:
                continue
            r = run_disagg(sel["pre"]["plan"], sel["pre"]["n"], sel["dec"]["plan"], sel["dec"]["n"],
                           sel["dec"]["bcap"], wl, P, 0.0)
            s = r.summary()
            H = wl[0][-1] - wl[0][0]
            r.energy_j = 0.25 * r.prov_kw * 1e3 * H + 0.75 * r.energy_j
            summary[S]["res"][key]["validated"] = dict(
                ttft_p99=s["ttft_p99"], tpot_p99=s["tpot_p99"], ttft_p50=s["ttft_p50"],
                tpot_p50=s["tpot_p50"], J_req=r.energy_j / len(wl[0]), xfer_p50=r.detail["xfer_p50"],
                util_p=r.detail["util_p"], util_d=r.detail["util_d"],
                ok=bool(s["ttft_p99"] <= slo.ttft and s["tpot_p99"] <= slo.tpot))
            print(S, key, {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in summary[S]["res"][key]["validated"].items()}, flush=True)


def phi_sensitivity(scens):
    out = {}
    for S in scens:
        prof = load(S)
        if prof is None:
            continue
        nreq = prof["meta"]["n"]
        out[S] = {}
        for phi in (0.15, 0.25, 0.40):
            a = select(prof, "Agg-Homo", phi=phi)
            dh = select(prof, "Dis-Homo", phi=phi)
            px = select(prof, "PhaseFit-X", phi=phi)
            out[S][phi] = dict(agg=a and a["energy"] / nreq, dis_homo=dh and dh["energy"] / nreq,
                               phasefit=px and px["energy"] / nreq,
                               choice=px and f"{px['pre']['label']}+{px['dec']['label']}")
    return out


def nic_sensitivity(scens):
    out = {}
    for S in scens:
        prof = load(S)
        if prof is None:
            continue
        model, wname, rx, slo = SCEN[S]
        wl = workload(wname, rx)
        sel = select(prof, "PhaseFit-X", phi=0.25)
        out[S] = {}
        for nic in (6.25e9, 12.5e9, 25e9, 50e9, 100e9):
            r = run_disagg(sel["pre"]["plan"], sel["pre"]["n"], sel["dec"]["plan"], sel["dec"]["n"],
                           sel["dec"]["bcap"], wl, P, 0.25, nic)
            s = r.summary()
            out[S][nic / 1e9] = dict(tpot_p99=s["tpot_p99"], xfer_p50=r.detail["xfer_p50"],
                                     ok=bool(s["tpot_p99"] <= slo.tpot))
    return out


def price_analysis(S, new="B200", old="H200"):
    """Rental-cost view: GPU-hours priced H200 = 1, new = r. Best config per phase/type."""
    prof = load(S)
    if prof is None:
        return None

    def best(phase, acc):
        c = [x for x in prof[phase] if x["acc"] == acc]
        return min(c, key=lambda x: x["gpus"]) if c else None

    bp_new, bp_old = best("pre", new), best("pre", old)
    bd_new, bd_old = best("dec", new), best("dec", old)
    rs = np.linspace(1.0, 5.0, 401)
    strat = {
        f"all {old}": bp_old["gpus"] + bd_old["gpus"] + 0 * rs,
        f"all {new}": (bp_new["gpus"] + bd_new["gpus"]) * rs,
        f"{new} prefill + {old} decode": bp_new["gpus"] * rs + bd_old["gpus"],
        f"{old} prefill + {new} decode": bp_old["gpus"] + bd_new["gpus"] * rs,
    }
    winner = [min(strat, key=lambda k: strat[k][i]) for i in range(len(rs))]
    bands = []
    for i, w in enumerate(winner):
        if not bands or bands[-1][0] != w:
            bands.append([w, float(rs[i]), float(rs[i])])
        else:
            bands[-1][2] = float(rs[i])
    return dict(scenario=S, gpus=dict(pre_new=bp_new["gpus"], pre_old=bp_old["gpus"],
                                       dec_new=bd_new["gpus"], dec_old=bd_old["gpus"],
                                       pre_new_cfg=bp_new["label"], pre_old_cfg=bp_old["label"],
                                       dec_new_cfg=bd_new["label"], dec_old_cfg=bd_old["label"]),
                bands=bands, rs=rs.tolist(), strat={k: v.tolist() for k, v in strat.items()})


# ------------------------------------------------------------------ figures
def fig_main(summary, scens):
    labels = {"A": "A: 70B Conv\nStd", "B": "B: 70B Conv\nInt", "C": "C: 70B Code\nStd",
              "D": "D: 70B Code\nInt", "E": "E: R1 Reason", "F": "F: Qwen3 Conv\nStd"}
    pols = [("Agg-Homo", "Agg-Homo", C[0], ""), ("Dis-HW", "Dis-HW (FP8)", C[1], ""),
            ("Dis-Homo", "Dis-Homo", C[2], ""), ("PhaseFit-X", "PhaseFit-X", C[6], "")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, fp, title in ((axs[0], "fp4", "(a) FP4 allowed"), (axs[1], "nofp4", "(b) FP4 not allowed")):
        S_ok = [S for S in scens if S in summary]
        w = 0.2
        for j, (pol, lab, col, h) in enumerate(pols):
            xs, ys = [], []
            for i, S in enumerate(S_ok):
                base = summary[S]["res"].get(f"Agg-Homo|fp4")
                v = summary[S]["res"].get(f"{pol}|{fp}")
                if base and v:
                    xs.append(i + (j - 1.5) * w); ys.append(v["J_req"] / base["J_req"])
            ax.bar(xs, ys, width=w - 0.02, color=col, label=lab, zorder=3, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(S_ok)))
        ax.set_xticklabels(S_ok, fontsize=8.5)
        ax.set_xlabel("Scenario")
        ax.axhline(1.0, color=INK2, lw=0.8, zorder=2)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Energy per request\n(norm. to Agg-Homo, FP4)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG + "fig_main.png"); fig.savefig(FIG + "fig_main.pdf")


def fig_split(summary, scens):
    S_ok = [S for S in scens if S in summary and summary[S]["res"].get("PhaseFit-X|fp4")]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    pre = [summary[S]["res"]["PhaseFit-X|fp4"]["e_pre"] for S in S_ok]
    dec = [summary[S]["res"]["PhaseFit-X|fp4"]["e_dec"] for S in S_ok]
    tot = np.array(pre) + np.array(dec)
    ax.bar(range(len(S_ok)), np.array(pre) / tot * 100, color=C[1], label="prefill", zorder=3, width=0.6)
    ax.bar(range(len(S_ok)), np.array(dec) / tot * 100, bottom=np.array(pre) / tot * 100, color=C[0],
           label="decode", zorder=3, width=0.6, edgecolor="white", linewidth=1.0)
    ax.set_xticks(range(len(S_ok))); ax.set_xticklabels(S_ok)
    ax.set_ylabel("Share of energy (%)"); ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(FIG + "fig_split.png"); fig.savefig(FIG + "fig_split.pdf")


def fig_price(pa_list):
    pa_list = [p for p in pa_list if p]
    fig, axs = plt.subplots(1, len(pa_list), figsize=(7.2, 2.6), squeeze=False)
    for ax, pa in zip(axs[0], pa_list):
        rs = np.array(pa["rs"])
        for k, (name, ys) in enumerate(pa["strat"].items()):
            ax.plot(rs, ys, color=C[k], lw=1.8, label=name)
        ax.set_xlabel("Price ratio B200 : H200 (per GPU-hour)")
        ax.set_ylabel("Rental cost (H200 GPU-h per h)")
        ax.set_title(f"({'ab'[pa_list.index(pa)]}) Scenario {pa['scenario']}", loc="left", fontsize=9)
        for w, lo, hi in pa["bands"]:
            if "prefill +" in w and hi > lo:
                ax.axvspan(lo, hi, color=GRID, alpha=0.8, zorder=0)
    axs[0][0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG + "fig_price.png"); fig.savefig(FIG + "fig_price.pdf")


def fig_brownfield():
    fn = RES + "brownfield.json"
    if not os.path.exists(fn):
        return None
    b = json.load(open(fn))
    tok = b["meta"]["out_tokens"]
    base = b["old_only"]["energy"] / tok * 1e3
    Ns = sorted({r["N"] for r in b["runs"]})
    def pick(N, cond):
        c = [r for r in b["runs"] if r["N"] == N and cond(r)]
        return min(c, key=lambda r: r["energy"]) if c else None
    series = {
        "New→prefill, FP8 (Splitwise-style)": lambda r: r["alpha"] == 1.0 and r["prec_pre"] == "fp8",
        "New→prefill, best precision": lambda r: r["alpha"] == 1.0,
        "New→decode, best precision": lambda r: r["alpha"] == 0.0,
        "PhaseFit (β, precisions)": lambda r: True,
    }
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    ax = axs[0]
    ax.axhline(base, color=INK2, lw=1, ls="--", label="H200 only (no B200)")
    agg = b.get("agg", {})
    if agg:
        xs = [int(k) for k in agg if int(k) > 0 and agg[k]]
        ys = [agg[str(k)]["energy"] / tok * 1e3 for k in xs]
        ax.plot(xs, ys, marker="v", color=C[7], lw=1.6, ms=6, label="Aggregated, mixed fleet")
    mk = ["s", "D", "^", "o"]
    for k, (name, cond) in enumerate(series.items()):
        ys = [pick(N, cond)["energy"] / tok * 1e3 for N in Ns]
        ax.plot(Ns, ys, marker=mk[k], color=[C[1], C[3], C[2], C[6]][k], lw=1.6, ms=6, label=name)
    ax.set_xlabel("New B200 GPUs added"); ax.set_ylabel("mJ per output token")
    ax.set_xticks(Ns)
    ax.set_title("(a) Energy vs. B200 budget", loc="left", fontsize=9)
    h, l = ax.get_legend_handles_labels()
    ax = axs[1]
    for k, N in enumerate(Ns):
        betas = sorted({r["alpha"] for r in b["runs"] if r["N"] == N})
        ys = [pick(N, lambda r, a=a: r["alpha"] == a)["energy"] / tok * 1e3 for a in betas]
        ax.plot(betas, ys, marker="o", color=C[k], lw=1.6, ms=5, label=f"{N} B200")
    ax.axhline(base, color=INK2, lw=1, ls="--")
    ax.set_xlabel("β = share of B200 GPUs given to prefill"); ax.set_ylabel("mJ per output token")
    ax.set_title("(b) Placement of the new GPUs", loc="left", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(FIG + "fig_brownfield.png"); fig.savefig(FIG + "fig_brownfield.pdf")
    out = dict(base=base, N={})
    for N in Ns:
        out["N"][N] = {name: dict(mJ=pick(N, cond)["energy"] / tok * 1e3,
                                  alpha=pick(N, cond)["alpha"], pp=pick(N, cond)["prec_pre"],
                                  pd=pick(N, cond)["prec_dec"],
                                  n_new_pre=pick(N, cond)["pre"]["n_new"], f_pre=pick(N, cond)["pre"]["f_new"],
                                  n_new_dec=pick(N, cond)["dec"]["n_new"], f_dec=pick(N, cond)["dec"]["f_new"],
                                  gpus_old=pick(N, cond)["gpus_old"], kw=pick(N, cond)["kw"])
                       for name, cond in series.items()}
        if agg and str(N) in agg and agg[str(N)]:
            out["N"][N]["agg"] = dict(mJ=agg[str(N)]["energy"] / tok * 1e3, f=agg[str(N)]["f_new"],
                                      n_old=agg[str(N)]["n_old"])
    if agg and "0" in agg and agg["0"]:
        out["agg0"] = agg["0"]["energy"] / tok * 1e3
    return out


if __name__ == "__main__" and len(sys.argv) == 1:
    scens = [s for s in "ABCDEF"]
    summ = main_results(scens)
    validate_choices(summ, scens)
    sens = dict(phi=phi_sensitivity(scens), nic=nic_sensitivity(["A", "C", "E"]))
    price = [price_analysis("A"), price_analysis("C")]
    fig_main(summ, scens); fig_split(summ, scens); fig_price(price)
    bf = fig_brownfield()
    json.dump(dict(main=summ, sens=sens, price=[p and {k: v for k, v in p.items() if k not in ("rs", "strat")}
                                                 for p in price], brownfield=bf),
              open(RES + "summary.json", "w"), indent=1, default=str)
    print("written")


def load_scale():
    out = {}
    for S in ("A10", "A20", "A"):
        prof = load(S)
        if prof is None:
            continue
        n = prof["meta"]["n"]
        row = {}
        for fp in (True, False):
            for pol in ("Agg-Homo", "Dis-Homo", "PhaseFit-X"):
                b = select(prof, pol, phi=0.25, allow_fp4=fp)
                if b is None:
                    continue
                d = (b["agg"]["label"] + f" x{b['agg']['n']}") if pol == "Agg-Homo" else \
                    f"{b['pre']['label']} x{b['pre']['n']} + {b['dec']['label']} x{b['dec']['n']}"
                row[f"{pol}|{'fp4' if fp else 'nofp4'}"] = dict(J_req=b["energy"] / n, gpus=b["gpus"], cfg=d)
        out[S] = dict(rate_x=prof["meta"]["rate_x"], res=row)
    return out


def perturbation():
    base = load("A")
    out = {}
    def sel_desc(prof, fp):
        b = select(prof, "PhaseFit-X", phi=0.25, allow_fp4=fp)
        return b and dict(J_req=b["energy"] / prof["meta"]["n"], gpus=b["gpus"],
                          cfg=f"{b['pre']['label']} x{b['pre']['n']} + {b['dec']['label']} x{b['dec']['n']}")
    out["base"] = {fp: sel_desc(base, fp) for fp in (True, False)}
    for tag in ("etac-20", "etam-12", "tau+100", "kappa+50"):
        fn = f"{RES}profile_A_{tag}.pkl"
        if not os.path.exists(fn):
            continue
        prof = fix_static(pickle.load(open(fn, "rb")))
        out[tag] = {fp: sel_desc(prof, fp) for fp in (True, False)}
        # regret: energy of the base decision evaluated under the perturbed model
        bb = select(base, "PhaseFit-X", phi=0.25)
        same = [(r, q) for r in prof["pre"] for q in prof["dec"]
                if r["label"] == bb["pre"]["label"] and q["label"] == bb["dec"]["label"]]
        if same:
            r, q = same[0]
            e = (energy_at(r, 0.25) + energy_at(q, 0.25)) / prof["meta"]["n"]
            out[tag]["regret_pct"] = (e / out[tag][True]["J_req"] - 1) * 100
    return out


def brownfield_e():
    fn = RES + "brownfield_E.json"
    if not os.path.exists(fn):
        return None
    b = json.load(open(fn))
    tok = b["meta"]["out_tokens"]
    n = b["meta"]["n"]
    base = b["old_only"]["energy"]
    Ns = sorted({r["N"] for r in b["runs"]})
    def pick(N, cond):
        c = [r for r in b["runs"] if r["N"] == N and cond(r)]
        return min(c, key=lambda r: r["energy"]) if c else None
    series = {"New→prefill, FP8": lambda r: r["alpha"] == 1.0 and r["prec_pre"].startswith("fp8"),
              "New→prefill, best": lambda r: r["alpha"] == 1.0,
              "New→decode, best": lambda r: r["alpha"] == 0.0,
              "PhaseFit": lambda r: True}
    out = dict(base_J_req=base / n, base_mJ=base / tok * 1e3, N={})
    for N in Ns:
        out["N"][N] = {}
        for k, c in series.items():
            r = pick(N, c)
            out["N"][N][k] = dict(J_req=r["energy"] / n, mJ=r["energy"] / tok * 1e3, beta=r["alpha"],
                                  pp=r["prec_pre"], pd=r["prec_dec"], n_new_pre=r["pre"]["n_new"],
                                  f_pre=r["pre"]["f_new"], n_new_dec=r["dec"]["n_new"], f_dec=r["dec"]["f_new"],
                                  saving=(1 - r["energy"] / base) * 100)
    # figure
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.axhline(base / n, color=INK2, lw=1, ls="--", label="H200 only")
    mk = ["s", "D", "^", "o"]
    cols = [C[1], C[3], C[2], C[6]]
    for k, (name, c) in enumerate(series.items()):
        ax.plot(Ns, [pick(N, c)["energy"] / n for N in Ns], marker=mk[k], color=cols[k], lw=1.6, ms=6, label=name)
    ax.set_xticks(Ns); ax.set_xlabel("New B200 GPUs added"); ax.set_ylabel("J per request")
    ax.legend(fontsize=6.5, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG + "fig_brownfield_E.png"); fig.savefig(FIG + "fig_brownfield_E.pdf")
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "extra":
    ext = dict(load_scale=load_scale(), perturbation=perturbation(), brownfield_E=brownfield_e())
    json.dump(ext, open(RES + "extra.json", "w"), indent=1, default=str)
    print(json.dumps(ext, indent=1, default=str)[:6000])
