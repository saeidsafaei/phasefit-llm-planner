"""Assemble the manuscript: fill generated tables, number citations, convert with pandoc
(LaTeX math -> native Word equations) and post-style the .docx with python-docx."""
import json
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, ".")
from hw import ACCELS, NIC_BW
from models import MODELS
from refs import R
from workloads import azure, reasoning, describe

MS = "/home/claude/paper/manuscript/"
OUT = "/home/claude/paper/out/"


# ------------------------------------------------------------------ tables
def t_hw():
    rows = ["| Accelerator | Vendor / year | BF16 (TFLOPS) | FP8 (TFLOPS) | FP4 (TFLOPS) | HBM (GB) | BW (TB/s) | Ridge FP8 (FLOP/B) | Scale-up (GB/s)\\* | TDP (W) |",
            "|---------|----------|--------:|--------:|--------:|-----:|-----:|-------:|-------:|-----:|"]
    for k in ["H100", "H200", "B200", "GB200", "MI300X", "MI325X", "MI355X", "Gaudi3"]:
        a = ACCELS[k]
        fp4 = f"{a.peak['fp4']/1e12:,.0f}" if a.peak["fp4"] else "–"
        name = {"Gaudi3": "Gaudi 3", "GB200": "GB200 (NVL72)†"}.get(k, k)
        rows.append(f"| {name} | {a.vendor} / {a.year} | {a.peak['bf16']/1e12:,.0f} | {a.peak['fp8']/1e12:,.0f} | {fp4} | "
                    f"{a.mem/1e9:.0f} | {a.bw/1e12:.2f} | {a.peak['fp8']/a.bw:.0f} | {a.link/1e9:.0f} | "
                    f"{a.tdp if a.tdp else '–'} |")
    rows.append("")
    rows.append("\\*Effective unidirectional per-device bandwidth assumed for collectives. †Used for validation only; no per-GPU TDP is published.")
    return "\n".join(rows)


def t_rw():
    cols = ["Splitwise [@patel2024split]", "DistServe [@zhong2024]", "HexGen-2 [@jiang2025]",
            "Helix [@mei2025]", "Mélange [@griggs2024]", "DynamoLLM [@stojkovic2025]", "Vidur [@agrawal2024vidur]",
            "**PhaseFit**"]
    feats = [
        ("Prefill–decode disaggregation", "✓✓✓–––– ✓"),
        ("Heterogeneous accelerator types", "✓–✓✓✓–– ✓"),
        ("Multiple vendors", "––––––– ✓"),
        ("Numeric precision as a per-phase decision", "––––––– ✓"),
        ("Energy or power objective", "✓––––✓– ✓"),
        ("MoE models with MLA", "––––––– ✓"),
        ("FP4-capable 2025 accelerators", "––––––– ✓"),
    ]
    head = "| Feature | " + " | ".join(cols) + " |"
    sep = "|--------------------|" + "|".join([":--------:"] * len(cols)) + "|"
    rows = [head, sep]
    for f, marks in feats:
        m = [c for c in marks if c in "✓–"]
        rows.append(f"| {f} | " + " | ".join(m) + " |")
    return "\n".join(rows)


def t_wl():
    rows = ["| Workload | Requests | Duration (min) | Rate (req/s) | Prompt mean / p50 / p99 | Output mean / p50 / p99 |",
            "|----------------|--------:|--------:|--------:|--------------|--------------|"]
    for name, wl in (("Azure-Conv", azure("conv")), ("Azure-Code", azure("code")), ("Reasoning (R1-like)", reasoning())):
        d = describe(wl)
        rows.append(f"| {name} | {d['n']:,} | {d['duration_s']/60:.0f} | {d['rate']:.2f} | "
                    f"{d['isl_mean']:,.0f} / {d['isl_p50']:,.0f} / {d['isl_p99']:,.0f} | "
                    f"{d['osl_mean']:,.0f} / {d['osl_p50']:,.0f} / {d['osl_p99']:,.0f} |")
    rows.append("")
    rows.append("Reasoning lengths are synthetic (log-normal, means matched to the MLPerf v5.1 DeepSeek-R1 dataset); its arrival process is that of Azure-Conv.")
    return "\n".join(rows)


def t_models():
    rows = ["| Model | Type | Layers | Total / active params (B) | Attention | Experts (routed, top-k, shared) | KV per token (KiB, BF16) |",
            "|--------------|------|------:|-----------|-----------|------------|----------:|"]
    for k, typ, att in (("Llama-3.1-70B", "Dense", "GQA 64/8 heads"),
                        ("Qwen3-235B-A22B", "MoE", "GQA 64/4 heads"),
                        ("DeepSeek-R1", "MoE", "MLA (512 + 64)")):
        m = MODELS[k]
        ex = f"{m.n_exp}, {m.top_k}, {m.n_shared}" if m.moe else "–"
        rows.append(f"| {k} | {typ} | {m.L} | {m.params_total/1e9:.1f} / {m.params_active/1e9:.1f} | {att} | {ex} | "
                    f"{m.kv_elems_per_token*2/1024:.1f} |")
    return "\n".join(rows)


def t_scen():
    from experiments import SCEN
    rows = ["| ID | Model | Workload | Load factor | Requests simulated | SLO (p99 TTFT / TPOT) |",
            "|---|---------------|------------|-------:|---------:|----------------|"]
    from experiments import workload
    for S in "ABCDEF":
        model, w, rx, slo = SCEN[S]
        n = len(workload(w, rx)[0])
        wn = {"conv": "Azure-Conv", "code": "Azure-Code", "reason": "Reasoning"}[w]
        rows.append(f"| {S} | {model} | {wn} | ×{rx} | {n:,} | {slo.name}: {slo.ttft:g} s / {slo.tpot*1e3:.0f} ms |")
    return "\n".join(rows)


def t_valid():
    d = json.load(open("../results/calibration.json"))
    rows = d["rows"]
    groups = [("TRT", "Llama-3.3-70B", "Fit", "TensorRT-LLM, max load", "B200, GB200, H200"),
              ("TRT", "DeepSeek-R1", "Fit", "TensorRT-LLM, max load", "B200, GB200, H200"),
              ("TRT", "Qwen3-235B-A22B", "Held out", "TensorRT-LLM, max load", "B200, GB200, H200"),
              ("MLP", "DeepSeek-R1", "Held out", "MLPerf v5.1 offline", "H200, GB200"),
              ("AMD", "Llama-3.1-70B", "Held out", "vLLM/ROCm batch latency", "8× MI300X")]
    out = ["| Set | Model | Source / metric | Accelerators | n | MAPE (%) | Median APE (%) | Max APE (%) |",
           "|-------|-------------|---------------|-----------|----:|-------:|-------:|-------:|"]
    for src, model, split, s, accs in groups:
        e = [r["ape"] for r in rows if r["src"] == src and r["model"] == model]
        out.append(f"| {split} | {model} | {s} | {accs} | {len(e)} | {np.mean(e):.1f} | {np.median(e):.1f} | {np.max(e):.1f} |")
    tr = [r["ape"] for r in rows if r["split"] == "train"]
    te = [r["ape"] for r in rows if r["split"] == "test"]
    out.append(f"| **All fit** | | | | {len(tr)} | **{np.mean(tr):.1f}** | {np.median(tr):.1f} | {np.max(tr):.1f} |")
    out.append(f"| **All held out** | | | | {len(te)} | **{np.mean(te):.1f}** | {np.median(te):.1f} | {np.max(te):.1f} |")
    return "\n".join(out)


TABLES = dict(TABLE_HW=t_hw, TABLE_RW=t_rw, TABLE_WL=t_wl, TABLE_MODELS=t_models, TABLE_SCEN=t_scen,
              TABLE_VALID=t_valid)


# ------------------------------------------------------------------ citations
def number_citations(text):
    order = []
    def repl(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in R:
                raise KeyError(k)
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        nums = sorted(set(nums))
        # compress ranges
        parts, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(x) for x in nums[i:j + 1]))
            i = j + 1
        return "[" + ", ".join(parts) + "]"
    text = re.sub(r"\[(@[^\]]+)\]", repl, text)
    refs = "\n".join(f"[{i+1}] {R[k]}\n" for i, k in enumerate(order))
    return text, refs, order


def build(parts, extra_tables=None):
    text = "\n\n".join(open(MS + p).read() for p in parts)
    tabs = dict(TABLES)
    if extra_tables:
        tabs.update(extra_tables)
    for k, f in tabs.items():
        if "{{" + k + "}}" in text:
            text = text.replace("{{" + k + "}}", f() if callable(f) else f)
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", text)
    if left:
        print("WARNING unfilled:", left)
    text, refs, order = number_citations(text)
    text = text.replace("{{REFERENCES}}", refs)
    open(OUT + "manuscript_full.md", "w").write(text)
    return order


# ------------------------------------------------------------------ result tables
def _summ():
    return json.load(open("../results/summary.json"))


def t_main():
    s = _summ()["main"]
    rows = ["| ID | Agg-Homo | Dis-HW (FP8) | Dis-Homo | **PhaseFit-X** | PhaseFit-X deployment (prefill; decode) | GPUs | p99 TTFT / TPOT | Saving vs Agg / vs Dis-HW |",
            "|----|-----|-----|-----|-----|---------------------|----:|---------|---------:|"]
    for S in "ABCDEF":
        if S not in s:
            continue
        r = s[S]["res"]
        a, h, d, p = (r.get(k) for k in ("Agg-Homo|fp4", "Dis-HW|fp4", "Dis-Homo|fp4", "PhaseFit-X|fp4"))
        v = p.get("validated", {})
        dep = f"{p['n_pre']}× {p['pre']}; {p['n_dec']}× {p['dec']}"
        rows.append(f"| {S} | {a['J_req']:.0f} | {h['J_req']:.0f} | {d['J_req']:.0f} | **{p['J_req']:.0f}** | {dep} | "
                    f"{p['gpus']} | {v.get('ttft_p99', float('nan')):.2f} s / {v.get('tpot_p99', float('nan'))*1e3:.0f} ms | "
                    f"{(1-p['J_req']/a['J_req'])*100:.1f}% / {(1-p['J_req']/h['J_req'])*100:.1f}% |")
    rows.append("")
    rows.append("NOTE: Energies are accelerator energies per request (J) at idle fraction φ = 0.25 with FP4 allowed. Latencies are from an end-to-end re-simulation of the PhaseFit-X deployment (prefill → KV transfer → decode).")
    return "\n".join(rows)


def t_nofp4():
    s = _summ()["main"]
    rows = ["| ID | Agg-Homo | Dis-Homo | PhaseFit-X | PhaseFit-X deployment | Saving vs Agg | Energy increase vs FP4 allowed |",
            "|----|------|------|------|----------------------|------|-------|"]
    for S in "ABCDEF":
        if S not in s:
            continue
        r = s[S]["res"]
        a, d, p, p4 = (r.get(k) for k in ("Agg-Homo|nofp4", "Dis-Homo|nofp4", "PhaseFit-X|nofp4", "PhaseFit-X|fp4"))
        if not (a and d and p):
            continue
        rows.append(f"| {S} | {a['J_req']:.0f} | {d['J_req']:.0f} | {p['J_req']:.0f} | {p['n_pre']}× {p['pre']}; {p['n_dec']}× {p['dec']} | "
                    f"{(1-p['J_req']/a['J_req'])*100:.1f}% | +{(p['J_req']/p4['J_req']-1)*100:.1f}% |")
    return "\n".join(rows)


def t_metrics():
    rows = ["| Accelerator | pJ per FLOP, FP8 | pJ per FLOP, best native format | J per TB of HBM traffic | J per GB of weights read, best format | HBM GB per kW |",
            "|----------|---------:|----------:|---------:|-----------:|--------:|"]
    from hw import weight_bytes
    for k in ["H100", "H200", "B200", "MI300X", "MI325X", "MI355X", "Gaudi3"]:
        a = ACCELS[k]
        best = a.peak["fp4"] or a.peak["fp8"]
        wb = weight_bytes(a, "fp4") if a.peak["fp4"] else 1.0
        jtb = a.tdp / (a.bw / 1e12)
        rows.append(f"| {'Gaudi 3' if k == 'Gaudi3' else k} | {a.tdp / a.peak['fp8'] * 1e12:.3f} | {a.tdp / best * 1e12:.3f} | {jtb:.0f} | "
                    f"{jtb * wb:.0f} | {a.mem / 1e9 / (a.tdp / 1e3):.0f} |")
    rows.append("")
    rows.append("All quantities at board power (TDP) and peak rates. Prefill energy tracks pJ/FLOP; decode energy tracks J per byte moved, which the weight format scales (one FP8 or FP4 element per parameter, including block-scale overhead).")
    return "\n".join(rows)


TABLES.update(TABLE_MAIN=t_main, TABLE_NOFP4=t_nofp4, TABLE_METRICS=t_metrics)


# ------------------------------------------------------------------ docx
def wrap_styles(md):
    """Table titles and table notes get their own paragraph styles."""
    out = []
    for para in md.split("\n\n"):
        p = para.strip()
        if re.match(r"^\*\*Table \d+\.\*\*", p):
            out.append('::: {custom-style="TableTitle"}\n' + p + "\n:::")
        elif p.startswith("NOTE:"):
            out.append('::: {custom-style="TableNote"}\n' + p[5:].strip() + "\n:::")
        elif re.match(r"^\[\d+\] ", p):
            out.append('::: {custom-style="Reference"}\n' + p + "\n:::")
        else:
            out.append(para)
    return "\n\n".join(out)


def to_docx(md_path, docx_path):
    md = open(md_path).read()
    md = wrap_styles(md)
    # figures at full text width
    md = re.sub(r"!\[(.*?)\]\((\.\./figs/[^)]+)\)", lambda m: f"![{m.group(1)}]({m.group(2)}){{width={'9cm' if ('fig_split' in m.group(2) or 'fig_brownfield_E' in m.group(2)) else '16cm'}}}", md)
    tmp = md_path.replace(".md", "_styled.md")
    open(tmp, "w").write(md)
    subprocess.run(["pandoc", tmp, "-f", "markdown+tex_math_dollars+pipe_tables+fenced_divs+superscript",
                    "-t", "docx", "--reference-doc", OUT + "reference.docx", "-o", docx_path,
                    "--resource-path", MS], check=True)
    post_style(docx_path)


def post_style(path):
    from docx import Document
    from docx.shared import Pt, Cm
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from docx.enum.table import WD_TABLE_ALIGNMENT
    d = Document(path)

    def border(cell, **kw):
        tcPr = cell._tc.get_or_add_tcPr()
        b = tcPr.find(qn("w:tcBorders"))
        if b is None:
            b = OxmlElement("w:tcBorders"); tcPr.append(b)
        for edge, val in kw.items():
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), val["val"]); e.set(qn("w:sz"), str(val["sz"]))
            e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
            b.append(e)

    for t in d.tables:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        nrows = len(t.rows)
        for i, row in enumerate(t.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.space_before = Pt(1)
                    for r in p.runs:
                        r.font.size = Pt(8.5)
                        r.font.name = "Times New Roman"
                        if i == 0:
                            r.font.bold = True
                kw = {}
                if i == 0:
                    kw["top"] = dict(val="single", sz=8); kw["bottom"] = dict(val="single", sz=4)
                if i == nrows - 1:
                    kw["bottom"] = dict(val="single", sz=8)
                if kw:
                    border(cell, **kw)
        # repeat header row
        trPr = t.rows[0]._tr.get_or_add_trPr()
        h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); trPr.append(h)
    # line numbers (continuous) and page numbers in footer
    for sec in d.sections:
        sectPr = sec._sectPr
        ln = OxmlElement("w:lnNumType")
        ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous"); ln.set(qn("w:distance"), "284")
        after = ["w:pgNumType", "w:cols", "w:formProt", "w:vAlign", "w:noEndnote", "w:titlePg",
                 "w:textDirection", "w:bidi", "w:rtlGutter", "w:docGrid", "w:printerSettings"]
        nxt = next((c for c in sectPr if c.tag in {qn(t) for t in after}), None)
        if nxt is not None:
            nxt.addprevious(ln)
        else:
            sectPr.append(ln)
        fp = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
        fp.alignment = 1
        pPr = fp._p.get_or_add_pPr()
        pPr.append(OxmlElement("w:suppressLineNumbers"))
        run = fp.add_run()
        for tag, text in (("begin", None), (None, "PAGE"), ("separate", None), ("end", None)):
            if tag:
                fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), tag); run._r.append(fc)
            else:
                it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = text
                run._r.append(it)
    d.save(path)


def t_fleets():
    f = json.load(open("../results/fleets.json"))
    fleets = ["2024 generation", "AMD only", "NVIDIA only", "Hopper + Gaudi 3"]
    rows = ["| ID | " + " | ".join(fleets) + " |", "|---|" + "|".join([":-----------:"] * len(fleets)) + "|"]
    for S in "ABCDEF":
        cells = []
        for fl in fleets:
            x = f[S][fl]
            g = x["gain"]
            pre, dec = x["cfg"].split(" + ")
            het = pre.split("/")[0] != dec.split("/")[0] or pre.split("/")[1] != dec.split("/")[1]
            cells.append(f"{g:.1f}%" + (f" ({pre.split('/')[0]}/{pre.split('/')[1]} + {dec.split('/')[0]}/{dec.split('/')[1]})" if het else ""))
        rows.append(f"| {S} | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("NOTE: Energy saving of PhaseFit-X over Dis-Homo when only the listed accelerators are available (FP4 allowed where supported). The heterogeneous pair selected is shown when it differs from a homogeneous one. 2024 generation = H100, H200, MI300X, MI325X, Gaudi 3.")
    return "\n".join(rows)


def t_sens():
    s = _summ()["sens"]
    e = json.load(open("../results/extra.json"))
    rows = ["| Factor | Setting | Effect on the PhaseFit-X decision | Effect on energy / latency |", "|-------------|--------------------|------------------------|----------------------------|"]
    ph = s["phi"]
    for S in "ABCDEF":
        if S not in ph:
            continue
        ch = {v["choice"] for v in ph[S].values()}
        sv = [ (1 - v["phasefit"] / v["agg"]) * 100 for v in ph[S].values()]
        rows.append(f"| Idle fraction φ | 0.15 / 0.25 / 0.40, scenario {S} | {'unchanged' if len(ch) == 1 else 'changes'} | saving vs Agg-Homo {min(sv):.1f}–{max(sv):.1f}% |")
    for S in ("A", "C", "E"):
        n = s["nic"][S]
        t6, t50 = n["6.25"]["tpot_p99"] * 1e3, n["50.0"]["tpot_p99"] * 1e3
        rows.append(f"| NIC bandwidth per GPU | 6.25–100 GB/s, scenario {S} | unchanged (SLO met at all settings) | p99 TPOT {t50:.1f} ms at 50 GB/s, {t6:.1f} ms at 6.25 GB/s |")
    L = e["load_scale"]
    for S in ("A10", "A20", "A"):
        r = L[S]["res"]
        a, p = r["Agg-Homo|fp4"], r["PhaseFit-X|fp4"]
        p8, d8 = r["PhaseFit-X|nofp4"], r["Dis-Homo|nofp4"]
        rows.append(f"| Load factor | ×{L[S]['rate_x']}, scenario A | {p['cfg']} | saving vs Agg-Homo {(1-p['J_req']/a['J_req'])*100:.1f}%; without FP4, heterogeneity gain {(1-p8['J_req']/d8['J_req'])*100:.1f}% |")
    P = e["perturbation"]
    names = {"etac-20": "η, η₄ −20%", "etam-12": "η_m 0.85 → 0.75", "tau+100": "τ ×2", "kappa+50": "κ ×1.5"}
    base_cfg = P["base"]["true"]["cfg"]
    for k, lab in names.items():
        if k not in P:
            continue
        c = P[k]["true"]["cfg"]
        same = [x.split(" x")[0] for x in c.split(" + ")] == [x.split(" x")[0] for x in base_cfg.split(" + ")]
        rows.append(f"| Model parameters | {lab}, scenario A | {'same configurations' if same else 'different configurations'} ({c}) | energy {(P[k]['true']['J_req']/P['base']['true']['J_req']-1)*100:+.1f}%; regret of the nominal decision {P[k].get('regret_pct', float('nan')):.1f}% |")
    return "\n".join(rows)


def t_brown():
    b = _summ()["brownfield"]
    e = json.load(open("../results/extra.json"))["brownfield_E"]
    rows = ["| Workload | New B200 | Splitwise-style | Prefill only | Decode only | **PhaseFit** | PhaseFit placement of the new GPUs |",
            "|------|----|------|------|------|-----|--------------------|"]
    base = b["base"]
    for N, d in b["N"].items():
        ss = lambda k: (1 - d[k]["mJ"] / base) * 100
        pf = d["PhaseFit (β, precisions)"]
        place = (f"β = {pf['alpha']:.2f}; P: " + (f"{pf['n_new_pre']} {pf['pp'].upper()} ({pf['f_pre']*100:.0f}%)" if pf['n_new_pre'] else "none")
                 + "; D: " + (f"{pf['n_new_dec']} {pf['pd'].upper()} ({pf['f_dec']*100:.0f}%)" if pf['n_new_dec'] else "none"))
        rows.append(f"| Conversation (A) | {N} | {ss('New→prefill, FP8 (Splitwise-style)'):.1f}% | {ss('New→prefill, best precision'):.1f}% | {ss('New→decode, best precision'):.1f}% | **{ss('PhaseFit (β, precisions)'):.1f}%** | {place} |")
    for N, d in e["N"].items():
        pf = d["PhaseFit"]
        rk = lambda c: int(re.search(r"(\d+)$", c).group(1))
        place = (f"β = {pf['beta']:.2f}; P: " + (f"{pf['n_new_pre'] * rk(pf['pp'])} {pf['pp'].upper()} ({pf['f_pre']*100:.0f}%)" if pf['n_new_pre'] else "none")
                 + "; D: " + (f"{pf['n_new_dec'] * rk(pf['pd'])} {pf['pd'].upper()} ({pf['f_dec']*100:.0f}%)" if pf['n_new_dec'] else "none"))
        rows.append(f"| Reasoning (E) | {N} | {d['New→prefill, FP8']['saving']:.1f}% | {d['New→prefill, best']['saving']:.1f}% | {d['New→decode, best']['saving']:.1f}% | **{pf['saving']:.1f}%** | {place} |")
    rows.append("")
    rows = [re.sub(r"\| -(\d)", r"| −\1", r_) for r_ in rows]
    rows.append(f"NOTE: Energy savings relative to serving the same load with H200 GPUs only (FP8; conversational: {base:.0f} mJ per output token; reasoning: {e['base_J_req']:.0f} J per request). Splitwise-style: new GPUs serve prefill at FP8. Prefill only / Decode only: new GPUs serve one phase at the best precision. B200 instances are TP1 for Llama-3.1-70B and DEP4 (FP4) or DEP8 (FP8/FP4) for DeepSeek-R1; the remaining load is served by H200 instances sized to the same SLO. Placement: number of B200 GPUs used in prefill (P) and decode (D), their precision (and expert-parallel degree for DeepSeek-R1), and the share of prompts (P) or requests (D) they serve.")
    return "\n".join(rows)


TABLES.update(TABLE_FLEETS=t_fleets, TABLE_SENS=t_sens, TABLE_BROWN=t_brown)


if __name__ == "__main__":
    parts = sys.argv[1].split(",") if len(sys.argv) > 1 else ["part0.md", "part1.md", "part2.md", "part3.md", "part4.md"]
    extra = json.load(open("../results/fill.json")) if len(sys.argv) > 2 else None
    order = build(parts, extra)
    to_docx(OUT + "manuscript_full.md", OUT + "manuscript.docx")
    print("refs cited:", len(order), "of", len(R))
