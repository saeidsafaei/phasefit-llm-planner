import pickle, sys
sys.path.insert(0,'.')
from planner import select, POLICIES, fix_static
def show(S, phi=0.25, **kw):
    prof = fix_static(pickle.load(open(f"../results/profile_{S}.pkl","rb")))
    m = prof["meta"]; tok = m["out_tokens"]
    print(f"== {S} {m['model']} {m['workload']} x{m['rate_x']} {m['slo']['name']} n={m['n']} H={m['horizon']:.0f}s  {kw}")
    for pol in ["Agg-Homo"] + list(POLICIES):
        b = select(prof, pol, phi=phi, **kw)
        if b is None: print(f"  {pol:11s} infeasible"); continue
        if pol == "Agg-Homo":
            a=b["agg"]; desc=f"{a['label']} x{a['n']} (chunk {a['chunk']}, B {a['bcap']})"
        else:
            desc=f"P {b['pre']['label']} x{b['pre']['n']} | D {b['dec']['label']} x{b['dec']['n']} (B {b['dec']['bcap']})"
        print(f"  {pol:11s} {b['energy']/tok*1000:7.1f} mJ/tok  GPUs {b['gpus']:3d}  {b['kw']:6.1f} kW   {desc}")
if __name__ == "__main__":
    for S in sys.argv[1:]:
        show(S); show(S, allow_fp4=False)
