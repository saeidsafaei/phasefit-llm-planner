"""Published measurements used to calibrate and validate the analytical model.

Sources
 [TRT]  NVIDIA TensorRT-LLM 1.2.1 'Performance Overview' (trtllm-bench, max load,
        uniform ISL/OSL, output tokens/s per GPU), page updated 16 Apr 2026.
 [AMD]  AMD ROCm performance results, vLLM 0.11.2 / ROCm 7.0.0, 8x MI300X, FP8,
        benchmark_latency (seconds per batch), released 11 Dec 2025.
 [MLP]  MLPerf Inference v5.1 DeepSeek-R1 offline, per-GPU values reported by
        NVIDIA (DGX H200: 1,253; GB200 NVL72: 4,024 tok/s/GPU); dataset mean
        ISL 800 / OSL 3,880 (MLCommons).
Each record: (source, model, accel, precision, parallelism, t, isl, osl, metric, value)
metric 'thr' = output tok/s per GPU at max load; 'lat' = batch latency (s) with batch size
stored in field `batch`.
"""

THR = []   # (src, model, acc, prec, par, t, isl, osl, value)

_llama = {
    ("B200", "fp4", "tp", 1): [6943, 3270, 1347, 279],
    ("GB200", "fp4", "tp", 1): [7576, 3856, 1593, 315],
    ("H200", "fp8", "tp", 2): [2637, 1995, 544, 120],
}
_llama_len = [(1024, 1024), (1024, 8192), (8192, 1024), (32768, 1024)]
for (acc, prec, par, t), vals in _llama.items():
    for (i, o), v in zip(_llama_len, vals):
        THR.append(("TRT", "Llama-3.3-70B", acc, prec, par, t, i, o, v))

_ds_len = [(1024, 1024), (1024, 8192), (1024, 32768), (8192, 1024)]
_ds = {
    ("B200", "fp4", "dep", 4): [5757, 4228, 1472, 1176],
    ("GB200", "fp4", "dep", 4): [4989, 4474, 1582, 1276],
    ("H200", "fp8", "dep", 8): [1724, 1335, None, None],
}
for (acc, prec, par, t), vals in _ds.items():
    for (i, o), v in zip(_ds_len, vals):
        if v is not None:
            THR.append(("TRT", "DeepSeek-R1", acc, prec, par, t, i, o, v))

_qw_len = [(1024, 1024), (1024, 8192), (1024, 32768), (8192, 1024), (32768, 1024)]
_qw = {
    ("B200", "fp4", "dep", 4): [6423, 3881, 1216, 1518, 326],
    ("GB200", "fp4", "dep", 4): [6777, 3955, 1240, 1575, 342],
    ("H200", "fp8", "dep", 4): [3494, 1677, None, 679, 143],
}
for (acc, prec, par, t), vals in _qw.items():
    for (i, o), v in zip(_qw_len, vals):
        if v is not None:
            THR.append(("TRT", "Qwen3-235B-A22B", acc, prec, par, t, i, o, v))

# MLPerf v5.1 DeepSeek-R1 offline (mean lengths); H200 run with DEP8 FP8, GB200 with FP4
THR.append(("MLP", "DeepSeek-R1", "H200", "fp8", "dep", 8, 800, 3880, 1253))
THR.append(("MLP", "DeepSeek-R1", "GB200", "fp4", "dep", 4, 800, 3880, 4024))

LAT = []   # (src, model, acc, prec, par, t, batch, isl, osl, seconds)
_lat = {
    (128, 2048): [16.015, 18.683, 19.245, 20.468, 22.137, 25.571, 32.987, 46.426],
    (2048, 2048): [16.421, 19.035, 20.221, 21.483, 24.350, 29.776, 40.625, 63.671],
}
for (i, o), vals in _lat.items():
    for b, v in zip([1, 2, 4, 8, 16, 32, 64, 128], vals):
        LAT.append(("AMD", "Llama-3.1-70B", "MI300X", "fp8", "tp", 8, b, i, o, v))

# Train / test split: fit on Llama-3.3-70B and DeepSeek-R1 (TRT); everything else held out
def is_train(rec):
    return rec[0] == "TRT" and rec[1] in ("Llama-3.3-70B", "DeepSeek-R1")
