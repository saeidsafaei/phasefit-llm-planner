"""Accelerator specifications used throughout the study.

All compute figures are *dense* peak tensor throughput (FLOP/s), taken from the
vendors' public datasheets (sparse figures divided by two where only sparse
numbers are published). Memory in bytes (decimal GB), bandwidth in bytes/s.
`link` is the effective per-GPU unidirectional scale-up bandwidth used for
collective operations; `nic` is the per-GPU scale-out bandwidth used for KV
transfer between prefill and decode pools (400 Gb/s = 50 GB/s baseline).
`tdp` is the vendor maximum board power (W); None where not published per chip.
"""
from dataclasses import dataclass, field

T = 1e12
G = 1e9


@dataclass(frozen=True)
class Accel:
    name: str
    vendor: str
    peak: dict            # precision -> dense FLOP/s
    mem: float            # bytes
    bw: float             # bytes/s
    link: float           # bytes/s, effective unidirectional scale-up per GPU
    tdp: float | None     # W
    year: int
    fp4_fmt: str | None = None   # 'nvfp4' / 'mxfp4'

    def supports(self, prec: str) -> bool:
        return self.peak.get(prec) is not None


ACCELS = {
    # NVIDIA Hopper (H100 SXM, H200 SXM): BF16 989 / FP8 1979 dense TFLOPS
    "H100": Accel("H100", "NVIDIA", {"bf16": 989 * T, "fp8": 1979 * T, "fp4": None},
                  80 * G, 3.35e12, 450 * G, 700, 2022),
    "H200": Accel("H200", "NVIDIA", {"bf16": 989 * T, "fp8": 1979 * T, "fp4": None},
                  141 * G, 4.8e12, 450 * G, 700, 2024),
    # NVIDIA Blackwell B200 (HGX, 180 GB): BF16 2.25 / FP8 4.5 / FP4 9 PFLOPS dense
    "B200": Accel("B200", "NVIDIA", {"bf16": 2250 * T, "fp8": 4500 * T, "fp4": 9000 * T},
                  180 * G, 8.0e12, 900 * G, 1000, 2025, "nvfp4"),
    # GB200 NVL72 GPU (used only for model validation; no per-GPU TDP published)
    "GB200": Accel("GB200", "NVIDIA", {"bf16": 2500 * T, "fp8": 5000 * T, "fp4": 10000 * T},
                   186 * G, 8.0e12, 900 * G, None, 2025, "nvfp4"),
    # AMD CDNA3 / CDNA4
    "MI300X": Accel("MI300X", "AMD", {"bf16": 1307.4 * T, "fp8": 2614.9 * T, "fp4": None},
                    192 * G, 5.3e12, 448 * G, 750, 2023),
    "MI325X": Accel("MI325X", "AMD", {"bf16": 1307.4 * T, "fp8": 2614.9 * T, "fp4": None},
                    256 * G, 6.0e12, 448 * G, 1000, 2024),
    "MI355X": Accel("MI355X", "AMD", {"bf16": 2516.6 * T, "fp8": 5033.2 * T, "fp4": 10066.3 * T},
                    288 * G, 8.0e12, 504 * G, 1400, 2025, "mxfp4"),
    # Intel Gaudi 3 (OAM): BF16 = FP8 MME 1835 TFLOPS, 128 GB HBM2e 3.7 TB/s
    "Gaudi3": Accel("Gaudi3", "Intel", {"bf16": 1835 * T, "fp8": 1835 * T, "fp4": None},
                    128 * G, 3.7e12, 600 * G, 900, 2024),
}

# Accelerators considered by the planner (official per-chip TDP available)
PLANNER_ACCELS = ["H100", "H200", "B200", "MI300X", "MI325X", "MI355X", "Gaudi3"]

NIC_BW = 50 * G          # 400 Gb/s per GPU (baseline); varied in sensitivity study

# bytes per element for weights by precision (FP4 includes block-scale overhead)
WEIGHT_BYTES = {
    "bf16": 2.0,
    "fp8": 1.0 + 1.0 / 128 / 128 * 4,  # block-wise FP32 scales are negligible
    "nvfp4": 0.5 + 1.0 / 16,           # one FP8 scale per 16 elements
    "mxfp4": 0.5 + 1.0 / 32,           # one E8M0 scale per 32 elements
}


def weight_bytes(acc: Accel, prec: str) -> float:
    if prec == "fp4":
        return WEIGHT_BYTES[acc.fp4_fmt]
    return WEIGHT_BYTES[prec]


def kv_bytes(prec: str) -> float:
    """KV-cache element size. KV is kept at FP8 whenever weights are FP8 or FP4."""
    return 2.0 if prec == "bf16" else 1.0
