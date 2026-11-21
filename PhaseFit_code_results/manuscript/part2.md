# 4. The PhaseFit model and planner

Fig. 2 outlines PhaseFit. A deployment request specifies a model, a workload trace, p99 TTFT and TPOT targets, the accelerator types available (optionally with inventory limits) and the admissible numeric precisions. PhaseFit (i) enumerates candidate *instance configurations* (accelerator, precision, parallelism), (ii) predicts the iteration time and power of each with the analytical model of Section 4.1, (iii) sizes each phase by trace-driven simulation (Section 4.3) and (iv) combines the per-phase results under a pairing policy (Section 4.4) or, for brownfield fleets, under inventory limits (Section 4.5).

![Fig. 2. Overview of PhaseFit. The analytical model is calibrated once on public measurements; planning then needs no access to the target hardware.](../figs/fig_overview.png)

## 4.1. Analytical iteration-time and power model

An *instance* is one model replica on $t$ accelerators of one type. Dense models use tensor parallelism (TP). MoE models use attention data parallelism combined with expert parallelism over the same $t$ devices (DEP), the layout adopted by current MoE serving stacks [@dsv3; @trtllm]. An *iteration* is one forward pass over $n_d$ decode tokens with mean context $\bar c$ plus $n_p$ prefill tokens. PhaseFit models its duration as

$$T = \max\left(T_{comp},\, T_{mem}\right) + T_{comm} + L\,\tau , \qquad (5)$$

where compute and memory traffic are assumed to overlap (roofline semantics [@williams2009]), communication is not overlapped, and $\tau$ is a fixed per-layer overhead covering kernel launches, synchronization and scheduling.

*Compute.* GEMM efficiency rises with the number of rows $M$ (tokens) a matrix multiplication processes. We use the saturating law

$$\eta(M) = \eta_{q}\,\frac{M}{M + M_{1/2}} , \qquad (6)$$

with a peak efficiency $\eta_q$ that depends on the precision class $q$ (one value for BF16/FP8 and one for FP4) and a half-saturation size $M_{1/2}$. For dense models all linear layers see $M=n_d+n_p$ rows. For MoE models the non-expert layers (attention projections, dense and shared FFNs, LM head) see $M=(n_d+n_p)/t$ rows per device under DEP, while each routed expert sees on average $M_e=(n_d+n_p)k/E$ rows. The compute time per device is

$$T_{comp} = \frac{F_{rep}}{\eta(M)\,P_{q'}} + \kappa\,\frac{F_{exp}}{\eta(M_e)\,P_{q}} + \frac{F_{att}}{\eta_{q}\,P_{bf16}} , \qquad (7)$$

where $F_{rep}$, $F_{exp}$ and $F_{att}$ are the per-device FLOPs of the non-expert linear layers, the routed experts and the attention score/value products (Eqs. (1)–(2)), $P_q$ is the dense peak of precision $q$, $q'$ is the precision of the non-expert weights (FP8 when experts are FP4, following public FP4 MoE checkpoints) and $\kappa \ge 1$ lumps the overheads of the expert path: load imbalance across experts, grouped-GEMM inefficiency and token permutation. For MLA we use the weight-absorbed form in decode (scores over $d_c+d_r$ dimensions, outputs over $d_c$) and the explicit form in prefill [@dsv2].

*Memory.* Per device and iteration,

$$T_{mem} = \frac{W_{touch} + n_d\,\bar c\,K/\sigma + (n_d+n_p)\,K/\sigma}{\eta_m\,BW} , \qquad (8)$$

where $\sigma$ is the number of devices sharing one token's KV cache ($\min(t,n_{kv})$ for GQA under TP; 1 for MLA and under DEP, where the batch rather than the heads is split, so the KV terms are divided by $t$), and $\eta_m$ is the achievable fraction of HBM bandwidth. For dense models $W_{touch}$ equals the resident weights. For MoE models only the experts that receive at least one token are read. With $n=n_d+n_p$ tokens routed to $k$ of $E$ experts, the expected touched fraction is

$$f_{touch}(n) = 1 - \left(1-\tfrac{1}{E}\right)^{n k} \approx 1 - e^{-nk/E} . \qquad (9)$$

This term explains why small-batch MoE decode costs far less than the total parameter count suggests, while large batches touch every expert.

*Communication.* Tensor parallelism issues two all-reduces per layer over BF16 activations. Expert parallelism issues a dispatch (FP8) and a combine (BF16) all-to-all per MoE layer:

$$T_{comm}^{TP} = 2L\left(\alpha + \tfrac{2(t-1)}{t}\,\tfrac{n\,d\,2}{BW_{link}}\right),\qquad T_{comm}^{DEP} = L_{moe}\left(2\alpha + \tfrac{n}{t}\,k\,d\,3\,\tfrac{t-1}{t}\,\tfrac{1}{BW_{link}}\right), \qquad (10)$$

with collective latency $\alpha = 8\,\mu s$ and the per-device scale-up bandwidth $BW_{link}$ of Table 1.

*Memory capacity.* An instance holds weights plus KV cache within a fraction 0.9 of HBM. In calibration (Eq. (14)) a request reserves cache for its prompt plus the maximum output length, which mirrors the no-eviction admission policy of production engines. In the trace-driven simulator a request reserves its prompt plus its actual output length, which assumes accurate output-length knowledge at admission (see Section 7.2).

*Power and energy.* Device power is modelled as an idle floor plus a utilization-proportional term,

$$P_{dev} = TDP\,\big(\phi + (1-\phi)\,u\big),\qquad u = \max\!\left(\tfrac{T_{comp}}{T},\,\tfrac{T_{mem}}{T}\right), \qquad (11)$$

with idle fraction $\phi = 0.25$ in the base case (varied from 0.15 to 0.40 in Section 6.4). The model reflects the observation that compute-bound prefill draws close to board power whereas memory-bound decode draws noticeably less [@patel2024split; @patel2024power]. Integrating Eq. (11) over a simulation of horizon $H$ gives, for $n$ instances of $t$ devices,

$$E = \phi\,E_{sta} + (1-\phi)\,E_{dyn},\qquad E_{sta} = n\,t\,TDP\,H,\qquad E_{dyn} = \sum_{\text{iterations}} t\,TDP\,u\,T . \qquad (12)$$

Storing $E_{sta}$ and $E_{dyn}$ separately lets every result be re-evaluated for any $\phi$ without re-simulation. Host CPU, network and cooling power are not included, so energies are accelerator energies.

*KV transfer.* When the two phases run on different instances, the KV cache of a prompt of $s$ tokens crosses the network in

$$T_{x} = \frac{s\,K}{BW_{nic}\,\min(t_p, t_d)} , \qquad (13)$$

using one 400 Gb/s NIC per device in the base case ($BW_{nic}$ = 50 GB/s; varied in Section 6.4). The cache is sent in the decode instance's format, so an FP8-KV decode pool receives FP8 data regardless of the prefill precision.

## 4.2. Calibration and validation

The model has five free parameters: $\eta_{bf16/fp8}$, $\eta_{fp4}$, $M_{1/2}$, $\tau$ and $\kappa$. The memory efficiency is fixed at $\eta_m = 0.85$, a typical achievable fraction of HBM bandwidth, rather than fitted. We fit the free parameters by minimizing the mean squared log error between predicted and measured output throughput on 22 public TensorRT-LLM measurements [@trtllm]: Llama-3.3-70B on B200 (FP4, TP1), GB200 (FP4, TP1) and H200 (FP8, TP2), and DeepSeek-R1 on B200 and GB200 (FP4, DEP4) and H200 (FP8, DEP8), at four input/output length pairs from 1k/1k to 32k/1k. For these maximum-load measurements the model's steady state is an aggregated instance whose iterations carry $B$ decode tokens and, on average, $B\,s/o$ prefill tokens, with

$$\Theta = \max_{B}\ \frac{B}{t\;T\!\left(B,\ s + o/2,\ B s/o\right)}\quad \text{s.t.}\ \ B(s + o_{max}) \le C_{KV},\ \ B \le B_{max},\ \ B(1 + s/o) \le N_{tok} , \qquad (14)$$

where $C_{KV}$ is the KV capacity in tokens and $B_{max}=2048$ and $N_{tok}=8192$ are the per-rank batch and token limits of the serving engine.

Validation uses 32 measurements that played no role in fitting: (i) 14 TensorRT-LLM throughput points for Qwen3-235B-A22B (B200, GB200, H200), an MoE model with GQA that was not used in fitting; (ii) the two MLPerf Inference v5.1 DeepSeek-R1 offline results quoted per GPU by NVIDIA (DGX H200: 1,253 tokens/s/GPU; GB200 NVL72: 4,024 tokens/s/GPU) [@nv_mlperf51], whose dataset has mean input and output lengths of 800 and 3,880 tokens and a 20k-token output cap [@mlc_r1]; and (iii) 16 batch-latency measurements of Llama-3.1-70B FP8 on 8× MI300X with vLLM 0.11.2 on ROCm 7.0 [@amdperf], which exercise a different vendor, software stack and the small-batch regime. Section 6.1 reports the results.

## 4.3. Decomposition under disaggregation and per-phase sizing

Let request $i$ arrive at $a_i$ with prompt length $s_i$ and output length $o_i$. Under disaggregation it is prefilled at time $f_i$ in the prefill pool (its first token is emitted then), its KV cache arrives at the decode pool at $f_i + T_{x,i}$, and its last token is produced at $e_i$. Then

$$\mathrm{TTFT}_i = f_i - a_i,\qquad \mathrm{TPOT}_i = \frac{e_i - f_i}{o_i - 1} = \frac{T_{x,i} + W^{d}_i + \sum_{j=1}^{o_i-1} T^{d}_{ij}}{o_i-1} , \qquad (15)$$

where $W^d_i$ is the wait for a decode slot and $T^d_{ij}$ are the decode step times seen by the request. $\mathrm{TTFT}_i$ depends only on the prefill pool's configuration and size. $\mathrm{TPOT}_i$ depends on the decode pool and on the stream of prefill completions, which is a smoothed version of the arrival stream, plus $T_{x,i}/(o_i-1)$. For the workloads studied, $T_{x,i}$ is a few milliseconds (Section 6.4), so this term is small. The joint problem

$$\min_{x_p,\,x_d}\ E_p(x_p) + E_d(x_d)\quad \text{s.t.}\ \ \mathrm{p99}(\mathrm{TTFT}) \le \hat T_{ttft},\ \ \mathrm{p99}(\mathrm{TPOT}) \le \hat T_{tpot},\ \ (x_p, x_d) \in \Pi,\ \ \textstyle\sum_{\text{phase}} g_h \le G_h\ \forall h \qquad (16)$$

therefore separates into a prefill problem and a decode problem. Here $x = (h, q, t, n)$ is an instance configuration with its count, $\Pi$ is the pairing policy (Section 4.4), $g_h$ is the number of devices of type $h$ used and $G_h$ is the inventory. PhaseFit solves each per-phase problem as follows:

1. **Enumerate** all $(h, q, t)$ whose weights fit and whose precision the device supports natively (FP4 only on B200 and MI355X); $t \in \{1,2,4,8\}$ for dense models and $t \in \{4, 8\}$ for MoE models.
2. **Screen** analytically. For decode, compute the largest batch $\hat B$ whose step time at context $\bar s + 0.75\,\bar o$ stays within $0.85\,\hat T_{tpot}$ and whose KV reservation fits, and estimate the instance count from Little's law, $n \approx \lambda\,\bar o\,T(\hat B)/\hat B$. For prefill, reject configurations that cannot prefill the p99 prompt within $\hat T_{ttft}$ on an idle instance. Every prefill candidate is kept, because prefill simulations are cheap and tight TTFT targets favour higher TP degrees. For decode, the best TP degree of each accelerator–precision pair is kept.
3. **Size by simulation.** Starting from the analytical estimate, find the smallest instance count whose simulated p99 TTFT (prefill) or p99 TPOT (decode, with a 3% margin for the transfer term) meets the target. This uses an exponential probe followed by binary search over the monotone feasibility predicate.
4. **Record** $E_{sta}$, $E_{dyn}$, device count and provisioned power for every sized configuration.

The cost of planning is linear in the number of configurations rather than quadratic in pairs. Pairs are scored afterwards in microseconds, and the selected pair is re-simulated end-to-end (prefill, KV transfer, decode) to confirm the SLOs.

**Simulator.** The discrete-event simulator replays the trace against the pools. Prefill instances serve first-come-first-served batches of up to 16,384 prompt tokens, and requests are dispatched to the instance with the smallest estimated backlog. Decode instances run continuous batching with batch cap $\hat B$ and admission by KV reservation, and new requests join the least-loaded instance with room. Because the batch composition of an instance changes only at arrivals and completions, the simulator advances groups of identical steps at once and re-evaluates Eq. (5) at the mid-interval context, which keeps a full planning run to minutes on a laptop-class CPU. The colocated baseline uses the same engine with Sarathi-style chunked prefill [@agrawal2024]: each iteration carries all running decodes plus up to $C$ prompt tokens, and $C \in \{256, 512, 1024, 2048\}$ and the batch cap are chosen analytically for the TPOT target before sizing by simulation. The first and last 5% of requests are excluded from latency percentiles to remove warm-up and drain effects.

## 4.4. Pairing policies and baselines

The pairing set $\Pi$ in Eq. (16) defines PhaseFit and the baselines, so all methods share the same model, simulator and sizing procedure and differ only in their degrees of freedom:

- **Agg-Homo**: colocated serving with chunked prefill on the best single configuration (Sarathi-Serve-style [@agrawal2024]).
- **Dis-Homo**: disaggregation with the same accelerator *and* precision in both phases; TP degree and counts per phase are free (DistServe-style [@zhong2024]).
- **Dis-HW**: different accelerators per phase allowed, precision fixed to FP8 in both (Splitwise/HexGen-2-style hardware heterogeneity [@patel2024split; @jiang2025]).
- **Dis-Prec**: same accelerator in both phases, precision chosen per phase.
- **PhaseFit-V**: accelerator, precision and parallelism chosen per phase, both phases from the same vendor (the case of KV-cache transfer within one software ecosystem).
- **PhaseFit-X**: as PhaseFit-V, but across vendors, which assumes a vendor-neutral KV transfer layer such as Mooncake's transfer engine [@qin2025].

Each method is evaluated with FP4 allowed and, as an accuracy-conservative variant, with FP4 disallowed.

## 4.5. Brownfield extension: scarce new accelerators in mixed pools

In a brownfield fleet, $G_{new}$ new accelerators join an installed base of older ones. If the new devices are too few to serve a whole phase, the per-phase pool must mix types. For a phase $\varphi$, a new configuration $x^{new}$ with $n^{new}$ instances and an old configuration $x^{old}$, PhaseFit routes a fraction $f$ of the requests to the new instances. Requests are selected deterministically by the golden-ratio sequence $\{i\cdot 0.618\ldots\} < f$, which preserves the burstiness of the trace in both sub-streams. PhaseFit finds the largest $f$ that the $n^{new}$ instances serve within the SLO by bisection, then sizes the old pool for the remaining $1-f$ by the procedure of Section 4.3. If the new instances can carry the whole phase, it deploys only as many as needed. The planner then searches the split $\beta = g^{new}_{pre}/G_{new}$ of new devices between the phases, $\beta \in \{0, 0.25, 0.5, 0.75, 1\}$, together with the new devices' precision in each phase, and returns the combination with minimum energy. Two common heuristics are special cases: *new-to-prefill* ($\beta=1$), which follows the Splitwise intuition that the compute-heavy prompt phase benefits most from the newest GPU [@patel2024split], and *new-to-decode* ($\beta=0$).

# 5. Experimental methodology

## 5.1. Workloads

We use the two public Azure LLM inference traces collected in November 2023 [@azure2023; @patel2024split]. Each record holds an arrival timestamp and the input and output token counts, but no content. *Azure-Conv* (conversation service) contains 19,366 requests in 58 minutes, and *Azure-Code* (coding assistant) contains 8,819 requests in 57 minutes. The traces are bursty (Fig. 3c) and have very different length profiles: Azure-Code prompts are almost twice as long as Azure-Conv prompts, while Azure-Code outputs are shorter by a factor of 7.6 (Table 3). To reach the load of a production cluster we compress inter-arrival times by a factor of 40 (Azure-Conv: 221 requests/s; Azure-Code: 103 requests/s) and replay the first 30 minutes of each trace, preserving burst structure and length statistics. We also evaluate load factors of 10 and 20.

Public reasoning-service traces with per-request lengths are not yet available. We therefore build a *Reasoning* workload that keeps the Azure-Conv arrival process (compressed 4×, i.e. 22 requests/s) and draws prompt and output lengths from log-normal distributions whose means match the MLPerf v5.1 DeepSeek-R1 dataset (800 and 3,880 tokens, outputs capped at 20,000 tokens) [@mlc_r1]. The dispersion parameters ($\sigma = 0.8$ for prompts and 0.9 for outputs) are our assumption. Fig. 3 and Table 3 characterize the three workloads.

![Fig. 3. Workload characterization. (a) Prompt-length and (b) output-length CDFs. (c) Request rate per minute of the original Azure traces.](../figs/fig_workloads.png)

**Table 3.** Workload statistics (original time base; lengths in tokens).

{{TABLE_WL}}

## 5.2. Models, SLOs and accelerators

We evaluate three model families with distinct bottlenecks (Table 4): Llama-3.1-70B [@llama3], a dense model with GQA; Qwen3-235B-A22B [@qwen3], an MoE model with GQA; and DeepSeek-R1 [@dsr1], an MoE model with MLA, whose KV cache is 4.7× smaller per token than Llama-3.1-70B's despite having 9.5× more parameters. Architectural parameters are taken from the public model configuration files.

**Table 4.** Models (parameter counts computed from the configuration files; KV size per token at BF16).

{{TABLE_MODELS}}

Two SLO tiers are used for the Azure workloads. *Standard* requires p99 TTFT ≤ 2 s and p99 TPOT ≤ 100 ms (10 tokens/s per user, as in the MLPerf Llama 3.1 8B server scenario). *Interactive* requires p99 TTFT ≤ 1 s and p99 TPOT ≤ 40 ms (25 tokens/s per user, the per-user rate of the MLPerf Llama 2 70B Interactive scenario [@nv_mlperf50]). The reasoning workload uses the MLPerf v5.1 DeepSeek-R1 server targets: p99 TTFT ≤ 2 s and p99 TPOT ≤ 80 ms [@mlc_r1]. The seven accelerators with published per-chip TDP in Table 1 are planning candidates. GB200 is used only for model validation.

## 5.3. Scenarios and metrics

Table 5 lists the six greenfield scenarios. For each, PhaseFit and the baselines of Section 4.4 are planned with energy as the objective. We report accelerator energy per request (J/request) and per output token (mJ/token), the number of accelerators, the provisioned board power (kW, the sum of TDPs), and the p99 TTFT and TPOT of an end-to-end re-simulation of the selected deployment. The brownfield case study (Section 6.5) starts from an all-H200 deployment of scenario A and adds 8, 16 or 32 B200 GPUs.

**Table 5.** Greenfield scenarios.

{{TABLE_SCEN}}

**Reproducibility.** The model, simulator, planner, calibration data and experiment scripts are written in Python (NumPy, SciPy, pandas) and run on a 2-core CPU. Every number in Section 6 is produced by these scripts from the public inputs cited above. The planner makes no use of measurements from the hardware being planned beyond those listed in Section 4.2.
