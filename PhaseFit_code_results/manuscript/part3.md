# 6. Results

## 6.1. Accuracy of the analytical model (RQ1)

The fitted parameters are $\eta_{bf16/fp8} = 0.463$, $\eta_{fp4} = 0.295$, $M_{1/2} = 24.7$ rows, $\tau = 96\ \mu s$ per layer and $\kappa = 2.95$. Three of these values are informative in themselves. FP4 GEMMs reach only 64% of the relative efficiency of FP8 GEMMs, so FP4 delivers about 1.3× rather than the nominal 2× of FP8 compute throughput in these measurements. The per-layer overhead amounts to 7.7 ms per iteration for an 80-layer model, which reflects host-side scheduling in the serving engine and dominates small-batch decode. The expert path runs about three times less efficiently than a dense GEMM of the same FLOP count, which captures load imbalance, grouped-GEMM inefficiency and token permutation together.

Table 6 and Fig. 4 summarize the accuracy. On the 22 fitting points the mean absolute percentage error (MAPE) is 7.5%. On the 32 held-out points it is 19.1% (median 17.9%). The model generalizes to an MoE family it was not fitted on (Qwen3-235B-A22B, 18.0%) and to the MLPerf v5.1 DeepSeek-R1 results (13.9%), whose length distribution and 20k-token output cap differ from the fitting data. It is least accurate in two regimes. First, it over-predicts Qwen3 throughput by 28–33% at 1k-token prompts on B200 and GB200, where very large MoE batches are probably limited by engine settings that the model does not capture. Second, for vLLM on 8× MI300X it over-predicts latency at batch 1 by 37–40%, because the fitted overhead $\tau$ belongs to a different engine, and under-predicts it at batch 128 by 42–45%, where that stack's attention kernels fall behind the roofline. The validation therefore supports using the model for *relative* comparisons of deployments within one software ecosystem. It also shows that cross-stack absolute predictions carry errors of up to about 45%. Section 6.4 shows that the planner's decisions are insensitive to errors of this size.

**Table 6.** Accuracy of the analytical model (APE = absolute percentage error).

{{TABLE_VALID}}

![Fig. 4. Predicted versus measured performance. (a) Maximum-load output throughput per GPU (TensorRT-LLM and MLPerf). (b) End-to-end batch latency of Llama-3.1-70B FP8 on 8× MI300X with vLLM (held out). The shaded band marks ±20%.](../figs/fig_validation.png)

## 6.2. Greenfield deployments (RQ2, RQ3)

Table 7 and Fig. 5 compare the policies of Section 4.4 in the six greenfield scenarios. Every deployment selected by PhaseFit-X met both p99 targets in the end-to-end re-simulation, with p99 TTFT between 0.21 s and 1.94 s and p99 TPOT between 21 ms and 38 ms.

**Table 7.** Greenfield results with FP4 allowed (J per request).

{{TABLE_MAIN}}

![Fig. 5. Energy per request of four planning policies in the six scenarios, normalized to the colocated baseline with FP4 allowed. (a) FP4 allowed. (b) FP4 not allowed. Scenarios: A, B = Llama-3.1-70B on Azure-Conv (Standard, Interactive); C, D = Llama-3.1-70B on Azure-Code (Standard, Interactive); E = DeepSeek-R1 on the reasoning workload; F = Qwen3-235B-A22B on Azure-Conv.](../figs/fig_main.png)

**Disaggregation.** Relative to the best colocated deployment, which also had all accelerators and precisions to choose from, disaggregation cut energy per request by 11.0–42.4%. The saving grows with the tightness of the TPOT target: from 11.0% to 29.3% between the Standard and Interactive tiers on Azure-Conv, and from 18.3% to 35.6% on Azure-Code. In a colocated instance every prefill chunk lengthens the decode step, so a tight TPOT forces both a small chunk and a small batch, and more instances are needed. The saving is largest for the MoE model (42.4% for Qwen3-235B-A22B). There, colocated prefill chunks touch nearly all experts in every iteration (Eq. (9)), which removes the weight-reuse advantage that pure decode batches enjoy. The disaggregated deployments also used 17–50% fewer accelerators.

**Precision.** Table 8 repeats the planning with FP4 disallowed. The best FP8-only deployments used 17.8–29.2% more energy than the FP4 deployments, so native FP4 reduced energy by 15.1–22.6%. The FP4 saving is smallest for the two MoE models (18.1% and 17.8% more energy without FP4), because their non-expert layers remain in FP8 and the expert path has lower GEMM efficiency. A Splitwise-style policy that permits different accelerators per phase but fixes FP8 (Dis-HW) used more energy than PhaseFit-X in every scenario; PhaseFit-X saved 15.1–22.6% relative to it. In scenario A, Dis-HW even used 12.8% more energy than colocated serving with FP4. On this hardware generation, precision is a first-order deployment decision.

**Table 8.** Greenfield results with FP4 disallowed (J per request).

{{TABLE_NOFP4}}

**Heterogeneity.** With FP4 allowed, PhaseFit-X chose the same accelerator, B200, for both phases in all six scenarios, and its energy equalled that of homogeneous disaggregation (Dis-Homo) exactly. It also chose the same precision in both phases. Heterogeneous pairs appeared only when FP4 was disallowed. In scenario C, B200 prefill was paired with a single MI300X decode instance (0.2% saving). In scenario E, an MI325X DEP4 prefill instance was paired with B200 decode (2.9% saving, four fewer accelerators). In the second case the gain comes from instance granularity rather than from phase specialization: DeepSeek-R1 in FP8 needs eight B200 GPUs per instance but fits in four MI325X GPUs (256 GB each), and the reasoning workload needs only one prefill instance. The same effect appears at low load. At load factor 10 without FP4, an MI355X decode instance replaced two B200 instances and saved 4.0% (Section 6.4).

Table 9 explains why. Prefill energy scales with joules per FLOP and decode energy with joules per byte moved, the latter scaled by the bytes per parameter of the weight format. With FP4, B200 leads both metrics by a wide margin: 0.111 pJ/FLOP and 70 J per GB of weights read, against 0.139 pJ/FLOP and 93 J/GB for MI355X and 0.287 pJ/FLOP and 142 J/GB for MI300X. When one device dominates both metrics, no pairing can beat it. A heterogeneous pair can win only when the two rankings disagree. MI355X and MI300X without FP4 are an example: MI355X leads in pJ/FLOP (0.278 against 0.287), whereas MI300X leads in J/TB (142 against 175). Even then the margins are a few percent. This contrasts with the A100/H100 setting of Splitwise, in which the token phase did not need the newer GPU's compute and could run at lower power and cost on the older one [@patel2024split].

**Table 9.** Energy-relevant efficiency metrics of the accelerators (from Table 1).

{{TABLE_METRICS}}

Restricting the fleet does not change the picture (Table 10). On 2024-generation-only, AMD-only, NVIDIA-only and Hopper + Gaudi 3 fleets, the best heterogeneous deployment improved on the best homogeneous one by at most 1.6%.

**Table 10.** Energy saving of heterogeneous over homogeneous disaggregation on restricted fleets.

{{TABLE_FLEETS}}

## 6.3. Where the energy goes

Fig. 6 splits the energy of the PhaseFit-X deployments by phase. For the Azure workloads prefill dominates, taking 83% of the energy on Azure-Conv and 96–97% on Azure-Code. The reason is that each request carries 5.5–73 times more prompt tokens than output tokens (Table 3), and each prompt token costs about as many FLOPs as an output token. For the reasoning workload the split reverses: decode takes 88% of the energy. These shares determine where efficiency gains matter, as the brownfield study shows (Section 6.5).

Utilization differs sharply by phase. Decode pools ran at 96–100% busy time, because continuous batching keeps them saturated up to the TPOT-limited batch size. Prefill pools ran at 26–73%. The lowest values occur under tight TTFT targets with long prompts (scenario D: 26%), because the pool must keep enough idle capacity to serve a 7.4k-token p99 prompt almost immediately. In such pools the idle power floor is a significant share of the energy, which is why PhaseFit selected TP2 rather than TP1 prefill instances in scenarios C and D: a shorter single-prompt latency allows fewer instances.

![Fig. 6. Share of accelerator energy spent in prefill and decode in the PhaseFit-X deployments (FP4 allowed).](../figs/fig_split.png)

## 6.4. Sensitivity and robustness

Table 11 summarizes the sensitivity studies.

*Idle power.* Varying the idle fraction from 0.15 to 0.40 changed no PhaseFit decision in any scenario. It shifted the saving over colocated serving by at most 5.1 percentage points (scenario B, 24.2–32.9% against 29.3% at φ = 0.25). A higher idle fraction favours disaggregation, because colocated instances are less fully utilized.

*KV-transfer bandwidth.* Reducing the per-GPU network bandwidth from 50 GB/s to 6.25 GB/s (50 Gb/s) left all SLOs satisfied. The p99 TPOT rose from 30.9 to 32.9 ms in scenario A and from 21.5 to 45.3 ms in scenario C. Scenario C has the longest prompts and the shortest outputs, so the transfer time is amortized over few tokens. For DeepSeek-R1 the transfer is negligible at any bandwidth, because MLA caches only 68.6 KiB per token. KV transfer is therefore not a binding constraint at 100 Gb/s per GPU or more for the models studied.

*Load.* At load factors 10, 20 and 40 the selected configuration family (B200, FP4, TP1 in both phases) did not change; only the instance counts did. The saving over colocated serving decreased from 20.4% to 11.0% as load grew, because large colocated pools amortize prefill interference better.

*Model error.* We re-planned scenario A with each fitted parameter perturbed: compute efficiencies −20%, memory efficiency 0.85 → 0.75, per-layer overhead ×2 and expert overhead ×1.5. In every case PhaseFit selected the same configurations. Only instance counts changed: the compute perturbation raised energy by 24.7% and required 31 instead of 25 accelerators. Evaluated under each perturbed model, the nominal decision had zero regret. Errors of the size measured in Section 6.1 thus move absolute energy estimates but not deployment decisions.

**Table 11.** Sensitivity of PhaseFit decisions.

{{TABLE_SENS}}

*Cost instead of energy.* Operators who rent accelerators may optimize cost instead. Fig. 7 prices GPU-hours with the H200 as the unit and varies the B200 price ratio $r$, using the per-phase instance counts of scenarios A and C. Below a ratio of 2.5 (A) or 2.0 (C), all-B200 deployments are cheapest. Above 2.9 (A) or 3.4 (C), all-H200 deployments are cheapest. Between these values, B200 prefill with H200 decode is cheapest. The reverse pairing never wins, because B200's advantage over H200 is larger in prefill (2.9× and 3.4× fewer GPUs) than in decode (2.5× and 2.0×). Cost-optimal heterogeneity therefore exists, but only inside a price band whose bounds are the per-phase GPU-count ratios. PhaseFit computes these bounds directly from its per-phase results.

![Fig. 7. Rental cost (H200 GPU-hours per hour) of four deployment strategies as a function of the B200:H200 price ratio, for scenarios A and C. Shaded: band in which a heterogeneous deployment is cheapest.](../figs/fig_price.png)

## 6.5. Case study 1: adding scarce Blackwell GPUs to a Hopper fleet (RQ4)

An operator serves Llama-3.1-70B on Azure-Conv (scenario A load and SLO) with an H200 fleet using FP8 disaggregation. Its energy is 839 mJ per output token. The operator receives 8, 16 or 32 B200 GPUs. Fig. 8 and Table 12 compare where to put them.

![Fig. 8. Case study 1. (a) Energy per output token as a function of the number of added B200 GPUs for placement strategies. (b) Energy as a function of the share β of B200 GPUs assigned to prefill (best precision per point).](../figs/fig_brownfield.png)

Three findings stand out. First, for this prefill-dominated workload the new GPUs save the most energy in prefill. Each B200 FP4 instance replaces about three H200 FP8 prefill instances, whereas in decode four B200s replace ten H200s and are then exhausted. With 8 B200s, PhaseFit assigned six to prefill (29% of prompts) and two to decode (58% of requests), saving 17.4%. With 16 it put all of them in prefill (33.7%). With 32 it used only 25, enough to serve both phases entirely, and saved 50.1%, which essentially reproduces the greenfield all-B200 deployment. Second, precision matters as much as placement. The Splitwise-style rule (new GPUs to prefill at FP8) captured 9.8%, 19.0% and 31.7%, that is, only 56–63% of the attainable saving. Third, colocated serving with a mixed fleet was worse than the disaggregated H200-only baseline until 32 B200s were available, because its H200 share inherits all the inefficiencies of colocation.

**Table 12.** Brownfield case studies: energy saving relative to the H200-only fleet.

{{TABLE_BROWN}}

## 6.6. Case study 2: serving a reasoning model under MLPerf SLOs (RQ4)

DeepSeek-R1 on the reasoning workload (scenario E) reverses the energy split (Fig. 6). In greenfield, PhaseFit chose a single B200 FP4 DEP4 instance for prefill and two B200 FP4 DEP8 instances for decode, with batches of up to 3,335 sequences. This deployment met the MLPerf v5.1 server targets with p99 TTFT of 0.21 s and p99 TPOT of 35 ms, and saved 15.9% relative to colocated serving. Decode dominates because each request generates 3,873 tokens on average, and every step re-reads the expert weights touched by the batch and a KV cache that reaches 4.7k tokens per sequence on average. FP4 halves the weight traffic and frees HBM for the cache. Without it, the best deployment used 17.8% more energy (Table 8).

In the brownfield setting the preferred placement flips relative to case study 1 (Fig. 9, Table 12). New B200s saved 50.1–54.2% when placed in decode, against 3.2% in prefill. The Splitwise-style rule saved nothing (−0.2%): one B200 FP8 prefill instance needs eight GPUs to hold the model and replaces only a single H200 instance. With 32 B200s, PhaseFit served both phases and saved 57.4%. Placing new hardware by the phase that dominates the workload's energy, rather than by a fixed rule, is the main lesson of the two case studies.

![Fig. 9. Case study 2. Energy per request of DeepSeek-R1 on the reasoning workload (MLPerf v5.1 server SLOs) when 8–32 B200 GPUs are added to an H200 fleet.](../figs/fig_brownfield_E.png)
