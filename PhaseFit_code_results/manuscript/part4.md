# 7. Discussion

## 7.1. Guidelines for system architects

The results translate into six guidelines for operators who plan LLM serving on the 2024–2026 hardware generation.

**G1. Disaggregate first.** Separating prefill from decode was the largest single source of savings in every scenario ({{G_DISAGG}} less energy than the best colocated deployment), and the savings grow as TPOT targets tighten and for MoE models. Colocated serving pays twice. Prefill chunks inflate decode step times, which forces smaller decode batches. The resulting pools must then be sized for the slower of the two phases.

**G2. Treat numeric precision as a hardware decision.** Allowing FP4 on accelerators that execute it natively reduced energy by a further {{G_FP4}} relative to the best FP8 deployment. For capacity-bound MoE models, FP4 also changes the feasible parallelism: it lets B200 hold DeepSeek-R1 in four GPUs rather than eight. Precision should be chosen per phase together with the accelerator, subject to the accuracy evidence available for the model.

**G3. Do not expect phase-specialized hardware to pay off in a greenfield cluster.** With FP4 allowed, the energy-optimal accelerator was the same for both phases in all six scenarios, and on restricted fleets heterogeneous pairs saved at most about 1%. The reason is structural (Table 9). The newest generation leads in both joules per FLOP, which governs prefill, and joules per byte moved, which governs decode. Phase-level heterogeneity pays only when these two rankings disagree, as for MI355X and MI300X without FP4, or when instance granularity differs (Section 6.2).

**G4. In brownfield fleets, place scarce new accelerators by marginal energy, not by intuition.** For the conversational workload, new B200s saved the most energy in prefill until the prefill phase was saturated, and only then in decode. For the reasoning workload the order {{G_BROWN_E}}. Using FP4 on the new devices mattered as much as their placement. A Splitwise-style heuristic that sends new GPUs to prefill at FP8 captured only {{G_SPLIT_SHARE}} of the attainable savings at the smallest budget.

**G5. Size prefill pools for the tail of the prompt distribution.** Under tight TTFT targets and long prompts (Azure-Code), prefill pools ran at {{G_UTIL_LOW}} utilization because the p99 prompt must be served almost immediately. Higher tensor parallelism shortens single-prompt latency and was selected automatically in these cases. Idle power then becomes a first-order term, which makes the idle fraction $\phi$ of the chosen hardware relevant to procurement.

**G6. Price ratios, not peak specifications, decide cost-optimal heterogeneity.** In rental-cost terms a mixed B200/H200 deployment is optimal only within a band of price ratios bounded by the per-phase GPU-count ratios (Fig. 7). Outside that band, one type should serve both phases. Operators can compute this band from per-phase instance counts with the method of Section 6.4.

## 7.2. Threats to validity

*Analytical model rather than measurement.* All results come from a calibrated model and simulator, not from running the deployments on hardware. We mitigate this risk in four ways: the model is validated on 32 held-out measurements, including a different vendor, software stack and model family (Section 6.1); decisions are re-checked under ±12–100% perturbations of the fitted parameters (Section 6.4); conclusions are stated as relative comparisons between deployments evaluated with the same model; and every selected deployment is re-simulated end-to-end. Absolute energies should nevertheless be read as estimates. The held-out error is about 19% on average and reaches 33–45% in the regimes discussed in Section 6.1.

*Software maturity.* The model is calibrated on TensorRT-LLM, and the held-out MI300X/vLLM data show larger errors at high batch. Applying TensorRT-LLM-level efficiency to AMD and Intel devices therefore assumes comparable software maturity across vendors. Where a vendor's stack is less mature, its accelerators would be chosen less often than reported here. The structural argument of G3 does not depend on this assumption.

*Power model.* Eq. (11) uses board power as the busy ceiling and a fixed idle fraction. Real devices draw less than TDP in memory-bound phases, and power capping or frequency scaling [@patel2024power; @stojkovic2025] can shift the trade-offs. We report results for $\phi \in [0.15, 0.40]$ and exclude host, network and cooling energy, which would add a roughly proportional overhead to every deployment.

*Admission with known output lengths.* The simulator admits a request to a decode instance by reserving its actual output length, which an engine can only approximate with output-length prediction or eviction. Reserving the maximum length instead would lower decode batch sizes for workloads with long-tailed outputs, most of all for the reasoning workload, and would raise decode energy for every policy alike.

*Workloads.* The Azure traces are from November 2023 and contain no content, so effects that depend on content, such as prefix caching [@zheng2024] or speculative decoding [@leviathan2023], are not modelled. The reasoning workload has synthetic lengths whose means, but not dispersion, match MLPerf. Time compression preserves burst structure but not the absolute duration of bursts.

*Accuracy of low-precision formats.* We treat FP4 as admissible on devices that execute it natively, based on MLPerf closed-division results. For other models or tasks the accuracy of FP4 must be re-established. The FP4-disallowed variant of every experiment brackets this uncertainty.

*Scope.* We consider single-model deployments, static pools and one KV transfer path. Autoscaling over diurnal load, multi-model consolidation, pipeline parallelism across nodes and expert parallelism beyond eight devices (as in large-scale DeepSeek-V3 serving [@dsv3]) are left for future work.

# 8. Conclusion

This paper asked how the accelerators, numeric precisions and parallelism of the 2024–2026 hardware generation should be assigned to the prefill and decode phases of LLM serving under latency SLOs. We built PhaseFit, a planner that rests on a phase-aware, cross-vendor model calibrated on public measurements, on a decomposition of the disaggregated planning problem into two per-phase problems, and on trace-driven sizing. Its extension to brownfield fleets places scarce new accelerators in mixed pools.

On real Azure traces and an MLPerf-matched reasoning workload, across three model families and seven accelerators, four findings stand out. First, disaggregation reduced accelerator energy by {{C_DISAGG}} relative to the best colocated deployment. Second, native FP4 reduced it by a further {{C_FP4}}. Third, in greenfield clusters the energy-optimal deployment used the same accelerator type for both phases in every scenario with FP4 allowed, because the newest parts lead in both joules per FLOP and joules per byte. Fourth, in brownfield fleets the placement and precision of scarce new GPUs matter: PhaseFit saved {{C_BROWN}} with 8–32 added B200 GPUs, against {{C_SPLIT}} for a Splitwise-style heuristic.

Future work will validate the planner on physical deployments, add autoscaling over diurnal load and multi-model consolidation, model power capping and frequency scaling explicitly, and extend the model to rack-scale systems with large expert-parallel domains and to emerging processing-in-memory designs [@park2024attacc; @heo2024].

# Declarations

**CRediT authorship contribution statement.** {{AUTHOR_1}}: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft. {{AUTHOR_2}}: Conceptualization, Methodology, Validation, Supervision, Writing – review & editing.

**Declaration of competing interest.** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Funding.** This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

**Data availability.** The Azure LLM inference traces are publicly available from the Azure Public Dataset [@azure2023]. All calibration measurements are taken from the public sources cited in Section 4.2. The model, simulator, planner and experiment scripts will be made available by the authors upon reasonable request [*authors: replace with a repository link if one is created*].

**Declaration of generative AI and AI-assisted technologies in the writing process.** During the preparation of this work the authors used Claude (Anthropic) to assist with literature search, code development for the simulation study, and drafting and editing of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

# References

{{REFERENCES}}
