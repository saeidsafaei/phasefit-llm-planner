"""Model architectures (from the public Hugging Face config.json files).

Derived quantities:
  params_total, params_active   : parameter counts
  dense_params_per_gpu(...)     : weights resident on one GPU for a parallel plan
  kv_elems_per_token            : KV-cache elements stored per token (all layers)
  lin_flops_per_token           : 2 x active linear parameters (incl. LM head)
  attn_flops_decode(ctx)        : attention FLOPs for one decode token at context ctx
  attn_flops_prefill(s)         : attention FLOPs for a whole causal prompt of length s
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str
    L: int
    d: int
    n_q: int
    n_kv: int
    hd: int
    ffn: int
    vocab: int
    tied: bool = False
    # MoE
    n_exp: int = 0
    top_k: int = 0
    d_exp: int = 0
    n_shared: int = 0
    first_dense: int = 0
    # MLA (DeepSeek)
    mla: bool = False
    q_lora: int = 0
    kv_lora: int = 0
    rope: int = 0
    nope: int = 0
    v_hd: int = 0

    # ---------------- parameter accounting ----------------
    def attn_params(self) -> int:
        if self.mla:
            qk = self.nope + self.rope
            return (self.d * self.q_lora + self.q_lora * self.n_q * qk
                    + self.d * (self.kv_lora + self.rope)
                    + self.kv_lora * self.n_q * (self.nope + self.v_hd)
                    + self.n_q * self.v_hd * self.d)
        return (self.d * self.n_q * self.hd + 2 * self.d * self.n_kv * self.hd
                + self.n_q * self.hd * self.d)

    @property
    def moe(self) -> bool:
        return self.n_exp > 0

    @property
    def n_moe_layers(self) -> int:
        return self.L - self.first_dense if self.moe else 0

    def dense_ffn_params(self) -> int:
        return 3 * self.d * self.ffn

    def expert_params(self) -> int:
        return 3 * self.d * self.d_exp

    def embed_params(self) -> int:
        return self.vocab * self.d

    @property
    def params_total(self) -> float:
        p = self.L * self.attn_params()
        if self.moe:
            p += self.first_dense * self.dense_ffn_params()
            p += self.n_moe_layers * ((self.n_exp + self.n_shared) * self.expert_params()
                                      + self.d * self.n_exp)
        else:
            p += self.L * self.dense_ffn_params()
        p += self.embed_params() * (1 if self.tied else 2)
        return p

    @property
    def params_active(self) -> float:
        """Parameters touched by one token (embedding lookup excluded, LM head included)."""
        p = self.L * self.attn_params()
        if self.moe:
            p += self.first_dense * self.dense_ffn_params()
            p += self.n_moe_layers * (self.top_k + self.n_shared) * self.expert_params()
        else:
            p += self.L * self.dense_ffn_params()
        return p + self.embed_params()

    def replicated_params(self) -> float:
        """Non-expert parameters (attention, dense FFN, shared experts, embeddings)."""
        p = self.L * self.attn_params() + self.embed_params() * (1 if self.tied else 2)
        if self.moe:
            p += self.first_dense * self.dense_ffn_params()
            p += self.n_moe_layers * (self.n_shared * self.expert_params() + self.d * self.n_exp)
        else:
            p += self.L * self.dense_ffn_params()
        return p

    def routed_params(self) -> float:
        return self.n_moe_layers * self.n_exp * self.expert_params() if self.moe else 0.0

    # ---------------- KV cache ----------------
    @property
    def kv_elems_per_token(self) -> int:
        if self.mla:
            return self.L * (self.kv_lora + self.rope)
        return self.L * 2 * self.n_kv * self.hd

    # ---------------- FLOPs ----------------
    @property
    def lin_flops_per_token(self) -> float:
        return 2.0 * self.params_active

    def attn_flops_decode(self, ctx: float) -> float:
        if self.mla:  # weight-absorbed MLA: scores over (kv_lora+rope), output over kv_lora
            return 2.0 * self.L * self.n_q * (self.kv_lora + self.rope + self.kv_lora) * ctx
        return 4.0 * self.L * self.n_q * self.hd * ctx

    def attn_flops_prefill(self, s: float) -> float:
        """Causal attention for a prompt of s tokens (s^2/2 query-key pairs)."""
        if self.mla:  # non-absorbed form during prefill
            return 2.0 * self.L * self.n_q * (self.nope + self.rope + self.v_hd) * s * s / 2.0
        return 4.0 * self.L * self.n_q * self.hd * s * s / 2.0


MODELS = {
    "Llama-3.1-8B": Model("Llama-3.1-8B", L=32, d=4096, n_q=32, n_kv=8, hd=128,
                          ffn=14336, vocab=128256),
    "Llama-3.1-70B": Model("Llama-3.1-70B", L=80, d=8192, n_q=64, n_kv=8, hd=128,
                           ffn=28672, vocab=128256),
    "Qwen3-235B-A22B": Model("Qwen3-235B-A22B", L=94, d=4096, n_q=64, n_kv=4, hd=128,
                             ffn=12288, vocab=151936, n_exp=128, top_k=8, d_exp=1536,
                             n_shared=0, first_dense=0),
    "DeepSeek-R1": Model("DeepSeek-R1", L=61, d=7168, n_q=128, n_kv=128, hd=128,
                         ffn=18432, vocab=129280, n_exp=256, top_k=8, d_exp=2048,
                         n_shared=1, first_dense=3, mla=True, q_lora=1536, kv_lora=512,
                         rope=64, nope=128, v_hd=128),
}
# Llama-3.3-70B shares the Llama-3.1-70B architecture
MODELS["Llama-3.3-70B"] = MODELS["Llama-3.1-70B"]


if __name__ == "__main__":
    for k, m in MODELS.items():
        print(f"{k:18s} total={m.params_total/1e9:7.1f}B active={m.params_active/1e9:6.1f}B "
              f"KV/token={m.kv_elems_per_token} elems "
              f"({m.kv_elems_per_token*2/1024:.1f} KiB @BF16)")
