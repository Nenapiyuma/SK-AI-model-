"""Reference PyTorch module graph for SK AI 1.3T MoE.

Use device="meta" for architecture validation. Real training requires a distributed
launcher and sharded parameter materialization; this module deliberately does not
allocate 1.3T weights on a developer workstation.
"""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class MoE13TConfig:
    vocab_size: int = 133120
    hidden_size: int = 16384
    layers: int = 59
    attention_heads: int = 128
    kv_heads: int = 8
    intermediate_size: int = 54490
    experts: int = 8
    active_experts: int = 2


class Expert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, device=None, dtype=None):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)


class MoELayer(nn.Module):
    def __init__(self, cfg: MoE13TConfig, device=None, dtype=None):
        super().__init__()
        self.router = nn.Linear(cfg.hidden_size, cfg.experts, bias=False, device=device, dtype=dtype)
        self.experts = nn.ModuleList([
            Expert(cfg.hidden_size, cfg.intermediate_size, device=device, dtype=dtype)
            for _ in range(cfg.experts)
        ])


class SK13TMetaGraph(nn.Module):
    """Architecture graph only. It is intentionally meta-device safe."""

    def __init__(self, cfg: MoE13TConfig = MoE13TConfig(), device="meta", dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([MoELayer(cfg, device=device, dtype=dtype) for _ in range(cfg.layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.output_bias = nn.Parameter(torch.zeros(cfg.vocab_size, device=device, dtype=dtype))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def expected_parameter_count(self):
        h = self.cfg.hidden_size
        hd = h // self.cfg.attention_heads
        attn = (
            h * (self.cfg.attention_heads * hd)
            + 2 * h * (self.cfg.kv_heads * hd)
            + (self.cfg.attention_heads * hd) * h
        )
        # The graph above only contains MoE/embedding/norm/bias. This method is
        # intentionally separate from the full Transformer equation used by the audit.
        moe = self.cfg.layers * (
            h * self.cfg.experts
            + 3 * h * self.cfg.intermediate_size * self.cfg.experts
        )
        return self.cfg.vocab_size * h + moe + h + self.cfg.vocab_size
