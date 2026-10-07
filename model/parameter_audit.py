"""Independent parameter accounting for SK AI's 1.3T MoE architecture."""

from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ArchitectureAudit:
    total_parameters: int
    active_parameters: int
    attention_parameters_per_layer: int
    expert_parameters_per_layer: int
    router_parameters_per_layer: int
    embedding_parameters: int
    output_bias_parameters: int

def audit_1_3t(vocab_size=133120, hidden_size=16384, layers=59,
               attention_heads=128, kv_heads=8, intermediate_size=54490,
               experts=8, active_experts=2, tied_embeddings=True,
               output_bias=True):
    if hidden_size % attention_heads:
        raise ValueError("hidden_size must be divisible by attention_heads")
    if attention_heads % kv_heads:
        raise ValueError("attention_heads must be divisible by kv_heads")
    if not (1 <= active_experts <= experts):
        raise ValueError("active_experts must be in [1, experts]")
    head_dim = hidden_size // attention_heads
    attention = (
        hidden_size * (attention_heads * head_dim)
        + 2 * hidden_size * (kv_heads * head_dim)
        + (attention_heads * head_dim) * hidden_size
    )
    expert_mlp = 3 * hidden_size * intermediate_size * experts
    router = hidden_size * experts
    embedding = vocab_size * hidden_size
    bias = vocab_size if output_bias else 0
    final_norm = hidden_size
    total = embedding + layers * (attention + 2 * hidden_size + expert_mlp + router) + final_norm + bias
    active = embedding + layers * (
        attention + 2 * hidden_size
        + 3 * hidden_size * intermediate_size * active_experts
        + router
    ) + final_norm + bias
    return ArchitectureAudit(total, active, attention, expert_mlp, router, embedding, bias)

def storage_bytes(parameters, bits_per_parameter):
    return math.ceil(parameters * bits_per_parameter / 8.0)

def int4_block_storage_bytes(parameters, block_size=128, scale_bytes=2):
    return math.ceil(parameters / 2) + math.ceil(parameters / block_size) * scale_bytes
