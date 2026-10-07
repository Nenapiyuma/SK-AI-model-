# SK AI

This repository is being developed from the uploaded SK AI project toward a verified 1.3 trillion-parameter production architecture.

## Current verified result

The uploaded implementation is **not** a 1.3T model. Its existing configurations count to approximately:

- Mini: 1,311,872 parameters
- Small: 1,213,302,784 parameters
- Medium: 6,195,253,248 parameters
- Large: 56,863,236,096 parameters

A new exact 1.3T MoE architecture specification is now tracked in configs/sk_1_3t_moe.json.

Run:

    python scripts/audit_1_3t.py

The audit must report exactly 1,300,000,000,000 parameters.

## 1.3T architecture

59 decoder layers, hidden size 16,384, 128 attention heads, 8 KV heads, 8 MoE experts with top-2 routing, 54,490 SwiGLU intermediate size per expert, 133,120 vocabulary, 131,072 context, tied embeddings and a trainable output bias.

Training/checkpoint precision is BF16. Inference is targeted at genuine INT4 block-128 weights with FP16 scales.

## Storage

Raw 1.3T weights are approximately 2.6 TB in BF16, 1.3 TB in INT8 and 650 GB in INT4 before quantization metadata. INT4 plus one FP16 scale per 128 parameters is approximately 670.313 GB before container/index overhead.

No fake padding, empty tensors or duplicate weights are used.

## Important status

The architecture specification is verified, but a 1.3T checkpoint has **not** been generated or trained in this development environment. Large-scale training requires a distributed GPU cluster and TB-scale external artifact storage.

GitHub stores source, configs, tests and CI. Model checkpoints will be released separately as sharded artifacts with SHA-256 manifests.
