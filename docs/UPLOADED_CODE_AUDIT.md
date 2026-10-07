# Uploaded Code Audit

Date: 2026-10-07

## Result

The uploaded archive was executed and inspected. The current implementation is functional as a small local demo, but it is **not** a production 1.3T model implementation.

## Parameter counts from the actual configuration equations

| Variant | Count |
|---|---:|
| Mini | 1,311,872 |
| Small | 1,213,302,784 |
| Medium | 6,195,253,248 |
| Large | 56,863,236,096 |

The repository documentation's claims of 2,756,736 for Mini, 1.3B, 7.2B and 70.4B do not match the tensor-shape equations in the code.

## Important implementation findings

1. The tensor backend uses Python array('f') float32 storage. The dtype field in configuration does not change tensor storage.
2. The checkpoint writer serializes embedding and projection/MLP matrices but omits RMSNorm weights. Therefore checkpoint byte size does not equal parameter_count × configured dtype bytes.
3. The quantization module measures reconstructed values but does not write a real INT8/INT4 checkpoint format.
4. The pretraining pipeline computes loss but contains no optimizer, backward pass or parameter update. It is therefore not a real training loop.
5. There is no implemented MoE layer in the inspected model.
6. FSDP/ZeRO is mentioned in the roadmap but is not implemented in the inspected training code.
7. The current inference generation path includes deterministic rule-based response text in addition to a tiny neural forward pass; it is not evidence of frontier-model capability.
8. The existing large configuration cannot be instantiated with the current pure-Python tensor backend at 1.3T scale.

## Actual installed artifact observed after the smoke test

The local Mini checkpoint was 5,243,072 bytes. Its audited parameter count is 1,311,872, which would require 5,247,488 raw FP32 weight bytes before a header. The approximately 4.6 KB discrepancy corresponds to omitted RMSNorm weights plus header accounting, confirming that the current serializer is incomplete for a faithful checkpoint.

## Smoke test actually run

The uploaded project smoke test completed all 16 listed subsystem checks successfully in the local environment.

Observed hardware during that run:

- CPU: 3 logical cores
- RAM: 5.81 GB
- Accelerator: CPU

Those results validate the demo/subsystem plumbing, not 1.3T training feasibility.

## Next engineering gates

The new 1.3T architecture is kept separate until these are implemented and tested:

- real PyTorch distributed MoE model
- exact tensor-shape parameter accounting against the instantiated module graph
- real BF16/FP16 checkpoint serialization
- sharded checkpoint index and SHA-256 manifest
- actual INT8/INT4 serialization and dequantization tests
- FSDP/ZeRO-3 plus expert/tensor/pipeline parallel configuration
- real optimizer/backward training loop
- tokenizer and data pipeline integration
- distributed resume/checkpoint tests
- capability evaluation on Sinhala, English, mixed language, coding, mathematics and reasoning

No 1.3T checkpoint or training success is claimed at this stage.
