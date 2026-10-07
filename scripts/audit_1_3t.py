#!/usr/bin/env python3
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.parameter_audit import audit_1_3t, storage_bytes, int4_block_storage_bytes
TARGET = 1300000000000

def main():
    a = audit_1_3t()
    assert a.total_parameters == TARGET, f"parameter mismatch: {a.total_parameters:,} != {TARGET:,}"
    print("SK AI 1.3T ARCHITECTURE AUDIT")
    print(f"total parameters : {a.total_parameters:,}")
    print(f"active parameters: {a.active_parameters:,}")
    print("experts          : 8")
    print("active experts   : 2")
    print("layers           : 59")
    print("hidden size      : 16,384")
    print("attention        : 128 heads / 8 KV heads")
    print("vocab            : 133,120")
    for name, value in [
        ("FP32 weights", storage_bytes(TARGET, 32)),
        ("BF16 weights", storage_bytes(TARGET, 16)),
        ("INT8 weights", storage_bytes(TARGET, 8)),
        ("INT4+FP16 scales", int4_block_storage_bytes(TARGET, 128, 2)),
    ]:
        print(f"{name:18s}: {value:,} bytes")
    q4 = int4_block_storage_bytes(TARGET, 128, 2)
    print(f"INT4+scales GiB  : {q4 / 2**30:.3f}")
    print(f"INT4+scales TiB  : {q4 / 2**40:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
