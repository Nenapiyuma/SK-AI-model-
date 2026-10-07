"""Distributed training entrypoint scaffold for the 1.3T MoE model.

This file validates launch-time requirements only. It does not pretend to train a
1.3T model on a single workstation.
"""
import os


def distributed_environment():
    return {
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "master_addr": os.environ.get("MASTER_ADDR", "127.0.0.1"),
        "master_port": int(os.environ.get("MASTER_PORT", "29500")),
    }


def validate_launch():
    env = distributed_environment()
    if env["world_size"] < 2:
        raise RuntimeError(
            "1.3T training requires distributed execution. "
            "Use torchrun with multiple processes/nodes; single-process training is unsupported."
        )
    return env


if __name__ == "__main__":
    print(validate_launch())
