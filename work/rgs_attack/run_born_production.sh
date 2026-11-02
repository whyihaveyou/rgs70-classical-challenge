#!/usr/bin/env bash
set -euo pipefail
born_group=$1
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_PATH="$PWD/deps/nvidia/cuda_runtime"
export LD_LIBRARY_PATH="$PWD/deps/nvidia/cuda_nvrtc/lib:$PWD/deps/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export CUPY_ACCELERATORS=""
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUPY_CACHE_DIR="$PWD/born_kernel_cache_${born_group}"
date +%s > "born_group${born_group}_start_unix.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv > "born_group${born_group}_hardware.txt"
python3 -u born_production.py --group "$born_group" --out "born_group${born_group}.jsonl"
python3 -u scan_oracle.py --device 0 --fp64 --reverse-order --samples-jsonl "born_group${born_group}.jsonl" --count 432 --out "born_group${born_group}_fp64.jsonl"
date +%s > "born_group${born_group}_end_unix.txt"
