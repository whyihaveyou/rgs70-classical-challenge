#!/usr/bin/env bash
set -euo pipefail
born_group=$1
case "$born_group" in 0|1) born_count=432;; 2|3) born_count=431;; *) echo "bad group" >&2; exit 2;; esac
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_PATH="$PWD/deps/nvidia/cuda_runtime"
export LD_LIBRARY_PATH="$PWD/deps/nvidia/cuda_nvrtc/lib:$PWD/deps/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export CUPY_ACCELERATORS=""
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUPY_CACHE_DIR="$PWD/born2_kernel_cache_${born_group}"
date +%s > "born2_group${born_group}_start_unix.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv > "born2_group${born_group}_hardware.txt"
python3 -u born_production.py --manifest born_production_manifest_seed2.json --group "$born_group" --out "born2_group${born_group}.jsonl"
python3 -u scan_oracle.py --device 0 --fp64 --reverse-order --samples-jsonl "born2_group${born_group}.jsonl" --count "$born_count" --out "born2_group${born_group}_fp64.jsonl"
date +%s > "born2_group${born_group}_end_unix.txt"
