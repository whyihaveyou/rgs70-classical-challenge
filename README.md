# RGS70 capped sampling and numerical audit

Author: **Zhongpu Qiu (邱仲普)**. Affiliation: **Not specified**.

This repository contains completed classical capped-rejection experiments for the public 70-qubit random-graph-sampling instance in [QAT submission #151](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/issues/151). It contains probability data, all proposal/acceptance random variables, independent complex128 scores, a complete cap=1 decision audit, and a separate scan-order benchmark. No pending Born-sampling prototype is included. QAT acceptance or supersession is not asserted.

| Completed experiment | GPUs | Samples / proposals | Independent XEB ± standard error | Generation wall time |
|---|---:|---:|---:|---:|
| cap=1, seed 314159 | 1 H200 | 1726 / 2769 | 0.3899245 ± 0.0247908 | 256.2344 s |
| cap=8, seeds 271828 and 271829 | 2 H200 | 1726 / 13673 | 1.0710865 ± 0.0352906 | 666.1743 s, maximum worker duration |

The recorded script start/end spans are **731 seconds for cap=1** and **782 seconds for cap=8**. These scopes differ: cap=1 includes independent scoring of its outputs, IBM samples and a uniform control, while cap=8 includes its generation and output scoring/validation. They are script spans on prepared inputs, **not raw-input end-to-end measurements**. The original 40-second CPU order search is additional; the later 158.094-second optimization and separate full-proposal audit are also separate experiments. See the preserved timestamps and per-stage records.

The QAT submission reports a 1950-second quantum runtime and an XEB challenge threshold of 0.092. The table is generation-only timing, not cold-start, raw-QASM-to-results, or fully verified end-to-end timing. Setup, scoring, full-audit and observed script timestamps are preserved separately. Queueing, installation, preprocessing search and earlier development are not included in the table. The cap=8 maximum worker duration is not a synchronized launcher measurement. Do not describe these numbers as certified end-to-end speedups or energy measurements.

Our independent score of the 1726 public IBM bitstrings is XEB 0.4660126 ± 0.0311013. The original submission's 0.37 is graph-state DFE, not its measured XEB. Cap=1 is not shown to outperform IBM on observed mean XEB: the classical-minus-IBM mean difference is -0.07609, with approximate 95% interval [-0.15404, 0.00187]. It does exceed the stated 0.092 threshold. Both classical runs have a fixed K=4 clipped-score Hoeffding lower bound above 0.092 (0.23350 and 0.81005, alpha=0.025 each; see recorded analyses and manifests).

## Algorithm and claim boundary

For each independent uniform 70-bit proposal x, compute p(x) and accept using a fresh independent uniform variate with probability min(2^70 p(x)/c,1). Stop at the fixed accepted count. In exact arithmetic, the resulting iid distribution is q_c(x) proportional to min(2^70 p(x),c). It is **not exact Born sampling**. The implementation uses complex64 probabilities, so its actual distribution uses the numerical probability approximation. Complex128 rescoring checks numerical consistency; it is not an exact-real error certificate. No Porter–Thomas assumption is used to generate or score these samples. XEB alone does not certify total-variation distance or quantum-state fidelity.

All accepted outputs in both runs are independently rescored with reversed scan order and complex128. For cap=1, all 2769 proposals, including 1043 rejected proposals, were additionally audited: zero acceptance decisions changed; maximum probability relative difference was 3.4801e-5 and minimum FP64 acceptance margin was 2.0459e-4. **Rejected cap=8 proposals have not all been recomputed in FP64**; their original probabilities, RNG values and decisions are provided and replayable.

A separate fixed-32-input alternating GPU benchmark changes only the contraction order: sum(2^rank) decreases from 9,982,443,517 to 7,919,632,381, maximum rank remains 30, and measured contraction time decreases by 19.20% (old/new ratio 1.23763). The CPU optimization consumed 158.094 seconds. This improvement is not retroactively applied to cap=1/cap=8 timings; amortization estimates are explicitly projections. Two complex64 main buffers require 16 GiB; FP64 requires 32 GiB, with additional workspace. The paired benchmark retains three engines concurrently and therefore needs substantially more memory (approximately 64 GiB plus overhead).

## Provenance and attribution

The scan compiler and CUDA kernel are adapted from **Kim Renaud's dcs-rankwidth**, copyright 2026 Kim Renaud (Calcul Québec), [upstream repository](https://github.com/kim-renaud/dcs-rankwidth), commit `2ddc1c94c31b922f0eb3118d0a8356cc92e6f1a7`. Apache-2.0 LICENSE and NOTICE are preserved. The original GPU kernel is not claimed as new work. Local changes include the NumPy indexing compatibility cast and a separately labelled complex128 kernel; see root NOTICE.

The exact target is the base 70-qubit ideal QASM, not the 97-qubit hardware error-detection circuit. QAT source commit: `b781c93aa44fd45a4e7cd42190532fd5ebcca564`. [Pinned QASM source](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/blob/b781c93aa44fd45a4e7cd42190532fd5ebcca564/data/classically-verifiable-problems/circuit-models/random_graph_sampling/nq70_depth70_checks27_basis_fstate.qasm). QASM SHA256: `8881167664ca6f75d11104b06bde2d6571c0b1558fdb67b55be69141de7e67d7`.

[Public IBM sample download](https://github.com/user-attachments/files/27220052/samples_nq70_depth70_checks27_basis_fstate.npy.zip) is included unchanged, together with the original NPY and q0-left conversion. Qiskit strings have q0 at the right: the conversion reverses each string. External QASM/sample inputs retain their original terms; the root code license does not relicense them.

`ORIGINAL_HASHES.json` records original hashes and byte-identical copies. `SHA256SUMS` covers release files. The audit manifest's transport-description field is sanitized; the public Stim wrapper writes to reproduced/ to preserve frozen files. Numerical probability records are unchanged. Some historical manifests mention omitted HPC runners or unrelated source hashes: those are historical provenance records, not instructions to fetch private files. `work/rgs_attack/graph_sampler.py` includes CPU-only reference conditional calculations used for validation, not the pending GPU Born implementation.

## Reproduce the analysis without a GPU

Run commands from the repository root, using Python 3.10 or newer:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_analysis.py
python work/rgs_attack/independent_stim_check.py
python work/rgs_attack/graph_sampler.py --out reproduced/cpu_checks
```

The first command verifies frozen probability-file hashes, replays all proposal RNG/acceptance decisions, reconstructs both cap analyses and the combined cap=8 score, and checks the stored full cap=1 FP64 audit. Results go to `reproduced/`, preserving frozen artifacts. Stim verifies all 70 canonical stabilizers of the original Clifford prefix against the graph reconstruction. CPU self-checks compare small complete statevectors. Analysis was validated with NumPy 2.2.6; original GPU dependency versions were not fully captured, so requirements-gpu.txt is a reproduction environment range rather than a complete historical lockfile.

## Reproduce on NVIDIA GPUs

Install a CUDA-compatible CuPy wheel and have a working CUDA toolkit/NVRTC available. The CUDA 12 example requirements are:

```bash
python -m pip install -r requirements-gpu.txt
cd work/rgs_attack
mkdir -p ../../reproduced/gpu
python scan_oracle.py --cpu --selftest --out ../../reproduced/gpu/cpu_scan_selftest.jsonl
python rejection_sampler.py --device 0 --count 1726 --seed 314159 --cap 1 --out ../../reproduced/gpu/cap1.jsonl
python scan_oracle.py --device 0 --fp64 --reverse-order --selftest --samples-jsonl ../../reproduced/gpu/cap1.jsonl.accepted.jsonl --count 1726 --out ../../reproduced/gpu/cap1_scored.jsonl
python audit_all_proposals_fp64.py --device 0 --proposals cap1_archive/capped1726.jsonl --accepted-scores cap1_archive/scored_fp64.jsonl --out ../../reproduced/gpu/cap1_full_audit.jsonl
python benchmark_scan_orders.py --device 0 --out ../../reproduced/gpu/scan_order_benchmark.json
```

For the cap=8 run, launch the following two commands simultaneously in separate terminals on two distinct GPUs; each generates exactly 863 accepted samples:

```bash
python rejection_sampler.py --device 0 --count 863 --seed 271828 --cap 8 --out ../../reproduced/gpu/cap8_part0.jsonl
python rejection_sampler.py --device 1 --count 863 --seed 271829 --cap 8 --out ../../reproduced/gpu/cap8_part1.jsonl
```

Score each accepted file with `scan_oracle.py --fp64 --reverse-order --count 863`, using its corresponding device and a distinct output filename. The generation uses the frozen original `scan_order.npy`; the optimized order is only used by the separate benchmark. Exact timings depend on hardware/software. Retain all outputs and rejected proposals; do not deduplicate or select runs by score.

## Artifact layout

- `work/rgs_attack/cap1_archive/`: complete cap=1 proposals, accepted samples, independent scores, IBM/control scores, analyses, hardware and timestamps.
- `work/rgs_attack/cap8_archive/`: both cap=8 proposal streams and scores, per-stream and combined analyses, hardware and timestamps.
- `work/rgs_attack/cap1_full_audit_artifact/`: complete cap=1 FP64 decision audit and paired scan-order benchmark.
- `work/rgs_attack/order_optimization/`: optimizer source, fixed output order, rank profiles, CPU search trace and timing; no machine-specific executable.
- `work/rgs_attack/upstream_scan/`: attributed compiler/kernels and original license/notice.
- `work/qat/data/.../`: original public QASM at the path expected by the independent validation script.

No private cluster runner, SSH configuration, raw cluster transport log, pending Born experiment or unconfirmed institutional affiliation is part of this release.
