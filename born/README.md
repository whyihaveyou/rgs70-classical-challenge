# Exact sampling of the RGS70 instance

Data and code for our exact-sampling submission on the random graph sampling instance of [QAT #151](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/issues/151): `random_graph_sampling_nq70_depth70_checks27_basis_fstate`, base 70-qubit QASM at QAT commit `b781c93aa44fd45a4e7cd42190532fd5ebcca564` (SHA-256 `8881167664ca6f75d11104b06bde2d6571c0b1558fdb67b55be69141de7e67d7`).

The sampler draws independent samples from the ideal output distribution. The circuit reduces to a 70-vertex graph state measured in a product basis; a balanced cut of full GF(2) rank makes the outcome on one half exactly uniform, and the other half is drawn from its $2^{35}$-amplitude conditional state, held in complex64 on two GPUs. Every sample was rescored in complex128 by an independent linear rank-width contraction.

| | Batch 1 (timed) | Batch 2 | Pooled |
|---|---:|---:|---:|
| Samples | 1,726 | 1,726 | 3,452 |
| Linear XEB | 1.096 ± 0.035 | 0.953 ± 0.033 | 1.025 ± 0.024 |
| Bootstrap 95% CI | [1.028, 1.167] | [0.889, 1.018] | [0.978, 1.073] |
| One-sided 95% lower bound (Hoeffding, $z$ clipped at 4) | 0.846 | 0.741 | 0.828 |
| Max. relative deviation, sampler vs. complex128 score | 3.3e-6 | 2.6e-6 | |
| GPUs | 8 × H200 | 8 × H200 | |
| Longest job, generation only | 1,052 s | 1,051 s | |
| End-to-end wall clock | 1,265 s | not recorded | |

IBM's 1,726 hardware samples, scored with the same amplitudes, give 0.466 ± 0.031. An ideal sampler has expected XEB $2^{70}\sum_x p(x)^2-1$, which is 1 for Porter–Thomas statistics; 16,442 uniformly random bitstrings (the proposals of our capped runs) give $E[z^2]-1 = 1.007 \pm 0.036$ with $z=2^{70}p$.

Batch 1 is the pre-declared run (manifests frozen on 2026-10-30). Its 1,726 records were produced twice from the same manifest, bit for bit identical; the second run supplied the 1,265 s, measured on one monotonic clock from job submission until all logs had returned (`batch1/central_timing.json`). Batch 2 was added to test whether batch 1, which lies 2.7 standard errors above 1, reflects a sampler bias: same code and inputs, new seeds frozen before submission (seed manifest created 2026-11-01T13:59:56Z, job submitted 14:20:24Z). Its launcher clock was not kept to the end, so only worker times are given.

## Verify on a CPU

```bash
python -m pip install -r requirements.txt
python born/verify.py
```

This takes a few seconds. It checks the code and input hashes in both production manifests, replays every PCG64 stream against the logged states (35 bits for the uniform half, then three uniforms per sample), rebuilds each 70-bit string from the logged indices, matches it to its complex128 score, and recomputes all statistics above, the IBM score and the uniform-bitstring estimate. Results go to `reproduced/born_verify.json`.

The draw routine can be tested separately (a few minutes, needs SciPy):

```bash
python work/rgs_attack/cpu_draw_test/run_all.py
```

It runs the unmodified production selection code on exact conditional distributions of 24- and 40-qubit induced subgraphs, with a float64 inverse-CDF sampler as reference. Two independent runs with 10⁷ draws each are stored in `results.json` and `results_replicate.json`: pooled XEB bias $(5.5 \pm 4.5)\times10^{-4}$ and $(1.6 \pm 4.4)\times10^{-4}$.

## Reproduce on GPUs

Each group needs two GPUs with at least 130 GiB of memory each (H200), CuPy 13 and NVRTC (`requirements-gpu.txt`).

```bash
cd work/rgs_attack
bash run_born_production.sh 0          # batch 1, group 0; groups 0-3 have 432, 432, 431, 431 samples
bash run_born_production_seed2.sh 0    # batch 2, group 0
```

Each command checks the manifest hashes, generates the group's samples (`born_group<g>.jsonl` or `born2_group<g>.jsonl`) and rescores them in complex128 (`*_fp64.jsonl`). The `deps` paths in the scripts pointed to a local wheel directory and are harmless where CuPy is installed. `run_born_archive*.sh` wrapped these commands on our cluster and printed the outputs into the job log. GPU checks on small instances: `python specialized_deferred_sampler.py --graph induced_40q.npz --devices 0,1 --validate --out validate40.jsonl` (our logs are in `specialized_pilot_artifact/`).

## Layout

- `batch1/`, `batch2/`: per-group production records (bitstring, logged indices, RNG states, uniforms, generation probability, timings), complex128 scores (`*_fp64.jsonl`), GPU model and driver, start and end timestamps, exit status. `batch1/` also holds the central timing ledger (`central_timing.json`, `events.jsonl`); `batch2/events.jsonl` is the partial launcher log. `analysis/` contains the summaries written during the runs.
- `pooled_summary.json`: two-batch statistics.
- `figures/`: figure and plotting script.
- `../work/rgs_attack/`: sampler (`born_production.py`, `specialized_deferred_sampler.py` and the modules it builds on), frozen production and seed manifests, graph and scan order, complex128 scorer (`scan_oracle.py`, `upstream_scan/`), small fixtures (`induced_*q.npz`), GPU pilot logs (`specialized_pilot_artifact/`) and the CPU draw test (`cpu_draw_test/`).

Bitstrings are stored with qubit 0 first; IBM's samples (Qiskit order) were reversed before scoring.

## Attribution

The complex128 scorer uses the scan compiler and CUDA kernel of Kim Renaud's [dcs-rankwidth](https://github.com/kim-renaud/dcs-rankwidth) (Apache-2.0, commit `2ddc1c94c31b922f0eb3118d0a8356cc92e6f1a7`); see the root `NOTICE`. The QASM file and IBM's samples come from the QAT repository and #151.
