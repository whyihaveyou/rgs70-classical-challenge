"""CPU verification of the two exact-sampling batches for random_graph_sampling_nq70_depth70_checks27_basis_fstate.

Run from the repository root:  python born/verify.py
Checks every record against the frozen manifests and seed streams, then recomputes all reported statistics.
Writes reproduced/born_verify.json. Needs only NumPy.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work' / 'rgs_attack'
BORN = ROOT / 'born'
D = 2.0 ** 70
BATCHES = {
    1: dict(manifest='born_production_manifest.json', seeds='production_seed_manifest.json', data=BORN / 'batch1', prefix='born_group'),
    2: dict(manifest='born_production_manifest_seed2.json', seeds='production_seed_manifest_seed2.json', data=BORN / 'batch2', prefix='born2_group'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def same_state(state, reference):
    return all(int(state['state'][k]) == int(reference['state'][k]) for k in ('state', 'inc')) and \
        int(state['has_uint32']) == int(reference['has_uint32']) and int(state['uinteger']) == int(reference['uinteger'])


def statistics(z, rng_seed=424242, resamples=20000, clip=4.0, alpha=0.05):
    rng = np.random.default_rng(rng_seed)
    boot = np.array([z[rng.integers(0, len(z), len(z))].mean() - 1 for _ in range(resamples)])
    return dict(n=int(len(z)), xeb=float(z.mean() - 1), standard_error=float(z.std(ddof=1) / np.sqrt(len(z))),
                bootstrap_95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                hoeffding_lower_95=float(np.minimum(z, clip).mean() - 1 - clip * np.sqrt(np.log(1 / alpha) / (2 * len(z)))))


def main():
    graph = np.load(WORK / 'rgs_graph.npz')
    A, B = graph['A'].tolist(), graph['B'].tolist()
    failures, report, z_by_batch, seen = [], {}, {}, {}
    for bid, cfg in BATCHES.items():
        manifest = json.loads((WORK / cfg['manifest']).read_text())
        manifest_sha = sha256(WORK / cfg['manifest'])
        for name, digest in manifest['files'].items():
            if sha256(WORK / name) != digest:
                failures.append(f'batch {bid}: frozen file changed: {name}')
        seeds = json.loads((WORK / cfg['seeds']).read_text())
        z, rel, norm_err = [], [], []
        for group in seeds['groups']:
            g = group['group_id']
            records = jsonl(cfg['data'] / f"{cfg['prefix']}{g}.jsonl")
            meta, samples = records[0]['metadata'], [r for r in records if 'sample' in r]
            scores = [s for s in jsonl(cfg['data'] / f"{cfg['prefix']}{g}_fp64.jsonl") if 'index' in s]
            if meta['manifest_sha256'] != manifest_sha or meta['group']['cli_seed_decimal'] != group['cli_seed_decimal']:
                failures.append(f'batch {bid} group {g}: manifest or seed mismatch')
            if len(samples) != group['sample_count'] or len(scores) != group['sample_count']:
                failures.append(f'batch {bid} group {g}: count mismatch')
            rng = np.random.Generator(np.random.PCG64(int(group['cli_seed_decimal'])))
            if not same_state(rng.bit_generator.state, group['initial_pcg64_state_decimal128']):
                failures.append(f'batch {bid} group {g}: initial PCG64 state mismatch')
            for k, (rec, score) in enumerate(zip(samples, scores)):
                where = f'batch {bid} group {g} sample {k}'
                if not same_state(rng.bit_generator.state, rec['rngstate_before']):
                    failures.append(f'{where}: RNG state before')
                b_bits = rng.integers(0, 2, len(B)).tolist()
                uniforms = [float(rng.random()) for _ in range(3)]
                if b_bits != rec['B_bits'] or uniforms != rec['draw_uniforms'] or not same_state(rng.bit_generator.state, rec['rngstate_after']):
                    failures.append(f'{where}: RNG replay')
                bits = [0] * len(graph['graph'])
                for j, q in enumerate(A):
                    bits[q] = (rec['A_index'] >> j) & 1
                for j, q in enumerate(B):
                    bits[q] = rec['B_bits'][j]
                bitstring = ''.join(map(str, bits))
                if bitstring != rec['bitstring_q0_left'] or bitstring != score['bitstring_q0_left'] or score['index'] != k:
                    failures.append(f'{where}: bit ordering or score alignment')
                if rec['global_sample_id'] != group['global_sample_id_start'] + k:
                    failures.append(f'{where}: global sample id')
                seen.setdefault(bitstring, []).append(rec['global_sample_id'])
                z.append(D * score['probability'])
                rel.append(abs(rec['probability'] - score['probability']) / score['probability'])
                norm_err.append(abs(rec['conditional_norm'] - 1))
        z = np.array(z)
        z_by_batch[bid] = z
        report[f'batch{bid}'] = dict(production_manifest_sha256=manifest_sha, seed_manifest_sha256=sha256(WORK / cfg['seeds']),
                                     statistics=statistics(z), max_relative_probability_deviation=float(max(rel)),
                                     median_relative_probability_deviation=float(np.median(rel)), max_norm_error=float(max(norm_err)))
    pooled = np.concatenate([z_by_batch[1], z_by_batch[2]])
    report['pooled'] = dict(statistics=statistics(pooled))
    report['duplicate_bitstrings'] = int(sum(len(v) > 1 for v in seen.values()))
    ibm = np.array([D * s['probability'] for s in jsonl(WORK / 'cap1_archive' / 'ibm_scored_fp64.jsonl') if 'index' in s])
    report['ibm_hardware_samples'] = dict(n=int(len(ibm)), xeb=float(ibm.mean() - 1), standard_error=float(ibm.std(ddof=1) / np.sqrt(len(ibm))))
    uniform = np.array([D * r['probability_raw'] for path in ('cap1_archive/capped1726.jsonl', 'cap8_archive/cap8_part0.jsonl', 'cap8_archive/cap8_part1.jsonl')
                        for r in jsonl(WORK / path) if 'proposal' in r])
    report['uniform_bitstrings'] = dict(n=int(len(uniform)), mean_z=float(uniform.mean()),
                                        ideal_xeb_estimate=float((uniform ** 2).mean() - 1),
                                        ideal_xeb_estimate_standard_error=float((uniform ** 2).std(ddof=1) / np.sqrt(len(uniform))))
    timing = json.loads((BORN / 'batch1' / 'central_timing.json').read_text())
    report['batch1_end_to_end_seconds'] = timing['central_monotonic_submit_to_all_logs_returned_seconds']
    report['failures'] = failures
    out = ROOT / 'reproduced'
    out.mkdir(exist_ok=True)
    (out / 'born_verify.json').write_text(json.dumps(report, indent=1) + '\n')
    for key in ('batch1', 'batch2', 'pooled'):
        s = report[key]['statistics']
        print(f"{key:7s} n={s['n']}  XEB {s['xeb']:.4f} ± {s['standard_error']:.4f}  bootstrap 95% [{s['bootstrap_95'][0]:.4f}, {s['bootstrap_95'][1]:.4f}]"
              f"  Hoeffding 95% lower {s['hoeffding_lower_95']:.4f}")
    print(f"IBM hardware samples: XEB {report['ibm_hardware_samples']['xeb']:.4f} ± {report['ibm_hardware_samples']['standard_error']:.4f}")
    print(f"uniform bitstrings: n={report['uniform_bitstrings']['n']}  mean z {report['uniform_bitstrings']['mean_z']:.4f}"
          f"  E[z^2]-1 = {report['uniform_bitstrings']['ideal_xeb_estimate']:.4f} ± {report['uniform_bitstrings']['ideal_xeb_estimate_standard_error']:.4f}")
    print(f"duplicate bitstrings: {report['duplicate_bitstrings']}   batch-1 end-to-end: {report['batch1_end_to_end_seconds']:.1f} s")
    print('PASS' if not failures else f'FAIL ({len(failures)}): ' + '; '.join(failures[:10]))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
