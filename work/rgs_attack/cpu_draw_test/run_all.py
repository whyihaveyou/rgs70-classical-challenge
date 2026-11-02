"""Statistical test of the production draw routine against exact conditional Born distributions.

The unmodified production selection code (DeferredGlobalMixin.draw -> DistributedSampler.draw) runs on CPU,
with the two GPU shards replaced by NumPy arrays that hold the exact complex64 conditional state of an
induced subgraph. Each trial uses a fresh uniform B outcome. For every trial we report a chi-square test over
64 equal-mass groups and the deviation of the empirical conditional XEB from its exact value. A float64
inverse-CDF sampler on the same distributions serves as the reference."""
import json, math, os, sys, time, importlib.util, numpy as np
from scipy import stats
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here); sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location('harness', os.path.join(here, 'draw_harness.py'))
harness = importlib.util.module_from_spec(spec); spec.loader.exec_module(harness)

def grouped_chi2(idx, p, G=64):
    order = np.argsort(p, kind='stable'); cm = np.cumsum(p[order]); grp = np.empty(len(p), np.int64)
    grp[order] = np.minimum((cm * G).astype(np.int64), G - 1)
    e = np.bincount(grp, weights=p, minlength=G) * len(idx); c = np.bincount(grp[idx], minlength=G)
    return float(stats.chi2.sf(((c - e) ** 2 / e).sum(), G - 1))

def run(fixture, tilebits, trials, draws, seed):
    fx = np.load(os.path.join(root, fixture)); g = fx['graph']; u = fx['measurement_unitaries']
    A = fx['A'].tolist(); B = fx['B'].tolist(); m = len(A); rng = np.random.default_rng(seed)
    res = {'fixture': fixture, 'conditional_qubits': m, 'tilebits': tilebits, 'trials': trials, 'draws_per_trial': draws, 'seed': seed}
    for name in ('production', 'reference'): res[name] = {'chi2_p': [], 'xeb_diff': [], 'xeb_se': []}
    for _ in range(trials):
        ob = rng.integers(0, 2, len(B)).tolist(); raw, final = harness.cpu_deferred(g, u, A, B, ob)
        p = np.abs(final) ** 2; p /= math.fsum(p); exact = (1 << m) * (p ** 2).sum() - 1
        eng = harness.CPUEngine(raw, u[A[-1]], tilebits)
        prod = np.fromiter((eng.draw(rng)[0] for _ in range(draws)), np.int64, draws)
        cdf = np.cumsum(p); ref = np.searchsorted(cdf, rng.random(draws) * cdf[-1], side='right')
        for name, idx in (('production', prod), ('reference', ref)):
            z = (1 << m) * p[idx]; r = res[name]
            r['chi2_p'].append(grouped_chi2(idx, p)); r['xeb_diff'].append(float(z.mean() - 1 - exact)); r['xeb_se'].append(float(z.std(ddof=1) / np.sqrt(draws)))
    for name in ('production', 'reference'):
        r = res[name]; d = np.array(r['xeb_diff']); s = np.array(r['xeb_se'])
        r['pooled_xeb_bias'] = float(d.mean()); r['pooled_xeb_bias_se'] = float(np.sqrt((s ** 2).sum()) / len(s))
        r['chi2_p_uniformity_ks_p'] = float(stats.kstest(r['chi2_p'], 'uniform').pvalue)
    return res

if __name__ == '__main__':
    offset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_name = sys.argv[2] if len(sys.argv) > 2 else 'results.json'
    t0 = time.time()
    runs = [run('induced_24q.npz', 4, 20, 200000, 20261102 + offset), run('induced_24q.npz', 11, 20, 200000, 20261103 + offset),
            run('induced_40q.npz', 10, 10, 200000, 20261104 + offset)]
    d = np.concatenate([r['production']['xeb_diff'] for r in runs]); s = np.concatenate([r['production']['xeb_se'] for r in runs])
    summary = {'total_production_draws': int(sum(r['trials'] * r['draws_per_trial'] for r in runs)),
               'pooled_production_xeb_bias': float(d.mean()), 'pooled_production_xeb_bias_se': float(np.sqrt((s ** 2).sum()) / len(s)),
               'all_chi2_p_uniformity_ks_p': float(stats.kstest(np.concatenate([r['production']['chi2_p'] for r in runs]), 'uniform').pvalue),
               'all_chi2_p_fisher_combined_p': float(stats.chi2.sf(-2 * np.log(np.concatenate([r['production']['chi2_p'] for r in runs])).sum(), 2 * sum(r['trials'] for r in runs))),
               'reference_all_chi2_p_uniformity_ks_p': float(stats.kstest(np.concatenate([r['reference']['chi2_p'] for r in runs]), 'uniform').pvalue),
               'seconds': time.time() - t0}
    json.dump({'summary': summary, 'runs': runs}, open(os.path.join(here, out_name), 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    for r in runs: print(r['fixture'], r['tilebits'], {k: (round(v['pooled_xeb_bias'], 5), round(v['pooled_xeb_bias_se'], 5), round(v['chi2_p_uniformity_ks_p'], 3)) for k, v in r.items() if k in ('production', 'reference')})
