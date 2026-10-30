import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def gf2_rank_dense(a):
    """Independent dense Boolean row elimination, no bit-packed scan reuse."""
    a = np.asarray(a, dtype=np.uint8).copy()
    row = 0
    for col in range(a.shape[1]):
        candidates = np.flatnonzero(a[row:, col])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        a[[row, pivot]] = a[[pivot, row]]
        others = np.flatnonzero(a[row+1:, col]) + row + 1
        a[others] ^= a[row]
        row += 1
        if row == a.shape[0]:
            break
    return row

def profile(g, order):
    return [gf2_rank_dense(g[np.ix_(order[:k], order[k:])]) for k in range(1, len(g)+1)]

def summarize(r):
    return {"profile":r,"maxrank":max(r),"sum_states":sum(1<<v for v in r),
            "two_buffers_complex64_GiB":2*8*(1<<max(r))/(1<<30)}

data = np.load(ROOT/'rgs_graph.npz')
g = data['graph']
u = data['measurement_unitaries']
fixed = [q for q in range(len(g)) if all(min(abs(u[q,b,0]),abs(u[q,b,1]))<1e-12 for b in (0,1))]
assert not fixed
baseline = np.load(ROOT/'scan_order.npy').astype(int).tolist()
assert np.array_equal(g,g.T) and not np.diag(g).any()
if sys.argv[1] == 'prepare':
    np.savetxt(HERE/'input.txt',np.concatenate([g.astype(int).reshape(-1),baseline]),fmt='%d')
    manifest={"graph_sha256":digest(ROOT/'rgs_graph.npz'),"baseline_order_sha256":digest(ROOT/'scan_order.npy'),
              "scope":"CPU plan optimization only, no GPU speed measurements","seed":20261030,
              "moves":["adjacent_swap","insert","segment_reverse","swap"],"maxrank_constraint":30,
              "search_budget_seconds":155,"baseline":summarize(profile(g,baseline))}
    (HERE/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['baseline']))
else:
    manifest=json.loads((HERE/'input_manifest.json').read_text())
    assert digest(ROOT/'rgs_graph.npz')==manifest['graph_sha256']
    assert digest(ROOT/'scan_order.npy')==manifest['baseline_order_sha256']
    best=list(map(int,(HERE/'best_order.txt').read_text().split()))
    assert sorted(best)==list(range(len(g)))
    rb=profile(g,baseline); r=profile(g,best)
    spec=importlib.util.spec_from_file_location('independent_scan_plan',ROOT/'upstream_scan/scan_plan.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    edges=[tuple(map(int,edge)) for edge in np.argwhere(np.triu(g,1))]
    _,scanr=module.make_plan(len(g),edges,best)
    assert r==scanr
    assert profile(g,best[::-1])==list(reversed([0]+r[:-1]))
    search=json.loads((HERE/'search_summary.json').read_text())
    assert search['baseline_cost']==sum(1<<v for v in rb)
    assert search['best_cost']==sum(1<<v for v in r)
    assert max(r)<=30
    np.save(HERE/'scan_order_optimized.npy',np.array(best,dtype=np.int64))
    before=summarize(rb);after=summarize(r)
    result={"baseline":before,"optimized":after,"optimized_order":best,
            "plan_cost_reduction_fraction":1-after['sum_states']/before['sum_states'],
            "plan_cost_ratio_baseline_over_optimized":before['sum_states']/after['sum_states'],
            "dense_gf2_all_cuts_match_upstream_plan":True,"frozen_inputs_unchanged":True,
            "structurally_fixed_vertices":fixed,"buffer_scope":"two full-sized main environment buffers only; excludes lookup tables, scratch and allocator overhead",
            "optimized_order_sha256":digest(HERE/'scan_order_optimized.npy'),
            "search":search,"scope":"Heuristic plan cost reduction; NOT a GPU runtime speedup; not an optimality proof"}
    (HERE/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['baseline','optimized','optimized_order']},indent=2))
