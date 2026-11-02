"""Independent CPU audit of the fixed 1726-output, four-group Born production.

This does not regenerate quantum probabilities or certify full-distribution TV.
It verifies frozen inputs, RNG replay, mapping, numerical scores and timing scope.
Run --self-test for temporary, explicitly synthetic failure-detection fixtures.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np

COUNTS = (432, 432, 431, 431)
BOOTSTRAPS = 20000
BOOTSTRAP_SEED = 424242


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_decimal(state):
    result = copy.deepcopy(state)
    for key in ('state', 'inc'):
        result['state'][key] = str(int(result['state'][key]))
    return result


def distribution_summary(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {'n': 0}
    return {'n': len(x), 'max': float(np.max(x)), 'p99': float(np.quantile(x, .99)),
            'median': float(np.median(x))}


def gf2_rank(matrix):
    matrix=np.asarray(matrix,dtype=np.uint8).copy()
    row=0
    for col in range(matrix.shape[1]):
        candidates=np.flatnonzero(matrix[row:,col])
        if not len(candidates):continue
        pivot=row+int(candidates[0]);matrix[[row,pivot]]=matrix[[pivot,row]]
        others=np.flatnonzero(matrix[row+1:,col])+row+1
        matrix[others]^=matrix[row];row+=1
        if row==matrix.shape[0]:break
    return row


def statistics(values, resamples=BOOTSTRAPS, primary=False):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {'n': 0}
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    bootstrap = []
    for start in range(0, resamples, 128):
        indices = rng.integers(0, len(x), size=(min(128, resamples-start), len(x)))
        bootstrap.extend(x[indices].mean(axis=1).tolist())
    mean = float(x.mean())
    se = float(x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else None
    clipped = float(np.minimum(x+1, 4).mean()-1)
    lower = clipped-4*math.sqrt(math.log(20)/(2*len(x)))
    return {'n': len(x), 'mean': mean, 'standard_error': se,
            'normal_95_CI_approximate': [mean-1.96*se, mean+1.96*se] if se is not None else None,
            'bootstrap_95_CI_approximate': np.quantile(bootstrap, [.025, .975]).tolist(),
            'bootstrap_resamples': resamples, 'bootstrap_seed': BOOTSTRAP_SEED,
            'min': float(x.min()), 'max': float(x.max()),
            'clipped_K4_mean': clipped, 'clipped_K4_one_sided_Hoeffding_lower_alpha_0_05': lower,
            'primary_predeclared_test': primary,
            'Hoeffding_scope': ('Single pooled production-run test; independent observations and valid target '
                               'probabilities required. Numerical probability error is not formally bounded.'
                               if primary else 'Per-group diagnostic only; no additional success claim or multiplicity-adjusted test.'),
            'primary_lower_above_QAT_0_092': bool(lower>.092) if primary else None}


def analyze(directory, seed_path, manifest_path, source_root, outdir=None, ledger_path=None,
            _expected_counts=COUNTS, _resamples=BOOTSTRAPS, _synthetic=False,
            _global_id_offset=0):
    directory, seed_path, manifest_path, source_root = map(Path, (directory, seed_path, manifest_path, source_root))
    failures, warnings, hashes, table, groups = [], [], {}, [], []

    def fail(code, group=None, sample=None, detail=None):
        item = {'code': code}
        if group is not None: item['group_id'] = group
        if sample is not None: item['sample'] = sample
        if detail is not None: item['detail'] = str(detail)
        failures.append(item)

    def check(condition, code, group=None, sample=None, detail=None):
        if not condition: fail(code, group, sample, detail)
        return condition

    def read_json(path):
        try:
            hashes[str(path)] = sha256(path)
            return json.loads(path.read_text())
        except (OSError, ValueError) as error:
            fail('unreadable_json', detail=f'{path}: {error}')
            return {}

    def read_lines(path):
        try:
            hashes[str(path)] = sha256(path)
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except (OSError, ValueError) as error:
            fail('unreadable_jsonl', detail=f'{path}: {error}')
            return []

    seeds, manifest = read_json(seed_path), read_json(manifest_path)
    expected_n = sum(_expected_counts)
    check(_synthetic or not manifest.get('synthetic_fixture'), 'synthetic_fixture_rejected_in_production')
    check(seeds.get('total_samples') == expected_n, 'seed_manifest_total_count')
    check(len(seeds.get('groups', [])) == 4, 'seed_manifest_group_count')
    if not _synthetic:
        check(manifest.get('total_samples')==1726 and manifest.get('group_counts')==list(COUNTS), 'production_manifest_sample_counts')
        check(manifest.get('GPUs_per_group')==2 and manifest.get('max_concurrent_GPUs')==8, 'production_manifest_GPU_layout')
        check(manifest.get('quality_thresholds')=={'norm_abs_error_max':.001,'FP64_probability_relative_error_max':.0001},
              'production_manifest_quality_thresholds')
        check(manifest.get('preselected_statistic')=={'K':4,'Hoeffding_one_sided_alpha':.05,'XEB_threshold':.092,
              'bootstrap_replicates':BOOTSTRAPS,'bootstrap_seed':BOOTSTRAP_SEED},'production_manifest_statistics')
    try:
        declared_seeds=[int(x['cli_seed_decimal']) for x in seeds.get('groups', [])]
        declared_states=[state_decimal(x['initial_pcg64_state_decimal128']) for x in seeds.get('groups', [])]
        check(len(declared_seeds)==4 and len(set(declared_seeds))==4, 'group_rng_seeds_not_distinct')
        check(len({x['state']['inc'] for x in declared_states})==4,'group_PCG64_stream_increments_not_distinct')
        check(len({(x['state']['state'],x['state']['inc']) for x in declared_states})==4,'group_PCG64_initial_states_not_distinct')
    except (KeyError,TypeError,ValueError):fail('malformed_group_seed_or_initial_state')
    files = manifest.get('files', {})
    check(bool(files), 'empty_frozen_source_manifest')
    if not _synthetic:
        required=['born_production.py','scan_oracle.py','graph_sampler.py','distributed_sampler.py',
                  'fourier_covariant_sampler.py','fourier_fused_sampler.py','deferred_global_sampler.py',
                  'scan_order.npy','run_born_production.sh','upstream_scan/scan_plan.py','upstream_scan/gpu_kernel_fp64.py']
        if manifest.get('engine')=='specialized_deferred':required.extend(['specialized_deferred_sampler.py','init_specialization.py'])
        for name in required:check(name in files,'executed_dependency_not_hash_bound',detail=name)
    for name, digest in files.items():
        path = source_root/name
        try:
            actual = sha256(path)
            hashes[str(path)] = actual
            check(actual == digest, 'frozen_file_hash_mismatch', detail=name)
        except OSError as error:
            fail('frozen_file_missing', detail=f'{name}: {error}')
    seed_declared = manifest.get('seed_manifest_sha256') or files.get(manifest.get('seed_manifest', ''))
    check(seed_declared == hashes.get(str(seed_path)), 'seed_manifest_not_bound_by_production_manifest')
    check(manifest.get('uniforms_per_sample') == 3, 'production_manifest_uniform_schedule_missing_or_not_3')
    if 'two rng.random' in seeds.get('sample_protocol', ''):
        warnings.append('Seed manifest retains the earlier two-uniform description; final production manifest and logs explicitly require three. Seeds/initial states are unchanged.')
    graph_path = source_root/manifest.get('graph', 'rgs_graph.npz')
    A = B = None
    try:
        with np.load(graph_path, allow_pickle=False) as graph:
            A, B = graph['A'].astype(int).tolist(), graph['B'].astype(int).tolist()
            adjacency=graph['graph']
        valid_partition=check(len(A)==len(B)==35 and sorted(A+B)==list(range(70)), 'invalid_frozen_partition')
        valid_graph=check(adjacency.shape==(70,70) and np.array_equal(adjacency,adjacency.T) and
                          not np.diag(adjacency).any() and np.isin(adjacency,[0,1]).all(), 'invalid_binary_graph')
        if valid_partition and valid_graph:
            check(gf2_rank(adjacency[np.ix_(A,B)])==35,'cross_partition_GF2_rank_not_35')
        check(manifest.get('graph') in files, 'graph_not_hash_bound')
    except (OSError, ValueError, KeyError) as error:
        fail('unreadable_graph_partition', detail=error)
    generation_sum = process_sum = scoring_sum = 0.
    script_spans = []
    global_ids = []
    for gid, expected_count in enumerate(_expected_counts):
        local_fail_start = len(failures)
        expected_start = _global_id_offset + sum(_expected_counts[:gid])
        group_seed = seeds.get('groups', [])[gid] if gid < len(seeds.get('groups', [])) else {}
        check(group_seed.get('group_id') == gid, 'seed_group_id', gid)
        check(group_seed.get('sample_count') == expected_count, 'seed_group_count', gid)
        check(group_seed.get('global_sample_id_start') == expected_start and
              group_seed.get('global_sample_id_end_inclusive') == expected_start+expected_count-1,
              'seed_global_id_range', gid)
        production = read_lines(directory/f'born_group{gid}.jsonl')
        scored = read_lines(directory/f'born_group{gid}_fp64.jsonl')
        metas = [r['metadata'] for r in production if isinstance(r, dict) and 'metadata' in r]
        ends = [r['summary'] for r in production if isinstance(r, dict) and 'summary' in r]
        rows = [r for r in production if isinstance(r, dict) and 'sample' in r]
        smetas = [r['metadata'] for r in scored if isinstance(r, dict) and 'metadata' in r]
        scores = [r for r in scored if isinstance(r, dict) and 'probability' in r]
        check(len(metas)==1 and len(ends)==1, 'production_header_footer_count', gid)
        check(len(smetas)==1, 'scoring_header_count', gid)
        check(len(rows)==expected_count, 'production_sample_count', gid, detail=len(rows))
        check(len(scores)==expected_count, 'fp64_sample_count', gid, detail=len(scores))
        meta, end, smeta = (metas[0] if metas else {}), (ends[0] if ends else {}), (smetas[0] if smetas else {})
        check(meta.get('group') == group_seed, 'logged_seed_group_not_manifest_group', gid)
        check(meta.get('manifest_sha256') == hashes.get(str(manifest_path)), 'logged_manifest_hash', gid)
        check(meta.get('engine') == manifest.get('engine'), 'logged_engine', gid)
        check(meta.get('A_vertices') == A and meta.get('B_vertices') == B and meta.get('conditional_qubits') == 35,
              'logged_partition_mapping', gid)
        check('3 uniform' in meta.get('rng_schedule', ''), 'logged_rng_schedule', gid)
        check(smeta.get('dtype')=='complex128' and smeta.get('reverse_order') is True and smeta.get('n')==70,
              'scoring_not_reverse_complex128_70q', gid)
        check(end.get('samples')==expected_count and end.get('group_id')==gid and end.get('all_count_completed') is True,
              'production_completion_footer', gid)
        try:
            rng = np.random.Generator(np.random.PCG64(int(group_seed['cli_seed_decimal'])))
            check(state_decimal(rng.bit_generator.state)==group_seed['initial_pcg64_state_decimal128'], 'initial_rng_state', gid)
        except (KeyError, ValueError, TypeError) as error:
            fail('invalid_rng_manifest', gid, detail=error)
            rng = None
        group_table = []
        for index, row in enumerate(rows):
            try:
                sid = row['global_sample_id']; global_ids.append(sid)
                check(row['sample']==index and row['group_id']==gid and sid==expected_start+index,
                      'sample_or_global_id', gid, index)
                if rng is not None:
                    check(state_decimal(row['rngstate_before'])==state_decimal(rng.bit_generator.state), 'rng_before', gid, index)
                    expected_b = rng.integers(0, 2, 35).tolist()
                    expected_u = [float(rng.random()) for _ in range(3)]
                    check(row['B_bits']==expected_b, 'B_rng_replay', gid, index)
                    check(row['draw_uniforms']==expected_u, 'three_uniforms_rng_replay', gid, index)
                    check(state_decimal(row['rngstate_after'])==state_decimal(rng.bit_generator.state), 'rng_after', gid, index)
                bits, ai, bb = row['bitstring_q0_left'], row['A_index'], row['B_bits']
                valid_bits = isinstance(bits, str) and len(bits)==70 and not set(bits)-set('01')
                valid_ai = isinstance(ai, int) and 0<=ai<2**35
                valid_bb = isinstance(bb, list) and len(bb)==35 and all(v in (0,1) for v in bb)
                check(valid_bits and valid_ai and valid_bb, 'invalid_bitstring_or_index', gid, index)
                if valid_bits and valid_ai and valid_bb and A is not None:
                    reconstructed = [0]*70
                    for j,q in enumerate(A): reconstructed[q]=(ai>>j)&1
                    for j,q in enumerate(B): reconstructed[q]=bb[j]
                    check(bits==''.join(map(str,reconstructed)), 'A_B_to_full_bitstring_mapping', gid, index)
                if 'source_sha256' in row:
                    check(row['source_sha256']==files.get('born_production.py'), 'sample_source_hash', gid, index)
                raw, normalized, norm = [float(row[k]) for k in ('probability_raw','probability','conditional_norm')]
                if not check(all(math.isfinite(v) and v>0 for v in (raw,normalized,norm)) and raw<=1 and normalized<=1,
                             'invalid_generated_probability_or_norm', gid, index):continue
                check(abs(norm-1)<1e-3, 'conditional_norm_threshold', gid, index)
                check(math.isclose(normalized,raw/norm,rel_tol=5e-15,abs_tol=0), 'normalized_probability_identity', gid, index)
                if index>=len(scores): continue
                score=scores[index]
                check(score.get('index')==index and score.get('bitstring_q0_left')==bits, 'fp64_index_bitstring_mapping', gid, index)
                if 'global_sample_id' in score: check(score['global_sample_id']==sid,'fp64_global_id',gid,index)
                fp64=float(score['probability']); seconds=float(score['seconds'])
                if not check(math.isfinite(fp64) and 0<fp64<=1 and math.isfinite(seconds) and seconds>=0,
                             'invalid_fp64_probability_or_time',gid,index): continue
                raw_rel=abs(raw-fp64)/fp64; norm_rel=abs(normalized-fp64)/fp64
                if not check(math.isfinite(raw_rel) and math.isfinite(norm_rel),'nonfinite_probability_error',gid,index):continue
                check(raw_rel<1e-4, 'raw_probability_relative_error_ge_1e_4',gid,index,raw_rel)
                elapsed=float(row['elapsed_seconds'])
                if not check(math.isfinite(elapsed) and elapsed>=0,'invalid_sample_time',gid,index):continue
                item={'group_id':gid,'sample':index,'global_sample_id':sid,'bitstring_q0_left':bits,
                      'A_index':ai,'B_bits':''.join(map(str,bb)),'p_raw':raw,'p_normalized':normalized,
                      'p_fp64':fp64,'conditional_norm':norm,'norm_absolute_error':abs(norm-1),
                      'raw_relative_error':raw_rel,'normalized_relative_error':norm_rel,
                      'raw_scaled_absolute_error':abs(raw-fp64)*2**70,
                      'normalized_scaled_absolute_error':abs(normalized-fp64)*2**70,
                      'xeb_fp64':2**70*fp64-1,'generation_sample_seconds':elapsed,'fp64_seconds':seconds}
                group_table.append(item);table.append(item)
            except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as error:
                fail('malformed_sample_record',gid,index,error)
        timing={}
        try:
            generation=float(end['generation_seconds']);process=float(end['process_elapsed_seconds']);setup=float(end['setup_seconds'])
            if not check(all(math.isfinite(x) and x>=0 for x in (generation,process,setup)), 'invalid_group_time',gid):
                raise ValueError('Nonfinite or negative group timing')
            check(process+1e-3>=generation+setup, 'group_time_order',gid)
            check(generation+1e-3>=sum(x['generation_sample_seconds'] for x in group_table), 'generation_less_than_sample_sum',gid)
            check(math.isclose(setup,float(meta['setup_seconds']),rel_tol=0,abs_tol=1e-12),'setup_header_footer_mismatch',gid)
            timing={'generation_seconds':generation,'process_elapsed_seconds':process,'setup_seconds':setup,
                    'process_to_ready_seconds':meta.get('process_to_ready_seconds'),
                    'generation_start_unix':end.get('generation_start_unix'),'generation_end_unix':end.get('generation_end_unix'),
                    'process_start_unix':meta.get('process_start_unix'),
                    'fp64_oracle_seconds_sum':sum(x['fp64_seconds'] for x in group_table),
                    'fp64_setup_seconds':smeta.get('setup_seconds')}
            generation_sum+=generation;process_sum+=process;scoring_sum+=timing['fp64_oracle_seconds_sum']
            startfile,endfile=directory/f'born_group{gid}_start_unix.txt',directory/f'born_group{gid}_end_unix.txt'
            if startfile.exists() and endfile.exists():
                span=float(endfile.read_text())-float(startfile.read_text());check(span>=0,'negative_script_span',gid)
                hashes[str(startfile)]=sha256(startfile);hashes[str(endfile)]=sha256(endfile)
                timing['recorded_script_span_seconds']=span;script_spans.append(span)
            exitfile=directory/f'born_group{gid}_exit.json'
            if exitfile.exists():check(read_json(exitfile).get('exit_status')==0,'worker_nonzero_exit',gid)
        except (KeyError,TypeError,ValueError) as error:fail('malformed_group_timing',gid,detail=error)
        groups.append({'group_id':gid,'expected_count':expected_count,'observed_count':len(rows),
                       'fp64_count':len(scores),'audit_passed':len(failures)==local_fail_start,
                       'score':statistics([x['xeb_fp64'] for x in group_table],_resamples),
                       'timing':timing,'errors':{k:distribution_summary([x[k] for x in group_table]) for k in
                           ('norm_absolute_error','raw_relative_error','normalized_relative_error','raw_scaled_absolute_error','normalized_scaled_absolute_error')}})
    check(global_ids==list(range(_global_id_offset, _global_id_offset+expected_n)),
          'complete_ordered_global_ids_expected_offset')
    check(len(table)==expected_n,'complete_fp64_scored_count',detail=len(table))
    external={}
    central=None
    ledger_mapping={}
    if ledger_path is not None:
        external=read_json(Path(ledger_path))
        central_key=('central_launch_to_return_seconds' if 'central_launch_to_return_seconds' in external
                     else 'central_monotonic_submit_to_all_logs_returned_seconds')
        scope_key='timer_scope' if 'timer_scope' in external else 'clock_scope'
        central=external.get(central_key)
        ledger_mapping={'central_launch_to_return_seconds_from':central_key,'timer_scope_from':scope_key,
                        'reported_scope':external.get(scope_key),
                        'note':'Canonical field alias only; original external ledger is retained unchanged.'}
        if 'central_launch_to_return_seconds' in external and 'central_monotonic_submit_to_all_logs_returned_seconds' in external:
            check(external['central_launch_to_return_seconds']==external['central_monotonic_submit_to_all_logs_returned_seconds'],
                  'conflicting_external_central_timer_fields')
        check(isinstance(central,(int,float)) and math.isfinite(central) and central>=0,
              'invalid_external_central_launch_to_return_seconds')
        check(bool(external.get(scope_key)), 'external_ledger_timer_scope_missing')
        if isinstance(central,(int,float)) and math.isfinite(central):
            longest_process=max((g['timing'].get('process_elapsed_seconds',0) for g in groups),default=0)
            check(central+1e-3>=longest_process,'central_span_shorter_than_worker_process')
            if script_spans:check(central+1>=max(script_spans),'central_span_shorter_than_worker_script')
    timing={'generation_seconds_sum':generation_sum,'process_seconds_sum':process_sum,
            'generation_GPU_seconds_2_per_group':2*generation_sum,
            'process_GPU_seconds_2_per_group':2*process_sum,
            'generation_GPU_hours_2_per_group':2*generation_sum/3600,
            'process_GPU_hours_2_per_group':2*process_sum/3600,
            'independent_FP64_oracle_seconds_sum':scoring_sum,
            'independent_FP64_oracle_GPU_seconds_one_device_each':scoring_sum,
            'script_GPU_seconds_2_per_group':2*sum(script_spans) if len(script_spans)==4 else None,
            'maximum_worker_generation_seconds_NOT_central_span':max((g['timing'].get('generation_seconds',0) for g in groups),default=None),
            'central_launch_to_return_seconds':central,
            'central_span_source':'external ledger only' if ledger_path is not None else 'not supplied; not inferred from workers',
            'scope':'Process timer begins inside Python main, after top-level imports; generation excludes setup. GPU-seconds assume two held devices per group and do not certify scheduler allocation duration. Script spans include scoring where available. FP64 oracle sums exclude scorer startup unless separately reported.'}
    result={'scope':'SYNTHETIC SELF-TEST ONLY' if _synthetic else 'Independent audit of one fixed conditional Born production run; finite-precision consistency, not a global TV or quantum-state fidelity certificate',
            'validation_passed':not failures,'expected_samples':expected_n,'matched_scored_samples':len(table),
            'failures':failures,'warnings':warnings,'groups':groups,
            'score':statistics([x['xeb_fp64'] for x in table],_resamples,primary=True),
            'statistics_valid_only_if_audit_passes':True,
            'errors':{k:distribution_summary([x[k] for x in table]) for k in
                      ('norm_absolute_error','raw_relative_error','normalized_relative_error','raw_scaled_absolute_error','normalized_scaled_absolute_error')},
            'duplicate_complete_bitstrings_preserved':len(table)-len({x['bitstring_q0_left'] for x in table}),
            'timing':timing,'external_ledger':external,'external_ledger_field_mapping':ledger_mapping,'input_sha256':hashes,
            'RNG_replay_limitation':'Replaying B and three uniforms verifies RNG consumption and logged mappings, not the unlogged full CDF/mass selection. GPU/CPU full-vector validation and independent probability scoring are separate evidence.',
            'production_seed_manifest_sha256':hashes.get(str(seed_path)),
            'production_manifest_sha256':hashes.get(str(manifest_path))}
    if failures:result['score']['primary_lower_above_QAT_0_092']=None
    if outdir is not None:
        outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
        (outdir/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        (outdir/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
        with (outdir/'samples.csv').open('w',newline='') as stream:
            fields=list(table[0]) if table else ['group_id','sample','global_sample_id']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(table)
    return result


def self_test():
    """No persistent fake experiment outputs; deliberately tiny mock logs only."""
    counts=(2,2,1,1)
    with tempfile.TemporaryDirectory(prefix='born_analyzer_SYNTHETIC_') as tmp:
        root=Path(tmp);A=list(range(35));B=list(range(35,70))
        graph=np.zeros((70,70),dtype=np.uint8);graph[np.arange(35),np.arange(35)+35]=1;graph|=graph.T
        np.savez(root/'graph.npz',A=A,B=B,graph=graph)
        (root/'born_production.py').write_text('# SYNTHETIC SOURCE-HASH FIXTURE ONLY\n')
        seeds={'total_samples':sum(counts),'groups':[]}
        for gid,count in enumerate(counts):
            rng=np.random.Generator(np.random.PCG64(101+gid))
            seeds['groups'].append({'group_id':gid,'sample_count':count,'global_sample_id_start':sum(counts[:gid]),
                'global_sample_id_end_inclusive':sum(counts[:gid+1])-1,'cli_seed_decimal':str(101+gid),
                'initial_pcg64_state_decimal128':state_decimal(rng.bit_generator.state)})
        sp=root/'seeds.json';sp.write_text(json.dumps(seeds))
        manifest={'synthetic_fixture':True,'graph':'graph.npz','seed_manifest':'seeds.json','uniforms_per_sample':3,
                  'engine':'synthetic','files':{name:sha256(root/name) for name in ('graph.npz','seeds.json','born_production.py')}}
        mp=root/'manifest.json';mp.write_text(json.dumps(manifest))
        for gid,entry in enumerate(seeds['groups']):
            rng=np.random.Generator(np.random.PCG64(int(entry['cli_seed_decimal'])))
            rows=[{'metadata':{'group':entry,'manifest_sha256':sha256(mp),'engine':'synthetic','A_vertices':A,'B_vertices':B,
                    'conditional_qubits':35,'rng_schedule':'35 integers then 3 uniforms','setup_seconds':.1}}]
            scores=[{'metadata':{'dtype':'complex128','reverse_order':True,'n':70,'setup_seconds':.1}}]
            for i in range(entry['sample_count']):
                before=state_decimal(rng.bit_generator.state);bb=rng.integers(0,2,35).tolist();uu=[float(rng.random()) for _ in range(3)]
                ai=i;bits=''.join(str((ai>>j)&1) for j in range(35))+''.join(map(str,bb))
                rows.append({'sample':i,'group_id':gid,'global_sample_id':entry['global_sample_id_start']+i,
                    'bitstring_q0_left':bits,'B_bits':bb,'A_index':ai,'rngstate_before':before,
                    'rngstate_after':state_decimal(rng.bit_generator.state),'draw_uniforms':uu,
                    'probability_raw':2.**-70,'probability':2.**-70,'conditional_norm':1.,'elapsed_seconds':.1})
                scores.append({'index':i,'bitstring_q0_left':bits,'probability':2.**-70,'seconds':.1})
            rows.append({'summary':{'samples':entry['sample_count'],'group_id':gid,'all_count_completed':True,
                                    'generation_seconds':1.,'process_elapsed_seconds':2.,'setup_seconds':.1}})
            (root/f'born_group{gid}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            (root/f'born_group{gid}_fp64.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in scores))
        def run():return analyze(root,sp,mp,root,_expected_counts=counts,_resamples=32,_synthetic=True)
        assert run()['validation_passed']
        cases=[];target=root/'born_group0.jsonl';original=target.read_text()
        mutations=[('uniform',lambda r:r[1]['draw_uniforms'].__setitem__(0,.123), 'three_uniforms_rng_replay'),
                   ('before',lambda r:r[1]['rngstate_before']['state'].__setitem__('state','0'),'rng_before'),
                   ('after',lambda r:r[1]['rngstate_after']['state'].__setitem__('inc','0'),'rng_after'),
                   ('B',lambda r:r[1]['B_bits'].__setitem__(0,1-r[1]['B_bits'][0]),'B_rng_replay'),
                   ('global_id',lambda r:r[1].__setitem__('global_sample_id',100),'sample_or_global_id'),
                   ('source_hash',lambda r:r[0]['metadata'].__setitem__('manifest_sha256','0'*64),'logged_manifest_hash'),
                   ('count',lambda r:r.pop(1),'production_sample_count')]
        for name,mutate,code in mutations:
            rows=[json.loads(x) for x in original.splitlines()];mutate(rows)
            target.write_text(''.join(json.dumps(x)+'\n' for x in rows));result=run()
            assert not result['validation_passed'] and code in [x['code'] for x in result['failures']],name
            cases.append(name);target.write_text(original)
        sf=root/'born_group0_fp64.jsonl';saved=sf.read_text();rows=[json.loads(x) for x in saved.splitlines()]
        rows[1]['bitstring_q0_left']='0'*70;sf.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        assert 'fp64_index_bitstring_mapping' in [x['code'] for x in run()['failures']];cases.append('FP64_mapping');sf.write_text(saved)
        source=root/'born_production.py';source_saved=source.read_text();source.write_text(source_saved+'# modified\n')
        assert 'frozen_file_hash_mismatch' in [x['code'] for x in run()['failures']];cases.append('actual_source_file_hash');source.write_text(source_saved)
        ledger=root/'ledger.json';ledger.write_text(json.dumps({'maximum_worker_seconds':1.}))
        ledger_result=analyze(root,sp,mp,root,ledger_path=ledger,_expected_counts=counts,_resamples=32,_synthetic=True)
        assert 'invalid_external_central_launch_to_return_seconds' in [x['code'] for x in ledger_result['failures']]
        cases.append('worker_maximum_cannot_become_central_span')
        ledger.write_text(json.dumps({'central_launch_to_return_seconds':1.,'timer_scope':'synthetic launcher start through all returns'}))
        ledger_result=analyze(root,sp,mp,root,ledger_path=ledger,_expected_counts=counts,_resamples=32,_synthetic=True)
        assert 'central_span_shorter_than_worker_process' in [x['code'] for x in ledger_result['failures']]
        cases.append('implausibly_short_central_span')
        ledger.write_text(json.dumps({'central_monotonic_submit_to_all_logs_returned_seconds':10.,
                                     'clock_scope':'synthetic launcher submit to all logs returned'}))
        ledger_result=analyze(root,sp,mp,root,ledger_path=ledger,_expected_counts=counts,_resamples=32,_synthetic=True)
        assert ledger_result['validation_passed'] and ledger_result['timing']['central_launch_to_return_seconds']==10.
        cases.append('real_launcher_ledger_field_aliases')
        strict=analyze(root,sp,mp,root,_resamples=32)
        assert not strict['validation_passed'];cases.append('production_requires_1726_and_rejects_synthetic')
    return {'self_test_passed':True,'scope':'Temporary synthetic fixtures only; no real production results generated',
            'valid_fixture_passed':True,'deliberate_corruptions_detected':cases}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?')
    parser.add_argument('--seed-manifest')
    parser.add_argument('--manifest')
    parser.add_argument('--source-root',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--ledger',help='External launcher ledger: central_launch_to_return_seconds/timer_scope or central_monotonic_submit_to_all_logs_returned_seconds/clock_scope; never inferred from worker maxima')
    parser.add_argument('--outdir')
    parser.add_argument('--global-id-offset', type=int, default=0,
                        help='Expected first global sample ID; use this for an independent appended seed run.')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: print(json.dumps(self_test(),indent=2));return
    if not args.directory: parser.error('directory is required unless --self-test is used')
    directory=Path(args.directory)
    result=analyze(directory,args.seed_manifest or directory/'production_seed_manifest.json',
                   args.manifest or directory/'born_production_manifest.json',args.source_root,
                   args.outdir or directory/'analysis',args.ledger,
                   _global_id_offset=args.global_id_offset)
    print(json.dumps({'validation_passed':result['validation_passed'],'matched_scored_samples':result['matched_scored_samples'],
                      'failure_count':len(result['failures']),'outdir':str(args.outdir or directory/'analysis')},indent=2))
    if not result['validation_passed']:raise SystemExit(1)


if __name__=='__main__':main()
