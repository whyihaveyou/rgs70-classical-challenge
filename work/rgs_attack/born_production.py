"""Fixed-count iid conditional Born production, with complete RNG and timing logs."""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
import numpy as np

def decimal_state(state):
    s=copy.deepcopy(state)
    for k in ('state','inc'):s['state'][k]=str(s['state'][k])
    return s

class DrawRecorder:
    def __init__(self,rng):self.rng=rng;self.uniforms=[]
    def random(self):
        v=float(self.rng.random());self.uniforms.append(v);return v

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',type=int,required=True);p.add_argument('--manifest',default='born_production_manifest.json');p.add_argument('--out',required=True);a=p.parse_args()
    process_start=time.perf_counter();wall_start=time.time();manifest_path=Path(a.manifest);manifest=json.loads(manifest_path.read_text())
    for path,digest in manifest['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(f'Frozen source/input changed: {path}')
    streams=json.loads(Path(manifest['seed_manifest']).read_text());group=streams['groups'][a.group]
    if group['group_id']!=a.group:raise ValueError('Group mismatch')
    rng=np.random.Generator(np.random.PCG64(int(group['cli_seed_decimal'])))
    if decimal_state(rng.bit_generator.state)!=group['initial_pcg64_state_decimal128']:raise ValueError('Initial RNG mismatch')
    data=np.load(manifest['graph']);g=data['graph'];u=data['measurement_unitaries'];A=data['A'].tolist();B=data['B'].tolist()
    if manifest['engine']=='specialized_deferred':
        from specialized_deferred_sampler import SpecializedDeferredSampler
        klass=SpecializedDeferredSampler
    elif manifest['engine']=='deferred_fused':
        from deferred_global_sampler import DeferredGlobalMixin
        from fourier_fused_sampler import FusedFourierSampler
        klass=type('ProductionDeferredFused',(DeferredGlobalMixin,FusedFourierSampler),{})
    else:raise ValueError('Unvalidated engine')
    t=time.perf_counter();engine=klass(g,u,A,B,[0,1],chunk_mib=1,fusion_bits=5);setup=time.perf_counter()-t
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Production output already exists; do not overwrite or silently resume')
    meta=dict(group=group,manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),process_start_unix=wall_start,
        setup_seconds=setup,process_to_ready_seconds=time.perf_counter()-process_start,engine=manifest['engine'],numpy=np.__version__,cupy=engine.cp.__version__,
        conditional_qubits=len(A),B_vertices=B,A_vertices=A,rng_schedule='fresh 35 B integers, then 3 uniform draws; no warmup consumes production RNG',
        statement='Exact-in-arithmetic conditional Born algorithm; complex64 numerical implementation, no formal full-distribution TV certificate')
    with out.open('x') as f:
        f.write(json.dumps({'metadata':meta})+'\n');f.flush();print(json.dumps({'metadata':meta}),flush=True)
        gen_start=time.perf_counter();gen_wall=time.time()
        for i in range(group['sample_count']):
            before=decimal_state(rng.bit_generator.state);start=time.perf_counter();Bbits=rng.integers(0,2,len(B)).tolist()
            stages=engine.compute(Bbits);draws=DrawRecorder(rng);index,pa,norm=engine.draw(draws)
            if len(draws.uniforms)!=3:raise AssertionError('RNG call schedule changed')
            if not (0<=index<(1<<len(A)) and np.isfinite(pa) and pa>0 and abs(norm-1)<1e-3):raise ArithmeticError('Invalid production result')
            bits=[0]*len(g)
            for j,q in enumerate(A):bits[q]=(index>>j)&1
            for j,q in enumerate(B):bits[q]=Bbits[j]
            rec=dict(sample=i,group_id=a.group,global_sample_id=group['global_sample_id_start']+i,
                bitstring_q0_left=''.join(map(str,bits)),B_bits=Bbits,A_index=index,
                rngstate_before=before,rngstate_after=decimal_state(rng.bit_generator.state),draw_uniforms=draws.uniforms,
                probability_raw=pa*2.**(-len(B)),probability=pa/norm*2.**(-len(B)),conditional_norm=norm,
                elapsed_seconds=time.perf_counter()-start,stages=stages)
            f.write(json.dumps(rec)+'\n');f.flush()
            if (i+1)%16==0:print(json.dumps({'progress':i+1,'group':a.group,'elapsed_seconds':time.perf_counter()-gen_start}),flush=True)
        engine.sync();f.flush();os.fsync(f.fileno())
        summary=dict(samples=group['sample_count'],group_id=a.group,generation_start_unix=gen_wall,generation_end_unix=time.time(),generation_seconds=time.perf_counter()-gen_start,
            process_elapsed_seconds=time.perf_counter()-process_start,setup_seconds=setup,all_count_completed=True)
        f.write(json.dumps({'summary':summary})+'\n');f.flush();os.fsync(f.fileno());print(json.dumps({'summary':summary}),flush=True)
if __name__=='__main__':main()
