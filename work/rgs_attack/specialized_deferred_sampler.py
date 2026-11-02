"""Lookup-initialized, fused conditional sampler with deferred global gates.

This independent variant combines three exact algebraic changes: Fourier
covariance, high-bit transform fusion, and the identical-B-unitary lookup
specialization. It inherits the already tested auxiliary-shard draw from
DeferredGlobalMixin. No frozen source is modified and no GPU is allocated
unless explicitly invoked through the command-line GPU mode.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from distributed_sampler import CUDA
from deferred_global_sampler import DeferredGlobalMixin
from fourier_fused_sampler import FusedFourierSampler, cpu_tile_transform, high_groups
from fourier_covariant_sampler import parity64
from graph_sampler import conditional, quadratic_phase, apply_one, gf2rank
from init_specialization import (build_specialization, output_mask, popcount64,
                                 CUDA_INIT_SPECIALIZATION)

SPECIALIZED_PAIR_CUDA=r'''
extern "C" __global__ void initcov_specialized_global(
    float2*v,ull off,ull n,ull outmask,ull map_xor,ull phase_diff_mask,
    const ulonglong2*t,const float2*weights,int m,int chunk_bits,int chunks){
 const ull high=1ull<<(m-1);
 for(ull j=(ull)blockIdx.x*blockDim.x+threadIdx.x;j<n;j+=(ull)gridDim.x*blockDim.x){
  ull id=j+off,low=id&(high-1);
  float4 pair=initcov_specialized_pair(low,outmask,map_xor,phase_diff_mask,t,weights,m,chunk_bits,chunks);
  float2 a=make_float2(pair.x,pair.y),b=make_float2(pair.z,pair.w);
  v[j]=(id&high)?sub(a,b):add(a,b);
 }
}
'''


class SpecializedDeferredSampler(DeferredGlobalMixin,FusedFourierSampler):
    def __init__(self,g,u,A,B,devices,chunk_mib=1,fusion_bits=5,lane_bits=5,lookup_bits=8):
        # Reject nonidentical measurement matrices before allocating a state.
        spec=build_specialization(g,u,A,B,chunk_bits=lookup_bits)
        super().__init__(g,u,A,B,devices,chunk_mib=chunk_mib,fusion_bits=fusion_bits,lane_bits=lane_bits)
        self.init_spec=spec;self.lookup_bits=lookup_bits;self.lookup_modules=[]
        self.lookup_map_xor,self.lookup_phase_mask=spec.toggle_metadata(self.m-1)
        self.host_weight_cache={}
        cp=self.cp
        for i,dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                mod=cp.RawModule(code=CUDA+CUDA_INIT_SPECIALIZATION+SPECIALIZED_PAIR_CUDA,options=('--std=c++11',))
                self.lookup_modules.append(mod)
                self.kern[i]['initcov_specialized_global']=mod.get_function('initcov_specialized_global')
                self.small[i]['lookup_tables']=cp.asarray(spec.packed_tables)
                self.small[i]['lookup_weights']=cp.empty((self.m+1,self.m+1),cp.complex64)
                cp.cuda.runtime.deviceSynchronize()

    def compute(self,outB):
        if len(outB)!=self.m:raise ValueError('wrong output condition length')
        mask=output_mask(outB);h=bin(mask).count('1');stage={};start=time.perf_counter()
        if h not in self.host_weight_cache:
            self.host_weight_cache[h]=self.init_spec.weight_table(outB,np.complex64)
        weights=self.host_weight_cache[h]
        for i,dev in enumerate(self.devices):
            with self.cp.cuda.Device(dev):self.small[i]['lookup_weights'].set(weights,stream=self.streams[i])
            self.launch(i,'initcov_specialized_global',(self.arr[i],np.uint64(i*self.n),np.uint64(self.n),
                np.uint64(mask),np.uint64(self.lookup_map_xor),np.uint64(self.lookup_phase_mask),
                self.small[i]['lookup_tables'],self.small[i]['lookup_weights'],np.int32(self.m),
                np.int32(self.init_spec.chunk_bits),np.int32(self.init_spec.chunks)))
        self.sync();stage['initialize_specialized_global']=time.perf_counter()-start;start=time.perf_counter()
        self.local_walsh();stage['walsh_fused']=time.perf_counter()-start;start=time.perf_counter()
        self.local_product_with_phase();stage['product_measurement_fused']=time.perf_counter()-start
        stage['linear_permutation_phase']=0.0
        return stage


def cpu_specialized_pair(spec,ids,outB,dtype=np.complex128):
    """CPU replica of the one-LUT-traversal pair helper used by the GPU."""
    ids=np.asarray(ids,np.uint64)
    if np.any(ids>>np.uint64(spec.m-1)):raise ValueError('pair IDs must have the global bit clear')
    mapped,phase=spec.map_and_phase(ids)
    map_xor,phase_mask=spec.toggle_metadata(spec.m-1)
    mapped1=mapped^np.uint64(map_xor)
    phase1=phase*(1.-2.*parity64((ids|np.uint64(1<<63))&np.uint64(phase_mask)))
    w=spec.weight_table(outB,dtype);mask=np.uint64(output_mask(outB))
    a=w[popcount64(mapped),popcount64(mapped&mask)]*phase
    b=w[popcount64(mapped1),popcount64(mapped1&mask)]*phase1
    return a.astype(dtype),b.astype(dtype)


def selfcheck(out):
    root=Path(__file__).resolve().parent;rng=np.random.default_rng(2026103103);records=[]
    for name in ('induced_12q.npz','induced_24q.npz','induced_32q.npz','induced_40q.npz','rgs_graph.npz','rgs_graph_alt_partition.npz'):
        d=np.load(root/name);g=d['graph'];u=d['measurement_unitaries'];A=d['A'].tolist();B=d['B'].tolist();spec=build_specialization(g,u,A,B);m=len(A);L=m-1
        ids=(np.arange(1<<L,dtype=np.uint64) if m<=20 else
             np.concatenate([np.array([0,(1<<L)-1],np.uint64),np.array([1<<j for j in range(L)],np.uint64),rng.integers(0,1<<L,10000,dtype=np.uint64)]))
        trials=1 if m==20 else 3
        for trial in range(trials):
            ob=([0]*m if trial==0 and m!=20 else [1]*m if trial==1 else rng.integers(0,2,m).tolist())
            a,b=cpu_specialized_pair(spec,ids,ob);ra=spec.generic(ids,ob);rb=spec.generic(ids|(np.uint64(1)<<np.uint64(L)),ob)
            diff=np.concatenate([a-ra,b-rb]);reference=np.concatenate([ra,rb]);relative=float(np.linalg.norm(diff)/np.linalg.norm(reference));assert relative<3e-14
            a32,b32=cpu_specialized_pair(spec,ids,ob,np.complex64)
            relative32=float(np.linalg.norm(np.concatenate([a32-ra,b32-rb]))/np.linalg.norm(reference));assert relative32<2e-6
            rec=dict(fixture=name,trial=trial,ids_checked=len(ids),pair_relative_l2_error=relative,fp32_lookup_relative_l2_error=relative32)
            if m<=20:
                raw=np.concatenate([a+b,a-b])
                for shard in np.split(raw,2):
                    for q in range(min(10,L)):apply_one(shard,np.array([[1,1],[1,-1]],complex),q)
                    for q,c in high_groups(L):cpu_tile_transform(shard,q,c)
                quadratic_phase(raw,g[np.ix_(A,A)])
                for shard in np.split(raw,2):
                    for q in range(min(10,L)):apply_one(shard,u[A[q]],q)
                    for q,c in high_groups(L):cpu_tile_transform(shard,q,c,unitaries=u[A])
                final=raw.copy();apply_one(final,u[A[-1]],L)
                ref,pb=conditional(g,u,A,B,ob);ref/=np.sqrt(pb)
                err=float(np.max(np.abs(final-ref)));l2=float(np.linalg.norm(final-ref));assert err<1e-12 and l2<1e-11
                before=np.abs(raw.reshape(2,-1))**2;after=np.abs(final.reshape(2,-1))**2
                marginal=float(np.max(np.abs(before.sum(0)-after.sum(0))));assert marginal<1e-12
                rec.update(full_amplitude_max_error=err,full_amplitude_l2_error=l2,local_marginal_max_error=marginal)
            records.append(rec)
    report=dict(status='CPU pair lookup, fused transforms, full amplitudes and deferred local marginal passed; GPU validation pending',checks=records)
    Path(out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selfcheck',action='store_true');p.add_argument('--graph');p.add_argument('--devices',default='0,1');p.add_argument('--samples',type=int,default=4);p.add_argument('--seed',type=int,default=2026103103);p.add_argument('--validate',action='store_true');p.add_argument('--fusion-bits',type=int,default=5);p.add_argument('--lookup-bits',type=int,default=8);p.add_argument('--out',required=True);a=p.parse_args()
    if a.selfcheck:selfcheck(a.out);return
    if not a.graph:p.error('--graph required')
    d=np.load(a.graph);g=d['graph'];u=d['measurement_unitaries'];A=d['A'].tolist();B=d['B'].tolist();devices=list(map(int,a.devices.split(',')));rng=np.random.default_rng(a.seed)
    if a.validate and len(A)>20:raise ValueError('CPU oracle limited to <=20 conditional qubits')
    start=time.perf_counter();engine=SpecializedDeferredSampler(g,u,A,B,devices,fusion_bits=a.fusion_bits,lookup_bits=a.lookup_bits);setup=time.perf_counter()-start
    paths=[Path(__file__),Path(__file__).with_name('init_specialization.py'),Path(__file__).with_name('fourier_fused_sampler.py'),Path(__file__).with_name('deferred_global_sampler.py')]
    meta=dict(graph=a.graph,n_qubits=len(g),cut_rank=gf2rank(g[np.ix_(A,B)]),devices=devices,dtype='complex64',terminal_unitary_dtype='complex128',setup_seconds=setup,seed=a.seed,
        fusion_bits=a.fusion_bits,lookup_bits=a.lookup_bits,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        no_full_vector_peer_exchange=True,specialization=engine.init_spec.operation_counts(),status='experimental conditional Born sampler; requires independent probability validation')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as fh:
        fh.write(json.dumps({'metadata':meta})+'\n');fh.flush();print(json.dumps(meta),flush=True);generation_start=time.perf_counter()
        for sample in range(a.samples):
            ob=rng.integers(0,2,len(B)).tolist();start=time.perf_counter();stages=engine.compute(ob);index,pa,norm=engine.draw(rng);elapsed=time.perf_counter()-start
            bits=[0]*len(g)
            for j,site in enumerate(A):bits[site]=(index>>j)&1
            for j,site in enumerate(B):bits[site]=ob[j]
            rec=dict(sample=sample,bitstring_q0_left=''.join(map(str,bits)),probability_raw=pa*2.**(-len(B)),probability=pa/norm*2.**(-len(B)),conditional_norm=norm,
                xeb_contribution=2.**len(A)*pa-1,xeb_sampling_q=2.**len(A)*pa/norm-1,elapsed_seconds=elapsed,stages=stages)
            if a.validate:
                ref,pb=conditional(g,u,A,B,ob);ref/=np.sqrt(pb);got=engine.host();rec['oracle_l2_error']=float(np.linalg.norm(ref-got));rec['oracle_max_abs_error']=float(np.max(abs(ref-got)));assert rec['oracle_l2_error']<1e-5
            fh.write(json.dumps(rec)+'\n');fh.flush();print(json.dumps(rec),flush=True)
        summary=dict(generation_loop_seconds=time.perf_counter()-generation_start,setup_seconds=setup,samples=a.samples,includes_CPU_validation=a.validate)
        fh.write(json.dumps({'summary':summary})+'\n');print(json.dumps({'summary':summary}),flush=True)


if __name__=='__main__':main()
