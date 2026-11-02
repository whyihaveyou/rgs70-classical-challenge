"""Two-device conditional Born sampler with no full-vector peer exchange.

The global Walsh bit is evaluated during initialization, because Walsh factors
commute. The last global measurement unitary is deferred until drawing: draw
an auxiliary shard and a local index from pre-unitary masses, discard that
shard, transform the two amplitudes at the local index, and draw the output bit.
Only scalar amplitudes cross to the host. No clipping or reuse of B outcomes.
This is an independent variant; frozen production sources remain unchanged.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from distributed_sampler import CUDA, DistributedSampler
from fourier_covariant_sampler import (EXTRA_CUDA, FourierCovariantSampler,
    gf2_inverse, map_indices, phase_for_ids)
from graph_sampler import conditional, walsh, quadratic_phase, apply_one, gf2rank

GLOBAL_CUDA=r'''
__device__ __forceinline__ float2 covariant_value(ull id,const float2*w,const ull*qm,const ull*invTrows,int m,float scale){
 ull b=0;for(int k=0;k<m;k++)b|=((ull)(__popcll(id&invTrows[k])&1))<<k;
 float2 z=make_float2(scale,0);unsigned par=0;
 for(int k=0;k<m;k++){unsigned bit=(b>>k)&1;z=mul(z,w[2*k+bit]);if(bit)par^=(__popcll(b&qm[k])&1);}
 if(par){z.x=-z.x;z.y=-z.y;}return z;
}
extern "C" __global__ void initcov_global(float2*v,ull off,ull n,const float2*w,const ull*qm,const ull*invTrows,int m,float scale){
 const ull high=1ull<<(m-1);
 for(ull j=(ull)blockIdx.x*blockDim.x+threadIdx.x;j<n;j+=(ull)gridDim.x*blockDim.x){
  ull id=j+off,low=id&(high-1);
  float2 a=covariant_value(low,w,qm,invTrows,m,scale);
  float2 b=covariant_value(low|high,w,qm,invTrows,m,scale);
  v[j]=(id&high)?sub(a,b):add(a,b);
 }
}
'''

class DeferredGlobalMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.ng!=2:raise ValueError('Deferred variant currently requires exactly two devices')
        cp=self.cp;self.deferred_modules=[]
        for i,dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                module=cp.RawModule(code=CUDA+EXTRA_CUDA+GLOBAL_CUDA,options=('--std=c++11',))
                self.deferred_modules.append(module)
                self.kern[i]['initcov_global']=module.get_function('initcov_global')
        self.deferred_global_unitary=np.asarray(self.u[self.A[-1]],np.complex128)
        if np.max(abs(self.deferred_global_unitary.conj().T@self.deferred_global_unitary-np.eye(2)))>1e-12:
            raise ValueError('Last product gate is not unitary')

    def launch(self,i,name,args,grid=None,block=256,shared=0):
        if name=='initcov':name='initcov_global'
        return super().launch(i,name,args,grid=grid,block=block,shared=shared)

    def global_gate(self,q,mats):
        if q!=self.m-1:raise ValueError('Unexpected nonlocal transform')
        h=np.array([[1,1],[1,-1]],np.complex64)
        if not (np.allclose(mats,h,atol=1e-7,rtol=0) or np.allclose(mats,self.deferred_global_unitary,atol=1e-7,rtol=0)):
            raise ValueError('Only the specified Walsh or terminal product unitary can be deferred')
        # H was already fused into initcov_global; U is applied in draw/host.

    def transformed_pair(self,local_index):
        if not 0<=local_index<self.n:raise ValueError('Local index outside shard')
        raw=[]
        for i,dev in enumerate(self.devices):
            with self.cp.cuda.Device(dev):raw.append(complex(self.cp.asnumpy(self.arr[i][local_index:local_index+1])[0]))
        return self.deferred_global_unitary@np.asarray(raw,np.complex128)

    def draw(self,rng):
        auxiliary,unused_probability,norm=DistributedSampler.draw(self,rng)
        local=auxiliary%self.n
        pair=self.transformed_pair(local);p=np.abs(pair)**2;total=float(p.sum())
        if not np.isfinite(total) or total<=0:raise ArithmeticError('Invalid transformed local mass')
        bit=int(rng.random()*total>=p[0])
        return bit*self.n+local,float(p[bit]),norm

    def probability_at(self,index):
        bit=int(index//self.n)
        if bit not in (0,1):raise ValueError('Bad global index')
        return float(abs(self.transformed_pair(index%self.n)[bit])**2)

    def host(self):
        raw=DistributedSampler.host(self).astype(np.complex128)
        apply_one(raw,self.deferred_global_unitary,self.m-1)
        return raw

class DeferredFourierSampler(DeferredGlobalMixin,FourierCovariantSampler):
    pass

def cpu_deferred(g,u,A,B,outB):
    m=len(A);N=gf2_inverse(g[np.ix_(B,A)]).T
    ids=np.arange(1<<(m-1),dtype=np.uint64)
    values=[]
    for bit in (0,1):
        z=ids|(np.uint64(bit)<<np.uint64(m-1));b=map_indices(z,N)
        v=np.full(len(ids),2.**(-m/2),np.complex128)
        for j,site in enumerate(B):v*=u[site,outB[j],((b>>np.uint64(j))&1).astype(int)]
        v*=phase_for_ids(b,g[np.ix_(B,B)]);values.append(v)
    raw=np.concatenate((values[0]+values[1],values[0]-values[1]))
    for bit in (0,1):walsh(raw[bit*len(ids):(bit+1)*len(ids)])
    quadratic_phase(raw,g[np.ix_(A,A)])
    for j,site in enumerate(A[:-1]):apply_one(raw,u[site],j)
    final=raw.copy();apply_one(final,u[A[-1]],m-1)
    return raw,final

def selfcheck(out):
    root=Path(__file__).resolve().parent;rng=np.random.default_rng(2026103101);rows=[]
    for name in ('induced_12q.npz','induced_24q.npz','induced_32q.npz'):
        d=np.load(root/name);g=d['graph'];u=d['measurement_unitaries'];A=d['A'].tolist();B=d['B'].tolist()
        for trial in range(3):
            ob=rng.integers(0,2,len(B)).tolist();raw,final=cpu_deferred(g,u,A,B,ob)
            ref,pb=conditional(g,u,A,B,ob);ref/=np.sqrt(pb)
            error=float(np.max(abs(final-ref)));l2=float(np.linalg.norm(final-ref));assert error<1e-12 and l2<1e-11
            before=abs(raw.reshape(2,-1))**2;after=abs(final.reshape(2,-1))**2
            marginal_error=float(np.max(abs(before.sum(0)-after.sum(0))))
            # Marginal local probabilities from the auxiliary draw, followed
            # by the post-unitary bit conditional, recover every joint entry.
            recombined=before.sum(0)[None,:]*(after/after.sum(0)[None,:])
            mass_error=float(np.max(abs(recombined-after)))
            assert marginal_error<1e-12 and mass_error<1e-12
            rows.append(dict(fixture=name,trial=trial,max_amplitude_error=error,l2_error=l2,
                maximum_local_marginal_error=marginal_error,maximum_joint_mass_error=mass_error))
    result=dict(status='CPU full-amplitude and all-outcome sampling-factorization checks passed; GPU not yet validated',checks=rows)
    Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selfcheck',action='store_true');p.add_argument('--graph');p.add_argument('--devices',default='0,1');p.add_argument('--samples',type=int,default=4);p.add_argument('--seed',type=int,default=2026103102);p.add_argument('--out',required=True);p.add_argument('--validate',action='store_true');p.add_argument('--fused',action='store_true');p.add_argument('--fusion-bits',type=int,default=5);p.add_argument('--verify-bitstring');a=p.parse_args()
    if a.selfcheck:selfcheck(a.out);return
    if not a.graph:p.error('--graph required')
    d=np.load(a.graph);g=d['graph'];u=d['measurement_unitaries'];A=d['A'].tolist();B=d['B'].tolist();devices=list(map(int,a.devices.split(',')));rng=np.random.default_rng(a.seed)
    if a.validate and len(A)>20:raise ValueError('CPU full-vector comparison limited to 20 conditional qubits')
    if a.verify_bitstring:
        if len(a.verify_bitstring)!=len(g) or set(a.verify_bitstring)-set('01'):raise ValueError('Bad fixed bitstring')
        a.samples=1
    extra={};klass=DeferredFourierSampler
    if a.fused:
        from fourier_fused_sampler import FusedFourierSampler
        klass=type('DeferredFusedSampler',(DeferredGlobalMixin,FusedFourierSampler),{})
        extra={'fusion_bits':a.fusion_bits}
    start=time.perf_counter();engine=klass(g,u,A,B,devices,chunk_mib=1,**extra);setup=time.perf_counter()-start
    metadata=dict(n_qubits=len(g),graph=a.graph,cut_rank=gf2rank(g[np.ix_(A,B)]),devices=devices,dtype='complex64',terminal_unitary_dtype='complex128',setup_seconds=setup,seed=a.seed,samples=a.samples,fused=a.fused,fusion_bits=a.fusion_bits,no_full_vector_peer_exchange=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Numerical implementation of conditional Born sampling; no rigorous full-distribution floating-point certificate')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as f:
        f.write(json.dumps({'metadata':metadata})+'\n');f.flush();print(json.dumps(metadata),flush=True)
        generation_start=time.perf_counter()
        for i in range(a.samples):
            ob=[int(a.verify_bitstring[b]) for b in B] if a.verify_bitstring else rng.integers(0,2,len(B)).tolist()
            start=time.perf_counter();stages=engine.compute(ob)
            if a.verify_bitstring:
                index=sum(int(a.verify_bitstring[q])<<j for j,q in enumerate(A));pa=engine.probability_at(index);_,norm,_,_=engine.masses()
            else:index,pa,norm=engine.draw(rng)
            elapsed=time.perf_counter()-start;bits=[0]*len(g)
            for j,q in enumerate(A):bits[q]=(index>>j)&1
            for j,q in enumerate(B):bits[q]=ob[j]
            rec=dict(sample=i,bitstring_q0_left=''.join(map(str,bits)),probability_raw=pa*2.**(-len(B)),probability=pa/norm*2.**(-len(B)),conditional_norm=norm,elapsed_seconds=elapsed,stages=stages)
            if a.validate:
                ref,pb=conditional(g,u,A,B,ob);ref/=np.sqrt(pb);got=engine.host();err=float(np.linalg.norm(ref-got));rec['oracle_l2_error']=err;rec['oracle_max_abs_error']=float(np.max(abs(ref-got)));assert err<1e-5
            f.write(json.dumps(rec)+'\n');f.flush();print(json.dumps(rec),flush=True)
        summary={'generation_loop_seconds':time.perf_counter()-generation_start,'setup_seconds':setup,'samples':a.samples,'includes_CPU_validation':a.validate}
        f.write(json.dumps({'summary':summary})+'\n');print(json.dumps({'summary':summary}),flush=True)
if __name__=='__main__':main()
