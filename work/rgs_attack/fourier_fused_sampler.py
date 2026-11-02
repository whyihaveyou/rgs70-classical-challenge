"""Multi-bit coalesced transforms for the Fourier-covariant exact sampler.

This separate experimental variant keeps the first version frozen. A tile
contains 32 contiguous lanes and 2**fusion_bits high-bit combinations.
All target butterflies are evaluated in shared memory, so one global read
and write implements up to five high-qubit gates. No approximation is added.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from distributed_sampler import CUDA
from fourier_covariant_sampler import FourierCovariantSampler, cpu_covariant
from graph_sampler import apply_one

FUSION_CUDA=r'''
extern "C" __global__ void fusedfwht(float2*v,int qstart,int count,int lane_bits){
 extern __shared__ float2 s[];
 unsigned tid=threadIdx.x;unsigned lane=tid&((1u<<lane_bits)-1);unsigned target=tid>>lane_bits;
 ull block=(ull)blockIdx.x;int middle=qstart-lane_bits;
 ull base=((block&((1ull<<middle)-1))<<lane_bits)|((block>>middle)<<(qstart+count));
 ull address=base|lane|((ull)target<<qstart);
 s[tid]=v[address];__syncthreads();
 for(int r=0;r<count;r++){
  unsigned mask=1u<<(lane_bits+r);float2 self=s[tid],peer=s[tid^mask];
  float2 val=(tid&mask)?sub(peer,self):add(self,peer);
  __syncthreads();s[tid]=val;__syncthreads();
 }
 v[address]=s[tid];
}
extern "C" __global__ void fusedprod(float2*v,int qstart,int count,int lane_bits,const float2*u){
 extern __shared__ float2 s[];
 unsigned tid=threadIdx.x;unsigned lane=tid&((1u<<lane_bits)-1);unsigned target=tid>>lane_bits;
 ull block=(ull)blockIdx.x;int middle=qstart-lane_bits;
 ull base=((block&((1ull<<middle)-1))<<lane_bits)|((block>>middle)<<(qstart+count));
 ull address=base|lane|((ull)target<<qstart);
 s[tid]=v[address];__syncthreads();
 for(int r=0;r<count;r++){
  unsigned mask=1u<<(lane_bits+r);unsigned bit=(tid&mask)!=0;
  float2 a=s[tid&~mask],b=s[tid|mask];int q=qstart+r;
  float2 val=add(mul(u[4*q+2*bit],a),mul(u[4*q+2*bit+1],b));
  __syncthreads();s[tid]=val;__syncthreads();
 }
 v[address]=s[tid];
}
'''


def high_groups(localbits,fusion_bits=5,start=10):
    return [(q,min(fusion_bits,localbits-q)) for q in range(start,localbits,fusion_bits)]


def tile_addresses(localbits,qstart,count,lane_bits=5):
    """CPU replica of CUDA tile layout, limited to validation-sized vectors."""
    if qstart<lane_bits or qstart+count>localbits or count<1:raise ValueError('invalid tile')
    threads=1<<(lane_bits+count)
    blocks=np.arange(1<<(localbits-lane_bits-count),dtype=np.uint64)[:,None]
    tid=np.arange(threads,dtype=np.uint64)[None,:]
    middle=qstart-lane_bits
    base=((blocks&np.uint64((1<<middle)-1))<<np.uint64(lane_bits))|((blocks>>np.uint64(middle))<<np.uint64(qstart+count))
    return base|(tid&np.uint64((1<<lane_bits)-1))|((tid>>np.uint64(lane_bits))<<np.uint64(qstart))


def cpu_tile_transform(v,qstart,count,lane_bits=5,unitaries=None):
    addresses=tile_addresses(int(np.log2(len(v))),qstart,count,lane_bits)
    s=v[addresses].copy();tid=np.arange(s.shape[1]);
    for r in range(count):
        mask=1<<(lane_bits+r)
        if unitaries is None:
            a=s.copy();peer=s[:,tid^mask];s=np.where((tid&mask)[None,:]!=0,peer-a,peer+a)
        else:
            bit=((tid&mask)!=0).astype(int);u=unitaries[qstart+r]
            s=u[bit,0][None,:]*s[:,tid&~mask]+u[bit,1][None,:]*s[:,tid|mask]
    v[addresses]=s


class FusedFourierSampler(FourierCovariantSampler):
    def __init__(self,g,u,A,B,devices,chunk_mib=256,fusion_bits=5,lane_bits=5):
        if fusion_bits<1 or lane_bits<1 or fusion_bits+lane_bits>10 or lane_bits>10:
            raise ValueError('tile must fit <=1024 threads with positive lane and target bits')
        super().__init__(g,u,A,B,devices,chunk_mib=chunk_mib)
        self.fusion_bits=fusion_bits;self.lane_bits=lane_bits
        self.groups=high_groups(self.localbits,fusion_bits)
        self.fusion_modules=[];self.fusion_kern=[]
        for dev in self.devices:
            with self.cp.cuda.Device(dev):
                mod=self.cp.RawModule(code=CUDA+FUSION_CUDA,options=('--std=c++11',))
                self.fusion_modules.append(mod)
                self.fusion_kern.append({name:mod.get_function(name) for name in ('fusedfwht','fusedprod')})

    def launch_fused(self,i,name,q,count):
        threads=1<<(self.lane_bits+count);blocks=1<<(self.localbits-self.lane_bits-count)
        args=(self.arr[i],np.int32(q),np.int32(count),np.int32(self.lane_bits))
        if name=='fusedprod':args+= (self.small[i]['uA'],)
        with self.cp.cuda.Device(self.devices[i]):
            self.fusion_kern[i][name]((blocks,),(threads,),args,shared_mem=threads*8,stream=self.streams[i])

    def local_walsh(self):
        k=min(10,self.localbits)
        for i in range(self.ng):self.launch(i,'lowfwht',(self.arr[i],np.uint64(self.n),np.int32(k)),grid=self.n>>k,block=1<<k,shared=(1<<k)*8)
        for q,count in self.groups:
            for i in range(self.ng):self.launch_fused(i,'fusedfwht',q,count)
        self.sync()

    def local_product_with_phase(self):
        k=min(10,self.localbits)
        for i in range(self.ng):self.launch(i,'lowprodphase',(self.arr[i],np.uint64(i*self.n),np.uint64(self.n),np.int32(k),self.small[i]['uA'],self.small[i]['qA'],np.int32(self.m)),grid=self.n>>k,block=1<<k,shared=(1<<k)*8)
        for q,count in self.groups:
            for i in range(self.ng):self.launch_fused(i,'fusedprod',q,count)
        self.sync()

    def compute(self,outB):
        import time
        cp=self.cp;stage={};start=time.perf_counter();w=self.u[self.B,np.array(outB),:].astype(np.complex64)
        for i,dev in enumerate(self.devices):
            with cp.cuda.Device(dev):self.small[i]['w'].set(w,stream=self.streams[i])
            self.launch(i,'initcov',(self.arr[i],np.uint64(i*self.n),np.uint64(self.n),self.small[i]['w'],self.small[i]['qB'],self.small[i]['inverse_transpose_rows'],np.int32(self.m),np.float32(2.**(-self.m/2))))
        self.sync();stage['initialize_covariant']=time.perf_counter()-start;start=time.perf_counter()
        self.local_walsh()
        for q in range(self.localbits,self.m):self.global_gate(q,np.array([[1,1],[1,-1]],np.complex64))
        stage['walsh_fused']=time.perf_counter()-start;stage['linear_permutation_phase']=0.0;start=time.perf_counter()
        self.local_product_with_phase()
        for q in range(self.localbits,self.m):self.global_gate(q,self.u[self.A[q]])
        stage['product_measurement_fused']=time.perf_counter()-start
        return stage


def selfcheck(out):
    rng=np.random.default_rng(2026103101);checks=[]
    # Exercise complete and short final groups; each index must be visited once.
    for L in (11,12,15,16,19,20):
        v=(rng.standard_normal(1<<L)+1j*rng.standard_normal(1<<L)).astype(np.complex128)
        u=[]
        for _ in range(L):
            unitary,_=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)));u.append(unitary)
        u=np.asarray(u)
        for q,count in high_groups(L):
            address=tile_addresses(L,q,count)
            assert np.array_equal(np.sort(address.ravel()),np.arange(1<<L))
            # In every warp, addresses must cover 32 consecutive complex values.
            assert np.all(np.diff(address.reshape(-1,32),axis=1)==1)
            for kind in ('walsh','product'):
                ref=v.copy();got=v.copy()
                for target in range(q,q+count):apply_one(ref,np.array([[1,1],[1,-1]],complex) if kind=='walsh' else u[target],target)
                cpu_tile_transform(got,q,count,unitaries=None if kind=='walsh' else u)
                error=float(np.max(np.abs(got-ref)));assert error<1e-12
                checks.append(dict(localbits=L,qstart=q,count=count,kind=kind,max_error=error,complete_index_partition=True,coalesced_warps=True))
    # A full-state check uses the actual fixture and a physically complete
    # distributed ordering, including the final global gate on two shards.
    from fourier_covariant_sampler import gf2_inverse,map_indices,phase_for_ids
    from graph_sampler import conditional,quadratic_phase
    root=Path(__file__).resolve().parent;full=[]
    for name in ('induced_24q.npz','induced_32q.npz','induced_40q.npz'):
        d=np.load(root/name);g=d['graph'];u=d['measurement_unitaries'];A=d['A'].tolist();B=d['B'].tolist();m=len(A);L=m-1
        ob=rng.integers(0,2,m).tolist();ids=np.arange(1<<m,dtype=np.uint64);b=map_indices(ids,gf2_inverse(g[np.ix_(B,A)]).T)
        v=np.full(1<<m,2.**(-m/2),complex)
        for j,site in enumerate(B):v*=u[site,ob[j],((b>>np.uint64(j))&1).astype(int)]
        v*=phase_for_ids(b,g[np.ix_(B,B)])
        for shard in np.split(v,2):
            for q in range(min(10,L)):apply_one(shard,np.array([[1,1],[1,-1]],complex),q)
            for q,count in high_groups(L):cpu_tile_transform(shard,q,count)
        apply_one(v,np.array([[1,1],[1,-1]],complex),L)
        quadratic_phase(v,g[np.ix_(A,A)])
        for shard in np.split(v,2):
            for q in range(min(10,L)):apply_one(shard,u[A[q]],q)
            for q,count in high_groups(L):cpu_tile_transform(shard,q,count,unitaries=u[A])
        apply_one(v,u[A[L]],L)
        ref,pb=conditional(g,u,A,B,ob);ref/=np.sqrt(pb)
        err=float(np.max(np.abs(v-ref)));l2=float(np.linalg.norm(v-ref));assert err<1e-12 and l2<1e-11
        full.append(dict(fixture=name,max_abs_error=err,l2_error=l2))
    summary=dict(status='CPU indexing and complete-amplitude tests passed; GPU validation pending',tile_tests=checks,full_amplitude_tests=full,full35_local_groups=high_groups(34),local_readwrite_passes_per_transform=1+len(high_groups(34)))
    Path(out).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selfcheck',action='store_true',required=True);p.add_argument('--out',required=True);a=p.parse_args();selfcheck(a.out)
