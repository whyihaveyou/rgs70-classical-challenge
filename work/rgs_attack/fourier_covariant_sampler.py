"""Exact Fourier-covariant conditional sampler; no statevector permutation.

Let M = G[B,A] and f(b)=(-1)^qB(b) prod_j U_B[j][outB_j,b_j].
The old implementation evaluates H(f)(Mx). This one evaluates H(g)(x),
where g(z)=f(M^{-T}z), using the exact binary Fourier covariance identity.
The normalization is fused into initialization and qA into the first local
measurement kernel. It keeps one distributed state plus bounded P2P scratch.
The existing sampler is imported unchanged; this is a separate experimental
implementation, not a replacement for its frozen results.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from distributed_sampler import CUDA, DistributedSampler
from graph_sampler import conditional, gf2rank, walsh, quadratic_phase, apply_one


def gf2_inverse(matrix):
    matrix=np.asarray(matrix,dtype=np.uint8)
    n=len(matrix)
    if matrix.shape!=(n,n):raise ValueError('square matrix required')
    aug=np.concatenate([matrix.copy(),np.eye(n,dtype=np.uint8)],axis=1)
    for col in range(n):
        pivots=np.flatnonzero(aug[col:,col])+col
        if not len(pivots):raise ValueError('singular GF(2) matrix')
        p=int(pivots[0]);aug[[col,p]]=aug[[p,col]]
        for row in range(n):
            if row!=col and aug[row,col]:aug[row]^=aug[col]
    inverse=aug[:,n:].copy()
    assert np.array_equal((matrix@inverse)&1,np.eye(n,dtype=np.uint8))
    return inverse


def row_masks(matrix):
    return np.asarray([sum(1<<int(j) for j in np.flatnonzero(row)) for row in matrix],dtype=np.uint64)


def parity64(values):
    y=np.asarray(values,dtype=np.uint64).copy()
    for shift in (32,16,8,4):y^=y>>np.uint64(shift)
    return (np.uint64(0x6996)>>(y&np.uint64(15)))&np.uint64(1)


def map_indices(ids,matrix):
    out=np.zeros_like(np.asarray(ids,dtype=np.uint64))
    for j,mask in enumerate(row_masks(matrix)):
        out|=parity64(np.asarray(ids,dtype=np.uint64)&mask)<<np.uint64(j)
    return out


def phase_for_ids(ids,graph):
    ids=np.asarray(ids,dtype=np.uint64);parity=np.zeros_like(ids)
    for i,row in enumerate(graph):
        mask=np.uint64(sum(1<<int(j) for j in np.flatnonzero(row[i+1:])+i+1))
        parity^=((ids>>np.uint64(i))&1)*parity64(ids&mask)
    return 1.0-2.0*parity.astype(float)


def cpu_covariant(g,u,A,B,outB):
    """Normalized conditional state, independent of the old permutation code."""
    m=len(A);M=g[np.ix_(B,A)];N=gf2_inverse(M).T
    z=np.arange(1<<m,dtype=np.uint64);b=map_indices(z,N)
    v=np.full(1<<m,2.**(-m/2),dtype=np.complex128)
    for j,site in enumerate(B):v*=u[site,outB[j],((b>>np.uint64(j))&1).astype(int)]
    v*=phase_for_ids(b,g[np.ix_(B,B)])
    walsh(v)
    quadratic_phase(v,g[np.ix_(A,A)])
    for j,site in enumerate(A):apply_one(v,u[site],j)
    return v


EXTRA_CUDA=r'''
extern "C" __global__ void initcov(float2*v,ull off,ull n,const float2*w,const ull*qm,const ull*invTrows,int m,float scale){
 for(ull j=(ull)blockIdx.x*blockDim.x+threadIdx.x;j<n;j+=(ull)gridDim.x*blockDim.x){
  ull id=j+off,b=0;
  for(int k=0;k<m;k++)b|=((ull)(__popcll(id&invTrows[k])&1))<<k;
  float2 z=make_float2(scale,0);unsigned par=0;
  for(int k=0;k<m;k++){unsigned bit=(b>>k)&1;z=mul(z,w[2*k+bit]);if(bit)par^=(__popcll(b&qm[k])&1);}
  if(par){z.x=-z.x;z.y=-z.y;}v[j]=z;
 }
}
extern "C" __global__ void lowprodphase(float2*v,ull off,ull n,int k,const float2*u,const ull*qm,int m){
 extern __shared__ float2 s[];ull base=(ull)blockIdx.x*(1ull<<k);unsigned i=threadIdx.x;ull id=off+base+i;unsigned par=0;
 for(int q=0;q<m;q++)if((id>>q)&1)par^=(__popcll(id&qm[q])&1);
 float2 value=v[base+i];if(par){value.x=-value.x;value.y=-value.y;}s[i]=value;__syncthreads();
 for(int q=0;q<k;q++){unsigned b=(i>>q)&1;float2 a=s[i&~(1u<<q)],c=s[i|(1u<<q)];float2 val=add(mul(u[4*q+2*b],a),mul(u[4*q+2*b+1],c));__syncthreads();s[i]=val;__syncthreads();}
 v[base+i]=s[i];
}
'''


class FourierCovariantSampler(DistributedSampler):
    def __init__(self,g,u,A,B,devices,chunk_mib=256):
        # host_spill=True only selects bounded scratch in the base constructor.
        # This subclass never calls its permutation(), hence never allocates
        # the 256-GiB host vector or performs random peer gathers.
        super().__init__(g,u,A,B,devices,host_spill=True,chunk_mib=chunk_mib)
        cp=self.cp
        self.cross_matrix=np.asarray(g[np.ix_(B,A)],np.uint8)
        self.inverse_transpose=gf2_inverse(self.cross_matrix).T
        inverse_rows=row_masks(self.inverse_transpose)
        self.covariant_modules=[]
        for i,dev in enumerate(self.devices):
            with cp.cuda.Device(dev):
                mod=cp.RawModule(code=CUDA+EXTRA_CUDA,options=('--std=c++11',))
                self.covariant_modules.append(mod)
                self.kern[i]['initcov']=mod.get_function('initcov')
                self.kern[i]['lowprodphase']=mod.get_function('lowprodphase')
                self.small[i]['inverse_transpose_rows']=cp.asarray(inverse_rows)
                cp.cuda.runtime.deviceSynchronize()

    def permutation(self):
        raise RuntimeError('Fourier-covariant implementation must not permute statevectors')

    def compute(self,outB):
        cp=self.cp;stage={};start=time.perf_counter()
        w=self.u[self.B,np.array(outB),:].astype(np.complex64)
        for i,dev in enumerate(self.devices):
            with cp.cuda.Device(dev):self.small[i]['w'].set(w,stream=self.streams[i])
            self.launch(i,'initcov',(self.arr[i],np.uint64(i*self.n),np.uint64(self.n),
                self.small[i]['w'],self.small[i]['qB'],self.small[i]['inverse_transpose_rows'],
                np.int32(self.m),np.float32(2.**(-self.m/2))))
        self.sync();stage['initialize_covariant']=time.perf_counter()-start;start=time.perf_counter();k=min(10,self.localbits)
        for i in range(self.ng):self.launch(i,'lowfwht',(self.arr[i],np.uint64(self.n),np.int32(k)),grid=self.n>>k,block=1<<k,shared=(1<<k)*8)
        for q in range(k,self.localbits):
            for i in range(self.ng):self.launch(i,'localfwht',(self.arr[i],np.uint64(self.n),np.int32(q)))
        self.sync()
        for q in range(self.localbits,self.m):self.global_gate(q,np.array([[1,1],[1,-1]],np.complex64))
        stage['walsh']=time.perf_counter()-start;stage['linear_permutation_phase']=0.0;start=time.perf_counter()
        for i in range(self.ng):self.launch(i,'lowprodphase',(self.arr[i],np.uint64(i*self.n),np.uint64(self.n),np.int32(k),self.small[i]['uA'],self.small[i]['qA'],np.int32(self.m)),grid=self.n>>k,block=1<<k,shared=(1<<k)*8)
        for q in range(k,self.localbits):
            for i in range(self.ng):self.launch(i,'localprod',(self.arr[i],np.uint64(self.n),np.int32(q),self.small[i]['uA']))
        self.sync()
        for q in range(self.localbits,self.m):self.global_gate(q,self.u[self.A[q]])
        stage['product_measurement_with_phase']=time.perf_counter()-start
        return stage


def selfcheck(out):
    rng=np.random.default_rng(2026103007);records=[]
    root=Path(__file__).resolve().parent
    for name in ('induced_12q.npz','induced_24q.npz','induced_32q.npz'):
        d=np.load(root/name);g=d['graph'];u=d['measurement_unitaries'];A=d['A'].tolist();B=d['B'].tolist()
        for trial in range(3):
            ob=rng.integers(0,2,len(B)).tolist();v=cpu_covariant(g,u,A,B,ob);ref,pb=conditional(g,u,A,B,ob);ref/=np.sqrt(pb)
            error=float(np.max(np.abs(v-ref)));l2=float(np.linalg.norm(v-ref))
            assert error<1e-12 and l2<1e-11
            records.append(dict(fixture=name,trial=trial,max_abs_error=error,l2_error=l2,norm=float(np.vdot(v,v).real)))
    matrices=[]
    for name in ('rgs_graph.npz','rgs_graph_alt_partition.npz'):
        d=np.load(root/name);g=d['graph'];A=d['A'].tolist();B=d['B'].tolist();m=len(A)
        M=g[np.ix_(B,A)];N=gf2_inverse(M).T
        ids=np.concatenate([np.zeros(1,np.uint64),np.array([1<<j for j in range(m)],np.uint64),rng.integers(0,1<<m,10000,dtype=np.uint64)])
        b=map_indices(ids,N)
        assert np.array_equal(map_indices(b,M.T),ids)
        x=rng.integers(0,1<<m,len(ids),dtype=np.uint64)
        assert np.array_equal(parity64(map_indices(x,M)&b),parity64(x&ids))
        # Exhaustively verify the Fourier identity on a restricted 10-D test:
        # all 2^10 output characters against all 2^10 input basis vectors,
        # embedded in the actual 35-D mappings (no 2^35 allocation).
        basis_inputs=rng.integers(0,1<<m,10,dtype=np.uint64)
        basis_outputs=rng.integers(0,1<<m,10,dtype=np.uint64)
        sub=np.arange(1024,dtype=np.uint64);iz=np.zeros(1024,np.uint64);ix=np.zeros(1024,np.uint64)
        for j in range(10):
            iz^=((sub>>np.uint64(j))&1)*basis_inputs[j]
            ix^=((sub>>np.uint64(j))&1)*basis_outputs[j]
        lhs=parity64(map_indices(ix,M)[:,None]&map_indices(iz,N)[None,:])
        rhs=parity64(ix[:,None]&iz[None,:])
        assert np.array_equal(lhs,rhs)
        matrices.append(dict(fixture=name,m=m,rank=gf2rank(M),inverse_transpose_sha256=hashlib.sha256(N.tobytes()).hexdigest(),random_and_basis_vectors=len(ids),character_pairs_checked=1024**2,identity='(Mx).(M^-T z)=x.z mod 2',all_passed=True))
    report=dict(status='CPU full-amplitude and actual 35x35 algebra checks passed; CUDA kernels still require GPU oracle checks',full_amplitude_checks=records,matrix_checks=matrices)
    Path(out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selfcheck',action='store_true');p.add_argument('--graph');p.add_argument('--devices',default='0,1');p.add_argument('--samples',type=int,default=4);p.add_argument('--seed',type=int,default=20261030);p.add_argument('--chunk-mib',type=int,default=256);p.add_argument('--validate',action='store_true');p.add_argument('--verify-bitstring');p.add_argument('--out',required=True);args=p.parse_args()
    if args.selfcheck:selfcheck(args.out);return
    if not args.graph:p.error('--graph required')
    data=np.load(args.graph);g=data['graph'];u=data['measurement_unitaries'];A=data['A'].tolist();B=data['B'].tolist();devices=[int(x) for x in args.devices.split(',')];rng=np.random.default_rng(args.seed)
    if args.validate and len(A)>20:raise ValueError('CPU oracle limited to <=20 conditional qubits')
    if args.verify_bitstring:
        if len(args.verify_bitstring)!=len(g) or set(args.verify_bitstring)-set('01'):raise ValueError('Bad fixed bitstring')
        args.samples=1
    start=time.perf_counter();engine=FourierCovariantSampler(g,u,A,B,devices,chunk_mib=args.chunk_mib);setup=time.perf_counter()-start
    meta=dict(graph=args.graph,n_qubits=len(g),cut_rank=gf2rank(g[np.ix_(A,B)]),devices=devices,dtype='complex64',setup_seconds=setup,host_statevector_spill=False,chunk_mib=args.chunk_mib,seed=args.seed,method='Fourier covariance: H[f](Mx)=H[f o M^-T](x)',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='experimental; no claim of quantum advantage supersession')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as fh:
        fh.write(json.dumps({'metadata':meta})+'\n');fh.flush();print(json.dumps(meta),flush=True)
        for sample in range(args.samples):
            ob=([int(args.verify_bitstring[b]) for b in B] if args.verify_bitstring else rng.integers(0,2,len(B)).tolist())
            start=time.perf_counter();stages=engine.compute(ob)
            if args.verify_bitstring:
                index=sum(int(args.verify_bitstring[a])<<j for j,a in enumerate(A));pa=engine.probability_at(index);_,norm,_,_=engine.masses()
            else:index,pa,norm=engine.draw(rng)
            elapsed=time.perf_counter()-start;bits=[0]*len(g)
            for j,a in enumerate(A):bits[a]=(index>>j)&1
            for j,b in enumerate(B):bits[b]=ob[j]
            rec=dict(sample=sample,bitstring_q0_left=''.join(map(str,bits)),probability=(pa/norm)*2.**(-len(B)),probability_raw=pa*2.**(-len(B)),xeb_contribution=2.**len(A)*pa-1,xeb_sampling_q=2.**len(A)*pa/norm-1,conditional_norm=norm,elapsed_seconds=elapsed,stages=stages)
            if args.validate:
                ref,pb=conditional(g,u,A,B,ob);ref/=np.sqrt(pb);got=engine.host();rec['oracle_max_abs_error']=float(np.max(np.abs(got-ref)));rec['oracle_l2_error']=float(np.linalg.norm(got-ref));assert rec['oracle_l2_error']<1e-5
            fh.write(json.dumps(rec)+'\n');fh.flush();print(json.dumps(rec),flush=True)


if __name__=='__main__':main()
