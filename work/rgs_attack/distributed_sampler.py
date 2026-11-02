"""CuPy multi-GPU exact conditional graph-state sampler.
Requires a balanced partition with full binary cut rank. No truncation or MCMC.
Native c64 arithmetic; verify against the CPU oracle before full-scale use.
"""
import argparse,json,math,time
from pathlib import Path
import numpy as np
from graph_sampler import conditional,gf2rank
CUDA=r'''
typedef unsigned long long ull;
__device__ __forceinline__ float2 add(float2 a,float2 b){return make_float2(a.x+b.x,a.y+b.y);}
__device__ __forceinline__ float2 sub(float2 a,float2 b){return make_float2(a.x-b.x,a.y-b.y);}
__device__ __forceinline__ float2 mul(float2 a,float2 b){return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
extern "C" __global__ void initv(float2*v,ull off,ull n,const float2*w,const ull*qm,int m){
 for(ull j=(ull)blockIdx.x*blockDim.x+threadIdx.x;j<n;j+=(ull)gridDim.x*blockDim.x){
  ull id=j+off;float2 z=make_float2(1,0);unsigned par=0;
  for(int k=0;k<m;k++){unsigned b=(id>>k)&1;z=mul(z,w[2*k+b]);if(b)par^=(__popcll(id&qm[k])&1);}
  if(par){z.x=-z.x;z.y=-z.y;}v[j]=z;
 }
}
extern "C" __global__ void lowfwht(float2*v,ull n,int k){
 extern __shared__ float2 s[];ull base=(ull)blockIdx.x*(1ull<<k);unsigned i=threadIdx.x;
 s[i]=v[base+i];__syncthreads();
 for(int q=0;q<k;q++){unsigned b=(i>>q)&1;float2 self=s[i],peer=s[i^(1u<<q)];float2 val=b?sub(peer,self):add(self,peer);__syncthreads();s[i]=val;__syncthreads();}
 v[base+i]=s[i];
}
extern "C" __global__ void localfwht(float2*v,ull n,int q){
 for(ull i=(ull)blockIdx.x*blockDim.x+threadIdx.x;i<n/2;i+=(ull)gridDim.x*blockDim.x){
  ull j=((i>>q)<<(q+1))|(i&((1ull<<q)-1));float2 a=v[j],b=v[j|(1ull<<q)];v[j]=add(a,b);v[j|(1ull<<q)]=sub(a,b);
 }
}
extern "C" __global__ void globalgate(float2*v,const float2*p,ull n,int bit,float2 u0,float2 u1){
 for(ull i=(ull)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(ull)gridDim.x*blockDim.x){float2 a=bit?p[i]:v[i],b=bit?v[i]:p[i];v[i]=add(mul(u0,a),mul(u1,b));}
}
extern "C" __global__ void permute(float2*out,const ull*ptrs,ull off,ull n,int localbits,const ull*cols,const ull*qm,int m,float scale){
 for(ull j=(ull)blockIdx.x*blockDim.x+threadIdx.x;j<n;j+=(ull)gridDim.x*blockDim.x){
  ull id=j+off,y=0;unsigned par=0;
  for(int k=0;k<m;k++){if((id>>k)&1){y^=cols[k];par^=(__popcll(id&qm[k])&1);}}
  const float2*src=(const float2*)ptrs[y>>localbits];float2 z=src[y&((1ull<<localbits)-1)];float s=par?-scale:scale;out[j]=make_float2(z.x*s,z.y*s);
 }
}
extern "C" __global__ void lowprod(float2*v,ull n,int k,const float2*u){
 extern __shared__ float2 s[];ull base=(ull)blockIdx.x*(1ull<<k);unsigned i=threadIdx.x;s[i]=v[base+i];__syncthreads();
 for(int q=0;q<k;q++){unsigned b=(i>>q)&1;float2 a=s[i&~(1u<<q)],c=s[i|(1u<<q)];float2 val=add(mul(u[4*q+2*b],a),mul(u[4*q+2*b+1],c));__syncthreads();s[i]=val;__syncthreads();}
 v[base+i]=s[i];
}
extern "C" __global__ void localprod(float2*v,ull n,int q,const float2*u){
 for(ull i=(ull)blockIdx.x*blockDim.x+threadIdx.x;i<n/2;i+=(ull)gridDim.x*blockDim.x){ull j=((i>>q)<<(q+1))|(i&((1ull<<q)-1));float2 a=v[j],b=v[j|(1ull<<q)];v[j]=add(mul(u[4*q],a),mul(u[4*q+1],b));v[j|(1ull<<q)]=add(mul(u[4*q+2],a),mul(u[4*q+3],b));}
}
extern "C" __global__ void blockmass(const float2*v,double*out,ull n,int tilebits){
 extern __shared__ double smass[];ull start=(ull)blockIdx.x<<tilebits;ull end=start+(1ull<<tilebits);if(end>n)end=n;double z=0;
 for(ull i=start+threadIdx.x;i<end;i+=blockDim.x){float2 a=v[i];z+=(double)a.x*a.x+(double)a.y*a.y;}smass[threadIdx.x]=z;__syncthreads();
 for(int off=blockDim.x/2;off;off/=2){if(threadIdx.x<off)smass[threadIdx.x]+=smass[threadIdx.x+off];__syncthreads();}if(threadIdx.x==0)out[blockIdx.x]=smass[0];
}
'''

def qmasks(g):return np.array([sum(1<<int(j) for j in np.flatnonzero(g[i,i+1:])+i+1) for i in range(len(g))],dtype=np.uint64)

class DistributedSampler:
 def __init__(self,g,u,A,B,devices,host_spill=False,chunk_mib=256):
  import cupy as cp
  self.cp=cp;self.g=g;self.u=u;self.A=A;self.B=B;self.devices=devices;self.m=len(A);self.ng=len(devices)
  assert len(A)==len(B) and self.ng&(self.ng-1)==0
  assert gf2rank(g[np.ix_(A,B)])==self.m,'B marginal is not uniform'
  self.localbits=self.m-int(math.log2(self.ng));self.n=1<<self.localbits;self.blocks=min(65535,(self.n+255)//256)
  self.host_spill=host_spill;self.buffer_elems=min(self.n,chunk_mib*1024**2//8) if host_spill else self.n
  self.arr=[];self.buf=[];self.streams=[];self.kern=[];self.small=[]
  cols=np.array([sum(1<<j for j,b in enumerate(B) if g[a,b]) for a in A],np.uint64)
  for dev in devices:
   with cp.cuda.Device(dev):
    for peer in devices:
     if peer!=dev:
      if not cp.cuda.runtime.deviceCanAccessPeer(dev,peer):raise RuntimeError(f'No P2P {dev}->{peer}')
      try:cp.cuda.runtime.deviceEnablePeerAccess(peer)
      except cp.cuda.runtime.CUDARuntimeError as e:
       if e.status!=704:raise
    free,total=cp.cuda.runtime.memGetInfo();need=(self.n+self.buffer_elems)*8+64*1024**2
    if free<need:raise MemoryError(f'GPU {dev}: need {need}, free {free}')
    self.streams.append(cp.cuda.Stream(non_blocking=True));self.arr.append(cp.empty(self.n,cp.complex64));self.buf.append(cp.empty(self.buffer_elems,cp.complex64))
    mod=cp.RawModule(code=CUDA,options=('--std=c++11',));self.kern.append({x:mod.get_function(x) for x in ['initv','lowfwht','localfwht','globalgate','permute','lowprod','localprod','blockmass']})
    self.small.append({'qA':cp.asarray(qmasks(g[np.ix_(A,A)])),'qB':cp.asarray(qmasks(g[np.ix_(B,B)])),'cols':cp.asarray(cols),'uA':cp.asarray(u[A].astype(np.complex64)),'w':cp.empty((self.m,2),cp.complex64),'ptrs':cp.empty(self.ng,cp.uint64)})
  for dev in devices:
   with cp.cuda.Device(dev):cp.cuda.runtime.deviceSynchronize()
  self.sync()
 def sync(self):
  for dev,s in zip(self.devices,self.streams):
   with self.cp.cuda.Device(dev):s.synchronize()
 def launch(self,i,name,args,grid=None,block=256,shared=0):
  cp=self.cp
  with cp.cuda.Device(self.devices[i]):self.kern[i][name]((grid or self.blocks,),(block,),args,shared_mem=shared,stream=self.streams[i])
 def global_gate(self,q,mats):
  cp=self.cp;bit=q-self.localbits
  for start in range(0,self.n,self.buffer_elems):
   count=min(self.buffer_elems,self.n-start)
   for i,dev in enumerate(self.devices):
    peer=i^(1<<bit)
    with cp.cuda.Device(dev):cp.cuda.runtime.memcpyPeerAsync(self.buf[i].data.ptr,dev,self.arr[peer].data.ptr+start*8,self.devices[peer],count*8,self.streams[i].ptr)
   self.sync()
   for i in range(self.ng):
    b=(i>>bit)&1;row=np.asarray(mats[b],dtype=np.complex64)
    self.launch(i,'globalgate',(self.arr[i][start:start+count],self.buf[i],np.uint64(count),np.int32(b),row[0],row[1]))
   self.sync()
 def permutation(self):
  cp=self.cp;ptrs=np.array([x.data.ptr for x in self.arr],np.uint64)
  for i,dev in enumerate(self.devices):
   with cp.cuda.Device(dev):self.small[i]['ptrs'].set(ptrs,stream=self.streams[i])
  host=np.empty(self.ng*self.n,np.complex64) if self.host_spill else None
  for start in range(0,self.n,self.buffer_elems):
   count=min(self.buffer_elems,self.n-start)
   for i in range(self.ng):
    self.launch(i,'permute',(self.buf[i],self.small[i]['ptrs'],np.uint64(i*self.n+start),np.uint64(count),np.int32(self.localbits),self.small[i]['cols'],self.small[i]['qA'],np.int32(self.m),np.float32(2.**(-self.m/2))))
   self.sync()
   if self.host_spill:
    for i,dev in enumerate(self.devices):
     with cp.cuda.Device(dev):self.buf[i][:count].get(out=host[i*self.n+start:i*self.n+start+count],stream=self.streams[i],blocking=True)
  if self.host_spill:
   # No source may be overwritten until every target amplitude was read.
   self.sync()
   for start in range(0,self.n,self.buffer_elems):
    count=min(self.buffer_elems,self.n-start)
    for i,dev in enumerate(self.devices):
     with cp.cuda.Device(dev):self.arr[i][start:start+count].set(host[i*self.n+start:i*self.n+start+count],stream=self.streams[i])
    self.sync()
   del host
  else:self.arr,self.buf=self.buf,self.arr
 def compute(self,outB):
  cp=self.cp;stage={};start=time.perf_counter();w=self.u[self.B,np.array(outB),:].astype(np.complex64)
  for i,dev in enumerate(self.devices):
   with cp.cuda.Device(dev):self.small[i]['w'].set(w,stream=self.streams[i])
   self.launch(i,'initv',(self.arr[i],np.uint64(i*self.n),np.uint64(self.n),self.small[i]['w'],self.small[i]['qB'],np.int32(self.m)))
  self.sync();stage['initialize']=time.perf_counter()-start;start=time.perf_counter();k=min(10,self.localbits)
  for i in range(self.ng):self.launch(i,'lowfwht',(self.arr[i],np.uint64(self.n),np.int32(k)),grid=self.n>>k,block=1<<k,shared=(1<<k)*8)
  for q in range(k,self.localbits):
   for i in range(self.ng):self.launch(i,'localfwht',(self.arr[i],np.uint64(self.n),np.int32(q)))
  self.sync()
  for q in range(self.localbits,self.m):self.global_gate(q,np.array([[1,1],[1,-1]],np.complex64))
  stage['walsh']=time.perf_counter()-start;start=time.perf_counter();self.permutation();stage['linear_permutation_phase']=time.perf_counter()-start;start=time.perf_counter()
  for i in range(self.ng):self.launch(i,'lowprod',(self.arr[i],np.uint64(self.n),np.int32(k),self.small[i]['uA']),grid=self.n>>k,block=1<<k,shared=(1<<k)*8)
  for q in range(k,self.localbits):
   for i in range(self.ng):self.launch(i,'localprod',(self.arr[i],np.uint64(self.n),np.int32(q),self.small[i]['uA']))
  self.sync()
  for q in range(self.localbits,self.m):self.global_gate(q,self.u[self.A[q]])
  stage['product_measurement']=time.perf_counter()-start
  return stage
 def masses(self):
  cp=self.cp;tilebits=min(20,self.localbits);nmass=(self.n+(1<<tilebits)-1)>>tilebits;massgpu=[]
  for i,dev in enumerate(self.devices):
   with cp.cuda.Device(dev):mg=cp.empty(nmass,cp.float64);massgpu.append(mg)
   self.launch(i,'blockmass',(self.arr[i],mg,np.uint64(self.n),np.int32(tilebits)),grid=nmass,shared=256*8)
  self.sync();mass=np.concatenate([cp.asnumpy(x) for x in massgpu]);norm=float(mass.sum())
  if not np.isfinite(norm) or abs(norm-1)>1e-3:raise ArithmeticError(f'Conditional norm check failed: {norm}')
  return mass,norm,tilebits,nmass
 def draw(self,rng):
  cp=self.cp;mass,norm,tilebits,nmass=self.masses();block=int(np.searchsorted(np.cumsum(mass),rng.random()*norm,side='right'));gpu=block//nmass;chunk=block%nmass
  with cp.cuda.Device(self.devices[gpu]):v=cp.asnumpy(self.arr[gpu][chunk*(1<<tilebits):min((chunk+1)*(1<<tilebits),self.n)])
  p=v.real.astype(float)**2+v.imag.astype(float)**2;within=int(np.searchsorted(np.cumsum(p),rng.random()*p.sum(),side='right'));index=gpu*self.n+chunk*(1<<tilebits)+within
  return index,float(p[within]),norm
 def probability_at(self,index):
  gpu=index//self.n;local=index%self.n
  with self.cp.cuda.Device(self.devices[gpu]):v=self.cp.asnumpy(self.arr[gpu][local:local+1])[0]
  return float(v.real)**2+float(v.imag)**2
 def host(self):return np.concatenate([self.cp.asnumpy(a) for a in self.arr])

def main():
 p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--devices',default='0');p.add_argument('--samples',type=int,default=1);p.add_argument('--seed',type=int,default=1);p.add_argument('--out',required=True);p.add_argument('--validate',action='store_true');p.add_argument('--host-spill',action='store_true');p.add_argument('--chunk-mib',type=int,default=256);p.add_argument('--verify-bitstring',help='Fixed original-qubit bitstring, qubit 0 LEFTMOST; computes probability instead of sampling');args=p.parse_args()
 data=np.load(args.graph);g=data['graph'];u=data['measurement_unitaries'];A=data['A'].tolist();B=data['B'].tolist();devices=[int(x) for x in args.devices.split(',')];rng=np.random.default_rng(args.seed);t=time.perf_counter();engine=DistributedSampler(g,u,A,B,devices,host_spill=args.host_spill,chunk_mib=args.chunk_mib);setup=time.perf_counter()-t
 meta={'graph':args.graph,'n_qubits':len(g),'cut_rank':gf2rank(g[np.ix_(A,B)]),'devices':devices,'dtype':'complex64','setup_seconds':setup,'host_spill':args.host_spill,'chunk_mib':args.chunk_mib,'seed':args.seed,'status':'experimental; no claim of quantum advantage supersession'}
 out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w') as f:
  f.write(json.dumps({'metadata':meta})+'\n');f.flush();print(json.dumps(meta),flush=True)
  if args.verify_bitstring:
   if len(args.verify_bitstring)!=len(g) or set(args.verify_bitstring)-set('01'):raise ValueError('Bad fixed bitstring')
   args.samples=1
  for s in range(args.samples):
   ob=([int(args.verify_bitstring[b]) for b in B] if args.verify_bitstring else rng.integers(0,2,len(B)).tolist())
   start=time.perf_counter();stages=engine.compute(ob)
   if args.verify_bitstring:
    index=sum(int(args.verify_bitstring[a])<<j for j,a in enumerate(A));pa=engine.probability_at(index);_,norm,_,_=engine.masses()
   else:index,pa,norm=engine.draw(rng)
   elapsed=time.perf_counter()-start;bits=[0]*len(g)
   for j,a in enumerate(A):bits[a]=(index>>j)&1
   for j,b in enumerate(B):bits[b]=ob[j]
   rec={'sample':s,'bitstring_q0_left':''.join(map(str,bits)),'probability':(pa/norm)*2.**(-len(B)),'probability_raw':pa*2.**(-len(B)),'xeb_contribution':2.**len(A)*pa-1,'xeb_sampling_q':2.**len(A)*pa/norm-1,'conditional_norm':norm,'elapsed_seconds':elapsed,'stages':stages}
   if args.validate:
    if len(A)>20:raise ValueError('CPU oracle limited to <=20 conditional qubits')
    ref,pb=conditional(g,u,A,B,ob);ref/=np.sqrt(pb);got=engine.host();rec['oracle_max_abs_error']=float(np.max(np.abs(got-ref)));rec['oracle_l2_error']=float(np.linalg.norm(got-ref));assert rec['oracle_l2_error']<1e-5
   f.write(json.dumps(rec)+'\n');f.flush();print(json.dumps(rec),flush=True)
if __name__=='__main__':main()
