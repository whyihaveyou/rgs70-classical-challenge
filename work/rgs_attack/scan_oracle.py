"""Adapter for Kim Renaud's Apache-2.0 dcs-rankwidth GPU scan on QAT RGS graph.
GPU kernel unmodified; scan_plan has a uint64-to-int64 indexing compatibility fix. No T injection needed.
"""
import argparse,importlib.util,json,math,sys,time
from pathlib import Path
import numpy as np

def loadmodule(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

class ScanOracle:
 def __init__(self,g,u,order,source,device=0,gpu=True,fp64=False):
  self.n=len(g);self.u=u;self.order=order
  self.plan=loadmodule('upstream_scan_plan',Path(source)/'scan_plan.py')
  self.edges=[(i,j) for i in range(self.n) for j in range(i+1,self.n) if g[i,j]]
  self.steps,self.ranks=self.plan.make_plan(self.n,self.edges,order)
  self.fut=self.plan.future_masks(self.n,self.edges,order)
  self.rmax=max(self.ranks);self.tables=[self.plan.byte_tables(x.Lcols,x.r_out) for x in self.steps];self.combos=[np.array(self.plan.kernel_combos(x.kernel),np.uint64) for x in self.steps]
  self.gpu=gpu
  if gpu:
   kernel=loadmodule('upstream_gpu_kernel',Path(source)/('gpu_kernel_fp64.py' if fp64 else 'gpu_kernel.py'))
   self.engine=type('GraphScan',(kernel.GPUMixin,),{})()
   for field in ['steps','rmax','tables','combos','fut']:setattr(self.engine,field,getattr(self,field))
   self.engine.alloc(device=device)
 def amplitude(self,bits):
  f={i:self.u[i,int(bits[i]),:] for i in range(self.n)}
  if self.gpu:val,scale=self.engine._sweep(f,{})
  else:val,scale=self.plan.sweep_reference(self.steps,f,{},self.fut)
  return val*math.exp(scale)*2.**(-self.n/2)

def selftest(source,device,gpu,fp64=False):
 from graph_sampler import graph_vector,apply_one
 rng=np.random.default_rng(20261030);worst=0;details=[]
 for n in [8,10,12]:
  g=rng.integers(0,2,(n,n),dtype=np.uint8);g=np.triu(g,1);g^=g.T
  u=[]
  for j in range(n):
   q,r=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)));u.append(q)
  u=np.array(u);v=graph_vector(g)
  for j in range(n):apply_one(v,u[j],j)
  order=rng.permutation(n).tolist();engine=ScanOracle(g,u,order,source,device,gpu,fp64)
  for t in range(5):
   bits=rng.integers(0,2,n).tolist();amp=engine.amplitude(bits);ref=v[sum(b<<j for j,b in enumerate(bits))];err=abs(amp-ref);rel=abs(abs(amp)**2-abs(ref)**2)/max(abs(ref)**2,1e-30);worst=max(worst,rel);details.append({'n':n,'abs_error':err,'probability_relative_error':rel});assert err<1e-5
  del engine
 return {'backend':'gpu' if gpu else 'cpu','cases':details,'worst_probability_relative_error':worst}

def main():
 p=argparse.ArgumentParser();p.add_argument('--graph',default='rgs_graph.npz');p.add_argument('--order',default='scan_order.npy');p.add_argument('--source',default='upstream_scan');p.add_argument('--device',type=int,default=0);p.add_argument('--cpu',action='store_true');p.add_argument('--fp64',action='store_true');p.add_argument('--reverse-order',action='store_true');p.add_argument('--selftest',action='store_true');p.add_argument('--bitstring');p.add_argument('--samples-jsonl');p.add_argument('--count',type=int,default=1);p.add_argument('--seed',type=int,default=123);p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w') as f:
  if args.selftest:
   rec=selftest(args.source,args.device,not args.cpu,args.fp64);print(json.dumps(rec),flush=True);f.write(json.dumps({'selftest':rec})+'\n');f.flush()
   if args.cpu:return
  data=np.load(args.graph);g=data['graph'];u=data['measurement_unitaries'];order=np.load(args.order).tolist();order=order[::-1] if args.reverse_order else order;n=len(g);t=time.perf_counter();engine=ScanOracle(g,u,order,args.source,args.device,not args.cpu,args.fp64);setup=time.perf_counter()-t;meta={'n':n,'rmax':engine.rmax,'sum_states':sum(2**r for r in engine.ranks),'two_buffers_GiB':(32 if args.fp64 else 16)*2**engine.rmax/2**30,'dtype':'complex128' if args.fp64 else 'complex64','reverse_order':args.reverse_order,'setup_seconds':setup,'source':'Kim Renaud dcs-rankwidth Apache-2.0; custom complex128 variant' if args.fp64 else 'Kim Renaud dcs-rankwidth Apache-2.0 (GPU kernel unmodified; NumPy indexing fix)','status':'probability oracle, not sampling challenge result'};print(json.dumps(meta),flush=True);f.write(json.dumps({'metadata':meta})+'\n');f.flush()
  refs=[]
  if args.samples_jsonl:
   for line in Path(args.samples_jsonl).read_text().splitlines():
    d=json.loads(line)
    if 'bitstring_q0_left' in d:refs.append(d)
   refs=refs[:args.count]
  elif args.bitstring:refs=[{'bitstring_q0_left':args.bitstring}]
  else:
   rng=np.random.default_rng(args.seed);refs=[{'bitstring_q0_left':''.join(map(str,rng.integers(0,2,n)))} for _ in range(args.count)]
  for i,ref in enumerate(refs):
   bitstring=ref['bitstring_q0_left'];assert len(bitstring)==n and not set(bitstring)-set('01');bits=list(map(int,bitstring));t=time.perf_counter();amp=engine.amplitude(bits);elapsed=time.perf_counter()-t;pval=abs(amp)**2;rec={'index':i,'bitstring_q0_left':bitstring,'re':amp.real,'im':amp.imag,'probability':pval,'xeb_contribution':2.**n*pval-1,'seconds':elapsed}
   if 'probability_raw' in ref:rec['reference_raw']=ref['probability_raw'];rec['relative_error_vs_conditional']=abs(pval-ref['probability_raw'])/ref['probability_raw']
   f.write(json.dumps(rec)+'\n');f.flush();print(json.dumps(rec),flush=True)
if __name__=='__main__':main()
