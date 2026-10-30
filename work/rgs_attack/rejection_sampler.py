"""Independent capped-probability sampler using the exact-contraction RGS oracle.
q(x) proportional to min(2^n p(x)/cap,1), not the Born distribution.
No MCMC. All candidates and acceptance uniforms recorded for auditing.
"""
import argparse,json,time,math
from pathlib import Path
import numpy as np
from scan_oracle import ScanOracle

def main():
 p=argparse.ArgumentParser();p.add_argument('--graph',default='rgs_graph.npz');p.add_argument('--order',default='scan_order.npy');p.add_argument('--source',default='upstream_scan');p.add_argument('--device',type=int,default=0);p.add_argument('--count',type=int,default=128);p.add_argument('--seed',type=int,default=314159);p.add_argument('--cap',type=float,default=1.0);p.add_argument('--max-proposals',type=int,default=100000);p.add_argument('--out',required=True);a=p.parse_args()
 if a.cap<=0:raise ValueError('cap must be positive')
 data=np.load(a.graph);g=data['graph'];u=data['measurement_unitaries'];order=np.load(a.order).tolist();n=len(g);rng=np.random.default_rng(a.seed);start=time.perf_counter();engine=ScanOracle(g,u,order,a.source,a.device);setup=time.perf_counter()-start
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);accepted=[];total=0;begin=time.perf_counter()
 with out.open('w') as f:
  meta={'target':'QAT RGS70' if n==70 else 'small validation instance','n':n,'seed':a.seed,'cap':a.cap,'distribution':'q(x) proportional to min(2^n*p(x)/cap,1); NOT Born-exact','independence':'Each candidate and acceptance uniform generated independently; accepted outputs iid under deterministic probability oracle','rank':engine.rmax,'setup_seconds':setup,'source':'Kim Renaud dcs-rankwidth Apache-2.0 CUDA contraction; custom graph preprocessing/order and capped sampler','claims':'Algorithm efficiency experiment; not a QAT supersession unless score/time/error criteria independently established'}
  f.write(json.dumps({'metadata':meta})+'\n');f.flush();print(json.dumps(meta),flush=True)
  while len(accepted)<a.count and total<a.max_proposals:
   bits=rng.integers(0,2,n).tolist();accept_u=float(rng.random());t=time.perf_counter();amp=engine.amplitude(bits);dt=time.perf_counter()-t;prob=abs(amp)**2;scaled=2.**n*prob;acc=min(scaled/a.cap,1.);keep=accept_u<acc
   if not np.isfinite(scaled):raise ArithmeticError('Invalid target probability')
   rec={'proposal':total,'bitstring_q0_left':''.join(map(str,bits)),'probability_raw':prob,'xeb_contribution':scaled-1,'acceptance_probability':acc,'acceptance_uniform':accept_u,'accepted':keep,'seconds':dt}
   if keep:rec['sample']=len(accepted);accepted.append(rec)
   f.write(json.dumps(rec)+'\n');f.flush();total+=1
   if total%16==0 or len(accepted)==a.count:print(json.dumps({'proposals':total,'accepted':len(accepted),'elapsed_seconds':time.perf_counter()-begin,'partial_xeb':float(np.mean([x['xeb_contribution'] for x in accepted])) if accepted else None}),flush=True)
  elapsed=time.perf_counter()-begin;z=np.array([x['xeb_contribution'] for x in accepted]);mean=float(z.mean()) if len(z) else None;se=float(z.std(ddof=1)/math.sqrt(len(z))) if len(z)>1 else None
  summary={'completed':len(accepted)==a.count,'accepted':len(accepted),'proposals':total,'acceptance_fraction':len(accepted)/total if total else 0,'generation_seconds':elapsed,'setup_seconds':setup,'xeb':mean,'standard_error':se,'normal_approx_95_CI':[mean-1.96*se,mean+1.96*se] if se is not None else None,'inference_note':'Normal approximation only; oracle and selected probabilities need independent numerical verification. No Porter-Thomas distribution assumption used in generation.'}
  f.write(json.dumps({'summary':summary})+'\n');f.flush();print(json.dumps(summary),flush=True)
 Path(str(out)+'.accepted.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in accepted))
 if len(accepted)!=a.count:raise SystemExit('Proposal limit reached')
if __name__=='__main__':main()
