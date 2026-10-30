import json,time,math,argparse
from pathlib import Path
import numpy as np

def profile(order,adj):
 n=len(order);future=(1<<n)-1;basis=[];out=[]
 for v in order:
  future &= ~(1<<v);piv={}
  for x in [b&future for b in basis]+[adj[v]&future]:
   while x:
    p=x.bit_length()-1
    if p in piv:x ^= piv[p]
    else:piv[p]=x;break
  basis=list(piv.values());out.append(len(basis))
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=30);a=p.parse_args();data=np.load(Path(__file__).with_name('rgs_graph.npz'));g=data['graph'];n=len(g);adj=[sum(1<<int(j) for j in np.flatnonzero(g[i])) for i in range(n)];rng=np.random.default_rng(100)
 def cost(o):
  r=profile(o,adj);return sum(2.**x for x in r),r
 order=list(range(n));val,r=cost(order);best=val;bestorder=order.copy();bestr=r.copy();initial={'maxrank':max(r),'sum_states':val};t0=time.perf_counter();count=0;temperature=0.4
 while time.perf_counter()-t0<a.seconds:
  cand=order.copy();i,j=map(int,rng.choice(n,2,replace=False));cand[i],cand[j]=cand[j],cand[i];nv,nr=cost(cand);count+=1
  if nv<val or rng.random()<math.exp(min(0,(val-nv)/(temperature*2**max(r)))):order,val,r=cand,nv,nr
  if val<best:best=val;bestorder=order.copy();bestr=r.copy();print(count,max(r),val,flush=True)
  temperature=max(.02,temperature*.9995)
 result={'scope':'full70 graph-state linear scan rank, heuristic upper bound not proven optimum','initial':initial,'maxrank':max(bestr),'sum_states':best,'profile':bestr,'order':bestorder,'iterations':count,'seconds':time.perf_counter()-t0,'two_buffers_complex64_GiB':16*2**max(bestr)/2**30}
 Path(__file__).with_name('scan_width.json').write_text(json.dumps(result,indent=2));np.save(Path(__file__).with_name('scan_order.npy'),bestorder);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
