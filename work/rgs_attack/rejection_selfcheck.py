"""Validate the sampling rule against its explicitly normalized small distribution."""
import json
from pathlib import Path
import numpy as np
from graph_sampler import graph_vector,apply_one
rng=np.random.default_rng(55);n=8;g=rng.integers(0,2,(n,n),dtype=np.uint8);g=np.triu(g,1);g^=g.T;v=graph_vector(g)
for q in range(n):
 u,r=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)));apply_one(v,u,q)
p=abs(v)**2;cap=1.;accept=np.minimum(len(v)*p/cap,1);q=accept/accept.sum();target_xeb=float(np.sum(q*(len(v)*p-1)));counts=np.zeros(len(v),int);nacc=0;nprop=0;seq=[]
while nacc<100000:
 x=int(rng.integers(len(v)));z=float(rng.random());nprop+=1
 if z<accept[x]:counts[x]+=1;nacc+=1;seq.append(len(v)*p[x]-1)
emp=counts/nacc;z=np.array(seq);tv=float(abs(emp-q).sum()/2);mean=float(z.mean());se=float(z.std(ddof=1)/np.sqrt(len(z)));lag1=float(np.corrcoef(z[:-1],z[1:])[0,1]);result={'n':n,'cap':cap,'accepted':nacc,'proposals':nprop,'acceptance_predicted':float(accept.mean()),'acceptance_observed':nacc/nprop,'target_xeb':target_xeb,'observed_xeb':mean,'standard_error':se,'empirical_total_variation':tv,'lag1_correlation':lag1,'distribution':'q proportional to min(Dp/cap,1), not Born','independence':'proposal bits and U iid; acceptance depends only on current pair; renewals therefore iid analytically'}
assert abs(mean-target_xeb)<6*se
assert tv<.05
Path(__file__).with_name('rejection_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
