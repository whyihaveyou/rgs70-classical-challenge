"""Fixed 32-input, alternating, same-GPU benchmark; independent reverse FP64 scores."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scan_oracle import ScanOracle

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='scan_order_benchmark.json');p.add_argument('--device',type=int,default=0);a=p.parse_args()
 data=np.load('rgs_graph.npz');g=data['graph'];u=data['measurement_unitaries'];old=np.load('scan_order.npy').tolist();new=np.load('scan_order_optimized.npy').tolist()
 engines={name:ScanOracle(g,u,order,'upstream_scan',a.device,True,fp64) for name,order,fp64 in [('old',old,False),('new',new,False),('reference',old[::-1],True)]}
 rng=np.random.default_rng(161803);bits=[rng.integers(0,2,len(g)).tolist() for _ in range(32)]
 # Fixed warm-up count, before timing. No adaptive selection of favorable cases.
 for engine in engines.values():
  for b in bits[:2]:engine.amplitude(b)
 rows=[]
 for i,b in enumerate(bits):
  row={'index':i,'bitstring_q0_left':''.join(map(str,b))}
  for name in (['old','new'] if i%2==0 else ['new','old'])+['reference']:
   t=time.perf_counter();prob=abs(engines[name].amplitude(b))**2;row[name+'_seconds']=time.perf_counter()-t;row[name+'_probability']=prob
   assert np.isfinite(prob) and prob>=0
  for name in ['old','new']:row[name+'_relative_probability_error']=abs(row[name+'_probability']-row['reference_probability'])/max(row['reference_probability'],np.finfo(float).tiny)
  rows.append(row)
 totals={name:sum(r[name+'_seconds'] for r in rows) for name in ['old','new','reference']}
 result={'metadata':{'n':70,'count':32,'seed':161803,'warmup_per_engine':2,'ordering':'old/new alternated each input, then reverse complex128 reference','GPU_count':1,'algorithm':'unchanged dcs-rankwidth kernel; changed scan order only','historical_cap1_cap8_unchanged':True,'old_order_sha256':hashlib.sha256(Path('scan_order.npy').read_bytes()).hexdigest(),'new_order_sha256':hashlib.sha256(Path('scan_order_optimized.npy').read_bytes()).hexdigest()},'summary':{'total_seconds':totals,'old_over_new_walltime_ratio':totals['old']/totals['new'],'runtime_reduction_fraction':1-totals['new']/totals['old'],'old_sum_states':sum(2**r for r in engines['old'].ranks),'new_sum_states':sum(2**r for r in engines['new'].ranks),'maximum_probability_relative_error':{name:max(r[name+'_relative_probability_error'] for r in rows) for name in ['old','new']}},'rows':rows}
 Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result['summary']),flush=True)
 assert max(result['summary']['maximum_probability_relative_error'].values())<1e-4
if __name__=='__main__':main()
