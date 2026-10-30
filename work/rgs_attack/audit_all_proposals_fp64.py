"""Rescore every frozen proposal, accepted and rejected, with reverse complex128 scan.
Does NOT generate new candidates or modify the frozen sample set.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scan_oracle import ScanOracle,selftest

def main():
 p=argparse.ArgumentParser();p.add_argument('--proposals',default='cap1_all_proposals.jsonl');p.add_argument('--graph',default='rgs_graph.npz');p.add_argument('--order',default='scan_order.npy');p.add_argument('--source',default='upstream_scan');p.add_argument('--device',type=int,default=0);p.add_argument('--out',default='cap1_all_fp64_audit.jsonl');p.add_argument('--selftest',action='store_true');p.add_argument('--accepted-scores',help='Reuse existing same-oracle reverse complex128 accepted scores, recompute rejected proposals only');a=p.parse_args()
 source=Path(a.proposals);allrows=[json.loads(l) for l in source.read_text().splitlines() if l.strip()];meta=next(r['metadata'] for r in allrows if 'metadata'in r);rows=[r for r in allrows if 'proposal'in r];n=int(meta['n']);seed=int(meta['seed']);cap=float(meta['cap']);rng=np.random.default_rng(seed)
 # Replay the exact pre-registered RNG and every original decision before any GPU work.
 for i,r in enumerate(rows):
  bits=''.join(map(str,rng.integers(0,2,n)));draw=float(rng.random());accept=min(2.**n*float(r['probability_raw'])/cap,1)
  assert r['proposal']==i and r['bitstring_q0_left']==bits and r['acceptance_uniform']==draw
  assert r['accepted']==(draw<accept) and abs(r['acceptance_probability']-accept)<1e-12
 cached={}
 if a.accepted_scores:
  cached_rows=[json.loads(l) for l in Path(a.accepted_scores).read_text().splitlines() if l.strip()]
  cached_meta=next(x['metadata'] for x in cached_rows if 'metadata'in x)
  assert cached_meta.get('dtype')=='complex128' and cached_meta.get('reverse_order') is True
  accepted_rows=[r for r in rows if r['accepted']];cached_vals=[r for r in cached_rows if 'probability'in r]
  assert len(accepted_rows)==len(cached_vals)
  for original,value in zip(accepted_rows,cached_vals):
   assert original['bitstring_q0_left']==value['bitstring_q0_left']
   cached[original['proposal']]=value
 data=np.load(a.graph);g=data['graph'];u=data['measurement_unitaries'];order=np.load(a.order).tolist()[::-1]
 records=[];start=time.perf_counter();out=Path(a.out)
 with out.open('w') as f:
  header={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'n':n,'seed':seed,'cap':cap,'proposals':len(rows),'accepted':sum(r['accepted'] for r in rows),'dtype':'complex128','order':'reverse of frozen generation order','original_RNG_and_decisions_replayed':True,'scope':'All frozen proposals; no candidate selection or resampling','accepted_scores_sha256':hashlib.sha256(Path(a.accepted_scores).read_bytes()).hexdigest() if a.accepted_scores else None,'accepted_cached_count':len(cached)};f.write(json.dumps({'metadata':header})+'\n');f.flush();print(json.dumps(header),flush=True)
  if a.selftest:
   test=selftest(a.source,a.device,True,True);f.write(json.dumps({'selftest':test})+'\n');f.flush();print(json.dumps({'selftest':test}),flush=True)
  ts=time.perf_counter();engine=ScanOracle(g,u,order,a.source,a.device,gpu=True,fp64=True);setup=time.perf_counter()-ts
  overlap=[]
  if cached:
   ids=sorted(cached);chosen=[ids[i] for i in np.linspace(0,len(ids)-1,8,dtype=int)]
   for idx in chosen:
    fresh=abs(engine.amplitude(list(map(int,rows[idx]['bitstring_q0_left']))))**2
    old=float(cached[idx]['probability']);rel=abs(fresh-old)/max(fresh,np.finfo(float).tiny)
    assert np.isfinite(fresh) and rel<1e-12, ('cached oracle mismatch',idx,rel)
    overlap.append({'proposal':idx,'cached_probability':old,'fresh_probability':fresh,'relative_error':rel})
   f.write(json.dumps({'cached_oracle_overlap_check':overlap})+'\n');f.flush()
  for r in rows:
   if r['proposal'] in cached:prob=float(cached[r['proposal']]['probability']);elapsed=0.;origin='accepted cached from independent earlier reverse FP64 run'
   else:
    t=time.perf_counter();amp=engine.amplitude(list(map(int,r['bitstring_q0_left'])));elapsed=time.perf_counter()-t;prob=abs(amp)**2;origin='new reverse FP64 computation'
   assert np.isfinite(prob) and prob>=0
   acc=min(2.**n*prob/cap,1);decision=r['acceptance_uniform']<acc;relative=abs(prob-r['probability_raw'])/max(prob,np.finfo(float).tiny)
   v={'proposal':r['proposal'],'bitstring_q0_left':r['bitstring_q0_left'],'p_fp32':r['probability_raw'],'p_fp64':prob,'relative_error':relative,'acceptance_uniform':r['acceptance_uniform'],'acceptance_fp32':r['acceptance_probability'],'acceptance_fp64':acc,'accepted_original':r['accepted'],'accepted_fp64':bool(decision),'decision_changed':bool(decision!=r['accepted']),'margin_fp64':abs(r['acceptance_uniform']-acc),'seconds':elapsed,'origin':origin};records.append(v);f.write(json.dumps(v)+'\n');f.flush()
   if len(records)%128==0 or len(records)==len(rows):print(json.dumps({'audited':len(records),'total':len(rows),'decision_changes':sum(x['decision_changed'] for x in records),'elapsed_seconds':time.perf_counter()-start}),flush=True)
  relative=np.array([x['relative_error'] for x in records]);changed=[x['proposal'] for x in records if x['decision_changed']];summary={'proposals_audited':len(records),'accepted_scores_reused':len(cached),'probabilities_recomputed_this_job':len(rows)-len(cached),'accepted_audited':sum(x['accepted_original'] for x in records),'rejected_audited':sum(not x['accepted_original'] for x in records),'all_decisions_preserved':not changed,'decision_changed_indices':changed,'relative_probability_error_max':float(relative.max()),'relative_probability_error_p99':float(np.quantile(relative,.99)),'relative_probability_error_median':float(np.median(relative)),'minimum_acceptance_margin_fp64':min(x['margin_fp64'] for x in records),'maximum_acceptance_probability_difference':max(abs(x['acceptance_fp64']-x['acceptance_fp32']) for x in records),'GPU_oracle_seconds_sum':sum(x['seconds'] for x in records),'setup_seconds':setup,'total_seconds':time.perf_counter()-start,'scope':'Numerical consistency at complex128 reverse order, not a rigorous exact-real error bound'}
  f.write(json.dumps({'summary':summary})+'\n');f.flush();print(json.dumps(summary),flush=True)
 Path(str(out)+'.summary.json').write_text(json.dumps(summary,indent=2))
 if changed:raise SystemExit('FP64 flips found; retain evidence and reassess, do not rewrite sample set')
if __name__=='__main__':main()
