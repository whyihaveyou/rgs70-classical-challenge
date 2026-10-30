"""Audit preregistered capped-sampling experiment and independent probability scores.
Assumes bitstring fields are qubit-0-left; IBM conversion must happen upstream.
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

def load(path):
 rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()];return rows

def stat(x,seed=424242,B=20000):
 x=np.asarray(x,dtype=np.float64)
 if not len(x):return {'n':0}
 mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None;rng=np.random.default_rng(seed);boots=[]
 for start in range(0,B,256):boots.extend(x[rng.integers(0,len(x),size=(min(256,B-start),len(x)))].mean(axis=1).tolist())
 ci=np.quantile(boots,[.025,.975]).tolist();clipped=np.minimum(x+1,4)-1;hoeffding=float(clipped.mean()-4*np.sqrt(np.log(40)/(2*len(x))));return {'clipped_K4_mean':float(clipped.mean()),'clipped_K4_one_sided_Hoeffding_lower_alpha_0_025':hoeffding,'clipped_K4_bound_scope':'Distribution-free iid lower bound on un-clipped population XEB, conditional on valid target probability evaluation; alpha .025 per two predeclared runs (Bonferroni .05 familywise)','n':len(x),'mean':mean,'standard_error':se,'normal_95_CI':[mean-1.96*se,mean+1.96*se] if se is not None else None,'bootstrap_95_CI':ci,'bootstrap_resamples':B,'bootstrap_seed':seed,'min':float(x.min()),'max':float(x.max())}

def score_rows(rows):return [x for x in rows if 'bitstring_q0_left' in x and 'probability' in x]

def find(base,pattern,required=True):
 found=sorted(Path(base).rglob(pattern))
 if len(found)!=1:
  if not found and not required:return None
  raise ValueError(f'Expected one {pattern}, found {found}')
 return found[0]

def main():
 p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--proposals');p.add_argument('--scored');p.add_argument('--ibm');p.add_argument('--uniform');p.add_argument('--out',default=None);a=p.parse_args();base=Path(a.directory);out=Path(a.out) if a.out else base/'analysis';out.mkdir(parents=True,exist_ok=True)
 proposals=Path(a.proposals) if a.proposals else find(base,'capped1726.jsonl');scored=Path(a.scored) if a.scored else find(base,'scored_fp64.jsonl');ibm=Path(a.ibm) if a.ibm else find(base,'ibm_scored_fp64.jsonl',False);uniform=Path(a.uniform) if a.uniform else find(base,'uniform_scored_fp64.jsonl',False)
 raw=load(proposals);meta=next(x['metadata'] for x in raw if 'metadata' in x);summary=next((x['summary'] for x in raw if 'summary' in x),{});props=[x for x in raw if 'proposal' in x];acc=[x for x in props if x['accepted']];scores=score_rows(load(scored));n=int(meta['n']);cap=float(meta['cap']);rng=np.random.default_rng(int(meta['seed']));mismatch=[]
 for i,row in enumerate(props):
  bits=''.join(map(str,rng.integers(0,2,n)));u=float(rng.random());prob=float(row['probability_raw']);expected_a=min(2.**n*prob/cap,1.)
  checks=[row['proposal']==i,row['bitstring_q0_left']==bits,row['acceptance_uniform']==u,abs(row['acceptance_probability']-expected_a)<1e-12,row['accepted']==(u<expected_a)]
  if not all(checks):mismatch.append(i)
 if mismatch:raise AssertionError(f'Failed decision/RNG replay: {mismatch[:10]}')
 if len(acc)!=len(scores):raise AssertionError(f'Count mismatch accepted={len(acc)} scores={len(scores)}')
 for x,y in zip(acc,scores):
  if x['bitstring_q0_left']!=y['bitstring_q0_left']:raise AssertionError('Scoring order mismatch')
 xeb=np.array([2.**n*x['probability']-1 for x in scores]);c64=np.array([x['probability_raw'] for x in acc]);c128=np.array([x['probability'] for x in scores]);rel=abs(c64-c128)/np.maximum(c128,np.finfo(float).tiny);delt=np.array([min(2.**n*x['probability']/cap,1.) for x in scores]);changed=sum(x['acceptance_uniform']>=b for x,b in zip(acc,delt))
 elapsed=sum(x['seconds'] for x in props);gen=summary.get('generation_seconds');result={'scope':'Independent capped-probability sampler; q proportional to min(D*p/cap,1), NOT exact Born and NOT state-fidelity certification','n_qubits':n,'cap':cap,'seed':meta['seed'],'proposals':len(props),'accepted':len(acc),'acceptance_rate':len(acc)/len(props),'RNG_and_acceptance_replay_passed':True,'duplicate_accepted_bitstrings':len(acc)-len(set(x['bitstring_q0_left'] for x in acc)),'score':stat(xeb),'fp64_vs_fp32_relative_probability_error':{'max':float(rel.max()),'median':float(np.median(rel)),'p99':float(np.quantile(rel,.99))},'accepted_decisions_that_would_reverse_using_fp64':int(changed),'all_proposals_min_abs_acceptance_margin_fp32':min(abs(x['acceptance_uniform']-x['acceptance_probability']) for x in props),'near_acceptance_boundary_within_1e_6':[x['proposal'] for x in props if abs(x['acceptance_uniform']-x['acceptance_probability'])<1e-6],'near_acceptance_boundary_within_1e_5':[x['proposal'] for x in props if abs(x['acceptance_uniform']-x['acceptance_probability'])<1e-5],'candidate_oracle_seconds_sum':elapsed,'generation_seconds_reported':gen,'setup_seconds':summary.get('setup_seconds',meta.get('setup_seconds')),'independent_scoring_seconds':sum(x['seconds'] for x in scores),'threshold_reference':{'QAT_XEB':.092,'quantum_runtime_seconds':1950,'bootstrap_lower_bound_above_0_092':stat(xeb)['bootstrap_95_CI'][0]>.092},'timing_scope':'Sampling job generation walltime and setup counted separately; queue/download/install/search/independent validation not included unless caller adds to report. GPU hours and total project walltime must be disclosed.','source_files':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [proposals,scored]}}
 for name,path in [('ibm',ibm),('uniform',uniform)]:
  if path:
   vals=score_rows(load(path));arr=np.array([2.**n*x['probability']-1 for x in vals]);result[name]={'score':stat(arr),'scoring_seconds':sum(x['seconds'] for x in vals),'duplicate_bitstrings':len(vals)-len(set(x['bitstring_q0_left'] for x in vals))};result['source_files'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 if 'ibm' in result:
  result['capped_minus_ibm_xeb']=result['score']['mean']-result['ibm']['score']['mean'];sx=result['score']['standard_error'];sy=result['ibm']['score']['standard_error'];d=result['capped_minus_ibm_xeb'];se=math.sqrt(sx*sx+sy*sy);result['difference_independent_normal_95_CI']=[d-1.96*se,d+1.96*se]
 with (out/'accepted_scores.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['sample','proposal','bitstring_q0_left','p_fp32','p_fp64','relative_error','xeb_fp64','proposal_seconds','fp64_seconds'])
  for i,(x,y,r,z) in enumerate(zip(acc,scores,rel,xeb)):w.writerow([i,x['proposal'],x['bitstring_q0_left'],x['probability_raw'],y['probability'],r,z,x['seconds'],y['seconds']])
 with (out/'proposal_audit.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['proposal','bitstring_q0_left','p_fp32','acceptance_probability','acceptance_uniform','accepted','seconds'])
  for x in props:w.writerow([x['proposal'],x['bitstring_q0_left'],x['probability_raw'],x['acceptance_probability'],x['acceptance_uniform'],x['accepted'],x['seconds']])
 (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
