"""CPU-only reproduction of the frozen capped results; run from repository root."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
base=root/'work/rgs_attack'
out=root/'reproduced';out.mkdir(exist_ok=True)
for rec in json.loads((root/'ORIGINAL_HASHES.json').read_text())['files']:
 if rec['byte_identical']:
  assert hashlib.sha256((root/rec['public_path']).read_bytes()).hexdigest()==rec['original_sha256'],rec['public_path']
sys.path.insert(0,str(base))
from analyze_challenge import load,score_rows,stat
subprocess.run([sys.executable,str(base/'analyze_challenge.py'),str(base/'cap1_archive'),'--out',str(out/'cap1')],check=True,stdout=subprocess.DEVNULL)
for j in range(2):
 subprocess.run([sys.executable,str(base/'analyze_challenge.py'),str(base/'cap8_archive'),'--proposals',str(base/f'cap8_archive/cap8_part{j}.jsonl'),'--scored',str(base/f'cap8_archive/cap8_scored{j}.jsonl'),'--out',str(out/f'cap8_part{j}')],check=True,stdout=subprocess.DEVNULL)
vals=[]
for j in range(2):vals.extend(score_rows(load(base/f'cap8_archive/cap8_scored{j}.jsonl')))
combined=stat([2.**70*r['probability']-1 for r in vals])
assert abs(combined['mean']-json.loads((base/'cap8_archive/combined_summary.json').read_text())['score']['mean'])<1e-12
(out/'cap8_combined_score.json').write_text(json.dumps(combined,indent=2)+'\n')
props=[r for r in load(base/'cap1_archive/capped1726.jsonl') if 'proposal'in r]
audit=[r for r in load(base/'cap1_full_audit_artifact/cap1_all_fp64_audit.jsonl') if 'proposal'in r]
assert len(props)==len(audit)==2769
for p,a in zip(props,audit):
 assert p['proposal']==a['proposal'] and p['bitstring_q0_left']==a['bitstring_q0_left']
 assert p['probability_raw']==a['p_fp32']
 assert (p['acceptance_uniform']<min(2.**70*a['p_fp64'],1))==p['accepted']==a['accepted_fp64']
print(json.dumps({'hashes_verified':True,'cap1_proposals_audited':len(audit),'cap1_all_decisions_preserved':True,'cap8_combined_XEB':combined['mean'],'output_directory':'reproduced'},indent=2))
