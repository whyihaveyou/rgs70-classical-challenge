import sys,json,re,time
from pathlib import Path
import numpy as np,stim
sys.path.insert(0,str(Path(__file__).parent))
from graph_sampler import parse_qasm,H,TDG
a=Path(__file__).parent
qasm=Path('work/qat/data/classically-verifiable-problems/circuit-models/random_graph_sampling/nq70_depth70_checks27_basis_fstate.qasm')
g,u,_,gates=parse_qasm(qasm);n=len(g)
original=stim.TableauSimulator();original.set_num_qubits(n)
rotations=np.repeat(np.eye(2,dtype=complex)[None],n,axis=0);tail=False
for name,arg,qs in gates:
 if name=='tdg':tail=True
 if not tail:
  gate={'h':original.h,'s':original.s,'sx':original.sqrt_x,'z':original.z,'cz':original.cz}[name];gate(*qs)
 else:
  if name=='h':mat=H
  elif name=='tdg':mat=TDG
  elif name=='rz':
   theta=float(arg);mat=np.diag([np.exp(-.5j*theta),np.exp(.5j*theta)])
  else:raise ValueError(name)
  rotations[qs[0]]=mat@rotations[qs[0]]
local=rotations.conj().transpose(0,2,1)@u
reconstructed=stim.TableauSimulator();reconstructed.set_num_qubits(n);reconstructed.h(*range(n))
for i in range(n):
 for j in range(i+1,n):
  if g[i,j]:reconstructed.cz(i,j)
for i in range(n):
 tab=stim.Tableau.from_unitary_matrix(local[i],endian='little')
 reconstructed.do_tableau(tab,[i])
a0=list(map(str,original.canonical_stabilizers()));a1=list(map(str,reconstructed.canonical_stabilizers()))
assert a0==a1
res={'n':n,'stim_version':stim.__version__,'canonical_stabilizers_equal':True,'checked_count':len(a0),'verification':'Original full 70q Clifford prefix versus independently simulated graph + local Clifford reconstruction; global phase irrelevant.'}
out=Path('reproduced/stim_validation.json');out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
