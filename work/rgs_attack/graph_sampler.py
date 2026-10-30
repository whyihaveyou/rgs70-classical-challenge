"""Exact graph-state conditional simulation; qubit zero is array-index LSB.
Research prototype. Does not approximate or simulate the full 70q instance yet.
"""
import argparse, json, math, re, time
from pathlib import Path
import numpy as np
H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
S=np.diag([1,1j]); Z=np.diag([1,-1]); SX=np.array([[1+1j,1-1j],[1-1j,1+1j]])/2
TDG=np.diag([1,np.exp(-1j*np.pi/4)])

class Tableau:
    def __init__(self,n):
        self.n=n; self.x=np.zeros((n,n),np.uint8); self.z=np.eye(n,dtype=np.uint8); self.r=np.zeros(n,np.uint8)
    def gate(self,name,qs):
        a=qs[0]
        if name=='h':
            self.r ^= self.x[:,a]&self.z[:,a]
            self.x[:,a],self.z[:,a]=self.z[:,a].copy(),self.x[:,a].copy()
        elif name in ('s','sdg'):
            self.r ^= self.x[:,a] & (self.z[:,a] if name=='s' else (1-self.z[:,a]))
            self.z[:,a] ^= self.x[:,a]
        elif name=='z': self.r ^= self.x[:,a]
        elif name=='sx':
            for g in ('h','s','h'): self.gate(g,[a])
        elif name=='cz':
            b=qs[1]; self.gate('h',[b]); self.gate('cx',[a,b]); self.gate('h',[b])
        elif name=='cx':
            b=qs[1]; self.r ^= self.x[:,a]&self.z[:,b]&(self.x[:,b]^self.z[:,a]^1)
            self.x[:,b] ^= self.x[:,a]; self.z[:,a] ^= self.z[:,b]
        else: raise ValueError(name)
    def mulrow(self,i,j):
        x1=self.x[i].astype(np.int64);z1=self.z[i].astype(np.int64)
        x2=self.x[j].astype(np.int64);z2=self.z[j].astype(np.int64)
        xn=x1^x2;zn=z1^z2
        exponent=(2*int(self.r[i])+2*int(self.r[j])+2*int(z1@x2)+int(x1@z1)+int(x2@z2)-int(xn@zn))%4
        assert exponent%2==0
        self.r[i]=exponent//2;self.x[i]=xn;self.z[i]=zn
    def to_graph(self):
        """Returns G,T where |G>=(tensor T_i)|stabilizer>, up to global phase."""
        local=np.repeat(np.eye(2,dtype=complex)[None],self.n,axis=0)
        for j in range(self.n):
            rows=np.flatnonzero(self.x[j:,j])+j
            if not len(rows):
                self.gate('h',[j]);local[j]=H@local[j]
                rows=np.flatnonzero(self.x[j:,j])+j
            if not len(rows):raise RuntimeError('No graph pivot')
            k=int(rows[0]);self.x[[j,k]]=self.x[[k,j]];self.z[[j,k]]=self.z[[k,j]];self.r[[j,k]]=self.r[[k,j]]
            for i in range(self.n):
                if i!=j and self.x[i,j]: self.mulrow(i,j)
        assert np.array_equal(self.x,np.eye(self.n))
        assert np.array_equal(self.z,self.z.T)
        for j in range(self.n):
            if self.z[j,j]:self.gate('sdg',[j]);local[j]=S.conj().T@local[j]
        for j in range(self.n):
            if self.r[j]:self.gate('z',[j]);local[j]=Z@local[j]
        assert not np.any(self.r)
        return self.z.copy(),local

def gf2rank(a):
    a=a.copy(); r=0
    for c in range(a.shape[1]):
        cand=np.flatnonzero(a[r:,c])+r
        if len(cand):
            p=cand[0];a[[r,p]]=a[[p,r]]
            for j in range(r+1,len(a)):
                if a[j,c]:a[j]^=a[r]
            r+=1
            if r==len(a):break
    return r

def apply_one(v,u,q):
    block=v.reshape(-1,2,1<<q)
    a=block[:,0,:].copy();b=block[:,1,:].copy()
    block[:,0,:]=u[0,0]*a+u[0,1]*b; block[:,1,:]=u[1,0]*a+u[1,1]*b

def quadratic_phase(v,g):
    ids=np.arange(len(v),dtype=np.uint64)
    for a in range(len(g)):
        mask=sum(1<<int(b) for b in np.flatnonzero(g[a,a+1:])+a+1)
        if not mask:continue
        y=ids & np.uint64(mask)
        # uint64 parity with no NumPy-2-only bit_count dependency
        y ^= y>>np.uint64(32);y ^= y>>np.uint64(16);y ^= y>>np.uint64(8);y ^= y>>np.uint64(4)
        parity=(np.uint64(0x6996)>>(y&15))&1
        v *= 1.0-2.0*((ids>>np.uint64(a))&parity)

def graph_vector(g):
    v=np.ones(1<<len(g),complex)/np.sqrt(1<<len(g));quadratic_phase(v,g);return v

def walsh(v):
    for q in range(int(math.log2(len(v)))):
        block=v.reshape(-1,2,1<<q);a=block[:,0,:].copy();b=block[:,1,:].copy()
        block[:,0,:]=a+b;block[:,1,:]=a-b

def conditional(g,u,A,B,outB):
    """Unnormalized postselected A vector and exact B marginal. Exponential in |A| and |B|."""
    ma=len(A);mb=len(B);ids=np.arange(1<<mb,dtype=np.uint64)
    v=np.ones(1<<mb,complex)
    for i,b in enumerate(B):v *= u[b,outB[i],((ids>>np.uint64(i))&1).astype(np.int64)]
    quadratic_phase(v,g[np.ix_(B,B)]);walsh(v)
    ida=np.arange(1<<ma,dtype=np.uint64);mapped=np.zeros_like(ida)
    for i,a in enumerate(A):
        mask=sum(1<<j for j,b in enumerate(B) if g[a,b])
        mapped ^= (((ida>>np.uint64(i))&1)*np.uint64(mask))
    va=v[mapped.astype(np.int64)].copy()/(2.0**(len(g)/2))
    quadratic_phase(va,g[np.ix_(A,A)])
    for i,a in enumerate(A):apply_one(va,u[a],i)
    norm=float(np.vdot(va,va).real)
    return va,norm

def parse_qasm(path):
    text=Path(path).read_text();n=int(re.search(r'qreg q\[(\d+)\]',text).group(1));tab=Tableau(n)
    rotations=np.repeat(np.eye(2,dtype=complex)[None],n,axis=0);tail=False;gates=[]
    for line in text.splitlines():
        m=re.match(r'(\w+)(?:\(([^)]*)\))? (q\[.*);',line)
        if not m:continue
        name,arg,ops=m.groups()
        if name=='qreg':continue
        qs=[int(x) for x in re.findall(r'q\[(\d+)\]',ops)]
        if name=='tdg':tail=True
        gates.append((name,arg,qs))
        if not tail:tab.gate(name,qs)
        else:
            if len(qs)!=1:raise ValueError('Entangler after magic tail')
            if name=='h':mat=H
            elif name=='tdg':mat=TDG
            elif name=='rz':
                theta=float(arg);mat=np.diag([np.exp(-.5j*theta),np.exp(.5j*theta)])
            else:raise ValueError(name)
            rotations[qs[0]]=mat@rotations[qs[0]]
    g,t=tab.to_graph();u=rotations@t.conj().transpose(0,2,1)
    return g,u,tab,gates

def selfcheck():
    rng=np.random.default_rng(20261030);records=[]
    for n in (4,6,8,10):
        for trial in range(4):
            tab=Tableau(n);v=np.zeros(1<<n,complex);v[0]=1
            for _ in range(8*n):
                a,b=rng.choice(n,2,replace=False);name=rng.choice(['h','s','sx','z','cz']);tab.gate(name,[a,b] if name=='cz' else [a])
                if name=='cz':
                    idx=np.arange(len(v));v[(idx&(1<<a)!=0)&(idx&(1<<b)!=0)] *= -1
                else:apply_one(v,dict(h=H,s=S,sx=SX,z=Z)[name],a)
            g,t=tab.to_graph();vg=graph_vector(g)
            for q in range(n):apply_one(vg,t[q].conj().T,q)
            overlap=abs(np.vdot(v,vg));assert abs(overlap-1)<1e-12
            u=[]
            for q in range(n):
                z=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2));Q,R=np.linalg.qr(z);u.append(Q)
            u=np.array(u);full=graph_vector(g)
            for q in range(n):apply_one(full,u[q],q)
            A=list(range(n//2));B=list(range(n//2,n));err=0;max_norm_err=0
            for z in range(1<<len(B)):
                ob=[(z>>i)&1 for i in range(len(B))];ca,pb=conditional(g,u,A,B,ob)
                exact=full[z*(1<<len(A)):(z+1)*(1<<len(A))]
                err=max(err,float(np.max(np.abs(ca-exact))))
                assert np.allclose(ca,exact,atol=2e-14)
                if gf2rank(g[np.ix_(A,B)])==len(B):max_norm_err=max(max_norm_err,abs(pb-2.**(-len(B))))
            records.append(dict(n=n,trial=trial,tableau_overlap=overlap,conditional_max_abs_error=err,full_rank_marginal_error=max_norm_err))
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--qasm');p.add_argument('--out',default='work/rgs_attack');p.add_argument('--benchmark',type=int,default=0);args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    result={'validation':selfcheck(),'scope':'Small-instance exact oracle + full-instance structural audit only. No full 70q samples generated.'}
    if args.qasm:
        start=time.perf_counter();g,u,tab,gates=parse_qasm(args.qasm);n=len(g);rng=np.random.default_rng(20261030);ranks=[]
        for trial in range(1000):
            perm=rng.permutation(n);A=perm[:n//2].tolist();B=perm[n//2:].tolist();r=gf2rank(g[np.ix_(A,B)]);ranks.append(r)
            if r==n//2:break
        np.savez(out/'rgs_graph.npz',graph=g,measurement_unitaries=u,A=A,B=B)
        result['instance']={'qasm':args.qasm,'n':n,'graph_edges':int(g.sum()//2),'graph_conversion_seconds':time.perf_counter()-start,'partition_A':A,'partition_B':B,'cross_rank':r,'rank_trials':ranks,'statevector_complex64_bytes':8*2**(n//2),'statevector_complex128_bytes':16*2**(n//2),'has_exact_uniform_B_marginal':r==len(B),'local_unitary_residual':float(np.max(np.abs(u@u.conj().transpose(0,2,1)-np.eye(2))))}
        if args.benchmark:
            m=args.benchmark;sg=g[:2*m,:2*m];su=u[:2*m];start=time.perf_counter();v,norm=conditional(sg,su,list(range(m)),list(range(m,2*m)),[0]*m);elapsed=time.perf_counter()-start
            result['benchmark']={'induced_graph_qubits':2*m,'conditional_qubits':m,'wall_seconds':elapsed,'vector_bytes':v.nbytes,'B_marginal':norm,'scope':'induced graph, not full QAT circuit'}
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='validation'},indent=2));print('independent oracle cases passed',len(result['validation']))
if __name__=='__main__':main()
