"""Exact algebraic specialization of FourierCovariantSampler initialization.

No frozen sampler is modified. CUDA code is provided but not run by this file.
All selected B measurement matrices MUST be exactly equal, not merely close.
Tables preserve arbitrary complex phases (including tiny imaginary parts).

For b=N z, N=M^{-T}, init(z)=2^{-m/2}(-1)^q_B(b) prod U[out_j,b_j].
1. Substitute b=N z in the GF(2) quadratic form, retaining its linear part.
2. Chunk LUTs map z->b and evaluate the transformed quadratic with five
   table lookups and one accumulated parity for m=35, with chunk_bits=8.
3. Four exponents are fixed by popcount(out), popcount(b), popcount(out & b),
   so a small complex table replaces all per-site complex multiplications.

GPU import: append CUDA_INIT_SPECIALIZATION to an existing CUDA module; copy
spec.packed_tables as uint64 and view its pointer as const ulonglong2*; copy
spec.weight_table(outB, np.complex64); call initcov_specialized_value from
another device function, or launch initcov_specialized. Scale is in weights.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import time
import numpy as np

from fourier_covariant_sampler import (
    gf2_inverse, row_masks, map_indices, phase_for_ids, parity64,
)
from graph_sampler import walsh, quadratic_phase, apply_one, conditional


def _host_popcount(x):
    # Retain macOS system Python 3.9 compatibility; this is host setup only.
    return bin(int(x)).count('1')


def popcount64(ids):
    x = np.asarray(ids, dtype=np.uint64).copy()
    x -= (x >> np.uint64(1)) & np.uint64(0x5555555555555555)
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def output_mask(outB):
    bits = np.asarray(outB)
    if bits.ndim != 1 or np.any((bits != 0) & (bits != 1)):
        raise ValueError('outB must contain bits')
    return sum(int(bit) << j for j, bit in enumerate(bits))


def transformed_quadratic(graph_B, N):
    """q(Nz)=sum_{i<j} Q_ij z_i z_j + ell.z, exactly over GF(2)."""
    upper = np.triu(np.asarray(graph_B, np.uint64), 1)
    P = (np.asarray(N, np.uint64).T @ upper @ np.asarray(N, np.uint64)) & 1
    linear = np.diag(P).copy().astype(np.uint8)
    Q = np.triu(P ^ P.T, 1).astype(np.uint8)
    return Q, linear


@dataclass
class InitSpecialization:
    m: int
    chunk_bits: int
    inverse_transpose: np.ndarray
    graph_B: np.ndarray
    unitary: np.ndarray
    phase_quadratic: np.ndarray
    phase_linear: int
    packed_tables: np.ndarray

    @property
    def chunks(self):
        return self.packed_tables.shape[0]

    @property
    def inverse_rows(self):
        return row_masks(self.inverse_transpose)

    @property
    def phase_rows(self):
        return row_masks(self.phase_quadratic)

    def toggle_metadata(self, bit):
        """For z -> z xor 2**bit, return (b XOR, marked-z phase mask).

        q(z xor e_j) xor q(z) = ell_j xor sum_{i!=j} Q_ij z_i.
        Bit63 holds ell_j, to be ANDed with z|(1<<63), as in CUDA.
        """
        if not 0 <= bit < self.m:
            raise ValueError('toggle bit outside domain')
        column = output_mask(self.inverse_transpose[:,bit])
        mask = output_mask(self.phase_quadratic[:,bit] ^ self.phase_quadratic[bit,:])
        mask |= ((self.phase_linear >> bit) & 1) << 63
        return column, mask

    def weight_table(self, outB, dtype=np.complex64):
        """Table[k,h], scale included; depends on outB ONLY through popcount.

        Build in complex128 from original matrix, then round once to dtype.
        This is algebraically equal to generic multiplication; floating-point
        rounding differs from repeated complex64 products in the old kernel.
        Invalid (k,h) combinations are zero but never addressed by valid IDs.
        """
        if len(outB) != self.m:
            raise ValueError('wrong outB length')
        r = _host_popcount(output_mask(outB))
        p = np.stack([np.power(complex(a), np.arange(self.m+1)) for a in self.unitary.flat])
        table = np.zeros((self.m+1, self.m+1), np.complex128)
        for k in range(self.m+1):
            for h in range(max(0, r+k-self.m), min(r, k)+1):
                n00, n01, n10, n11 = self.m-r-k+h, k-h, r-h, h
                table[k,h] = 2.**(-self.m/2) * p[0,n00]*p[1,n01]*p[2,n10]*p[3,n11]
        return table.astype(dtype)

    def map_and_phase(self, ids):
        ids = np.asarray(ids, np.uint64)
        if np.any(ids >> np.uint64(self.m)):
            raise ValueError('id outside m-bit domain')
        b = np.zeros_like(ids)
        par = np.zeros_like(ids)
        marked = ids | np.uint64(1 << 63)
        mask = np.uint64((1 << self.chunk_bits)-1)
        for c in range(self.chunks):
            code = ((ids >> np.uint64(c*self.chunk_bits)) & mask).astype(np.int64)
            entry = self.packed_tables[c,code]
            b ^= entry[...,0]
            par ^= parity64(marked & entry[...,1])
        return b, 1.-2.*par.astype(float)

    def evaluate(self, ids, outB, dtype=np.complex128):
        b, phase = self.map_and_phase(ids)
        w = self.weight_table(outB, dtype)
        return (w[popcount64(b), popcount64(b & np.uint64(output_mask(outB)))] * phase).astype(dtype)

    def generic(self, ids, outB, dtype=np.complex128):
        """Independent per-site oracle following the existing initcov kernel."""
        b = map_indices(ids, self.inverse_transpose)
        v = np.full(np.asarray(ids).shape, 2.**(-self.m/2), dtype=dtype)
        U = self.unitary.astype(dtype)
        for j, out in enumerate(outB):
            v *= U[out, ((b >> np.uint64(j)) & 1).astype(int)]
        return (v * phase_for_ids(b, self.graph_B)).astype(dtype)

    def operation_counts(self):
        return dict(m=self.m, chunk_bits=self.chunk_bits, chunks=self.chunks,
                    baseline_per_output=dict(mapping_popcounts=self.m,
                        complex_multiplies=self.m, phase_bit_branches=self.m,
                        phase_popcounts='0..m (about m/2 for uniform mapped bits)'),
                    specialized_per_output=dict(complex_multiplies=0,
                        popcounts=3, paired_uint64_table_loads=self.chunks,
                        mapped_mask_xors=self.chunks, complex_weight_loads=1,
                        data_dependent_branches=0),
                    invariant_table_bytes=self.packed_tables.nbytes,
                    per_outB_popcount_weight_table_bytes=(self.m+1)**2*8,
                    note='Operation count only. CUDA performance not measured.')


def build_specialization(g, u, A, B, chunk_bits=8):
    m = len(B)
    if len(A) != m or not 1 <= m <= 63:
        raise ValueError('balanced 1..63-bit partition required')
    if not 1 <= chunk_bits <= 10:
        raise ValueError('chunk_bits must be 1..10')
    selected = np.asarray(u)[B]
    if not np.array_equal(selected, np.broadcast_to(selected[0], selected.shape)):
        raise ValueError('all B measurement matrices must be exactly equal')
    N = gf2_inverse(np.asarray(g)[np.ix_(B,A)]).T
    gb = np.asarray(g, np.uint8)[np.ix_(B,B)]
    Q, linear = transformed_quadratic(gb, N)
    linear_mask = sum(int(bit) << i for i, bit in enumerate(linear))
    rows = row_masks(Q)
    columns = row_masks(N.T)
    chunks = math.ceil(m/chunk_bits)
    lut = np.zeros((chunks, 1 << chunk_bits, 2), dtype=np.uint64)
    for c in range(chunks):
        start, stop = c*chunk_bits, min(m, (c+1)*chunk_bits)
        higher = ((1 << m)-1) ^ ((1 << stop)-1)
        for code in range(1 << (stop-start)):
            z = code << start
            mapped, cross, inner = 0, 0, _host_popcount(z & linear_mask) & 1
            for j in range(start, stop):
                if (z >> j) & 1:
                    mapped ^= int(columns[j])
                    cross ^= int(rows[j]) & higher
                    inner ^= _host_popcount(z & int(rows[j])) & 1
            lut[c,code] = mapped, cross | (inner << 63)
    return InitSpecialization(m, chunk_bits, N, gb, selected[0].copy(), Q, linear_mask, lut)


# Standalone names avoid collisions with existing CUDA add/mul/ull helpers.
# t is the uint64 packed_tables buffer reinterpreted as ulonglong2. No texture,
# shared-memory, or constant-memory assumption is required.
CUDA_INIT_SPECIALIZATION = r'''
__device__ __forceinline__ float2 initcov_specialized_value(
    unsigned long long id, unsigned long long outmask,
    const ulonglong2* t, const float2* weights,
    int m, int chunk_bits, int chunks) {
    unsigned long long b=0, pm=0, marked=id|(1ull<<63);
    const unsigned mask=(1u<<chunk_bits)-1;
    for(int c=0;c<chunks;c++) {
        unsigned code=(unsigned)(id>>(c*chunk_bits))&mask;
        ulonglong2 entry=t[((unsigned)c<<chunk_bits)+code];
        b ^= entry.x;
        pm ^= entry.y & marked;
    }
    // XOR parity may be delayed: one popcount replaces one per chunk.
    unsigned par=__popcll(pm)&1;
    unsigned k=__popcll(b), h=__popcll(b&outmask);
    float2 v=weights[k*(m+1)+h];
    float sign=1.0f-2.0f*(float)par;
    return make_float2(sign*v.x,sign*v.y);
}
// Pair for z and z XOR a chosen global bit; toggle metadata is host-computed.
// Returns (Re g0, Im g0, Re g1, Im g1). Saves the second chunk-LUT traversal.
__device__ __forceinline__ float4 initcov_specialized_pair(
    unsigned long long id, unsigned long long outmask,
    unsigned long long map_xor, unsigned long long phase_diff_mask,
    const ulonglong2* t, const float2* weights,
    int m, int chunk_bits, int chunks) {
    unsigned long long b=0, pm=0, marked=id|(1ull<<63);
    const unsigned mask=(1u<<chunk_bits)-1;
    for(int c=0;c<chunks;c++) {
        unsigned code=(unsigned)(id>>(c*chunk_bits))&mask;
        ulonglong2 entry=t[((unsigned)c<<chunk_bits)+code];
        b ^= entry.x;
        pm ^= entry.y & marked;
    }
    unsigned par=__popcll(pm)&1;
    unsigned par1=par^(__popcll(marked&phase_diff_mask)&1);
    unsigned long long b1=b^map_xor;
    float2 v0=weights[__popcll(b)*(m+1)+__popcll(b&outmask)];
    float2 v1=weights[__popcll(b1)*(m+1)+__popcll(b1&outmask)];
    float s0=1.0f-2.0f*(float)par, s1=1.0f-2.0f*(float)par1;
    return make_float4(s0*v0.x,s0*v0.y,s1*v1.x,s1*v1.y);
}
extern "C" __global__ void initcov_specialized(
    float2* v, unsigned long long off, unsigned long long n,
    unsigned long long outmask, const ulonglong2* t, const float2* weights,
    int m, int chunk_bits, int chunks) {
    for(unsigned long long j=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
        j<n;j+=(unsigned long long)gridDim.x*blockDim.x) {
        v[j]=initcov_specialized_value(j+off,outmask,t,weights,m,chunk_bits,chunks);
    }
}
'''


def finish_conditional(v, g, u, A):
    v = v.copy()
    walsh(v)
    quadratic_phase(v, g[np.ix_(A,A)])
    for j, site in enumerate(A):
        apply_one(v, u[site], j)
    return v


def selfcheck(out):
    root = Path(__file__).resolve().parent
    rng = np.random.default_rng(2026103101)
    records = []
    for name in ('induced_12q.npz', 'induced_24q.npz', 'induced_32q.npz',
                 'rgs_graph.npz', 'rgs_graph_alt_partition.npz'):
        d = np.load(root/name)
        g,u,A,B = d['graph'],d['measurement_unitaries'],d['A'].tolist(),d['B'].tolist()
        t0=time.perf_counter();spec=build_specialization(g,u,A,B);build=time.perf_counter()-t0
        ids = (np.arange(1 << spec.m, dtype=np.uint64) if spec.m<=16 else
               np.concatenate((np.array([0,(1<<spec.m)-1],np.uint64),
                   np.array([1<<j for j in range(spec.m)],np.uint64),
                   rng.integers(0,1<<spec.m,10000,dtype=np.uint64))))
        mapped, phase = spec.map_and_phase(ids)
        assert np.array_equal(mapped,map_indices(ids,spec.inverse_transpose))
        assert np.array_equal(phase,phase_for_ids(mapped,spec.graph_B))
        toggle_bit=spec.m-1
        map_xor,phase_mask=spec.toggle_metadata(toggle_bit)
        mapped1,phase1=spec.map_and_phase(ids ^ np.uint64(1<<toggle_bit))
        assert np.array_equal(mapped1,mapped ^ np.uint64(map_xor))
        phase_delta=1.-2.*parity64((ids|np.uint64(1<<63)) & np.uint64(phase_mask))
        assert np.array_equal(phase1,phase*phase_delta)
        for trial in range(3):
            ob = ([0]*spec.m if trial==0 else [1]*spec.m if trial==1 else rng.integers(0,2,spec.m).tolist())
            t0=time.perf_counter();v=spec.evaluate(ids,ob);special_seconds=time.perf_counter()-t0
            t0=time.perf_counter();ref=spec.generic(ids,ob);generic_seconds=time.perf_counter()-t0
            rel=float(np.linalg.norm(v-ref)/np.linalg.norm(ref))
            err=float(np.max(np.abs(v-ref)))
            assert rel<3e-14
            v32=spec.evaluate(ids,ob,np.complex64);ref32=spec.generic(ids,ob,np.complex64)
            rel32=float(np.linalg.norm(v32.astype(complex)-ref32.astype(complex))/np.linalg.norm(ref))
            assert rel32<2e-5
            rec=dict(fixture=name,trial=trial,ids_checked=len(ids),max_abs_error=err,
                relative_l2_error=rel,complex64_vs_old_fp32_relative_l2_error=rel32,
                generic_cpu_seconds=generic_seconds,specialized_cpu_seconds=special_seconds,
                setup_seconds=build,algebra_exact=True)
            if spec.m<=16:
                final=finish_conditional(v,g,u,A)
                oracle,pb=conditional(g,u,A,B,ob);oracle/=np.sqrt(pb)
                rec['conditional_max_abs_error']=float(np.max(abs(final-oracle)))
                rec['conditional_l2_error']=float(np.linalg.norm(final-oracle))
                rec['conditional_norm']=float(np.vdot(final,final).real)
                assert rec['conditional_l2_error']<2e-11
            records.append(rec)
    # Reject an almost-equal matrix: the optimization must never silently fit U.
    bad=u.copy();bad[B[-1],0,0]+=1e-15
    try:build_specialization(g,bad,A,B)
    except ValueError:pass
    else:raise AssertionError('near-equal U should be rejected')
    report=dict(status='CPU algebra and full conditional-vector checks passed; CUDA not executed',
        tests=records,operation_counts=spec.operation_counts(),
        actual_all_70_unitaries_exactly_equal=bool(np.array_equal(u,np.broadcast_to(u[0],u.shape))),
        floating_point='complex128 host powers then one cast; exact algebra, finite-precision arithmetic; no phase approximation')
    Path(out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default=str(Path(__file__).with_name('init_specialization_selfcheck.json')))
    args=p.parse_args();selfcheck(args.out)
