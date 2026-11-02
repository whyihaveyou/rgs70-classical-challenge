"""CPU test of the production draw logic (DeferredGlobalMixin.draw -> DistributedSampler.draw)
against exact conditional Born distributions of an induced fixture. The GPU arrays are replaced by
NumPy complex64 shards holding the exact pre-terminal-gate conditional state; the selection code
itself is the unmodified production code."""
import sys, types, contextlib, numpy as np
from scipy import stats
from distributed_sampler import DistributedSampler
from deferred_global_sampler import DeferredGlobalMixin, cpu_deferred

class FakeCuda:
    @staticmethod
    @contextlib.contextmanager
    def Device(dev): yield
fake_cp = types.SimpleNamespace(cuda=FakeCuda, asnumpy=lambda a: np.asarray(a))

class CPUEngine(DeferredGlobalMixin, DistributedSampler):
    def __init__(self, raw, unitary, tilebits):
        self.cp = fake_cp; self.devices = [0, 1]; self.ng = 2
        self.n = len(raw) // 2; self.m = int(np.log2(len(raw)))
        self.arr = [raw[:self.n].astype(np.complex64), raw[self.n:].astype(np.complex64)]
        self.deferred_global_unitary = np.asarray(unitary, np.complex128); self.tilebits = tilebits
    def masses(self):  # NumPy replica of the blockmass kernel (float64 tile sums), cached per state
        if not hasattr(self, '_mass'):
            t = self.tilebits; nmass = (self.n + (1 << t) - 1) >> t
            mass = np.concatenate([(np.abs(a.astype(np.complex128)) ** 2).reshape(nmass, -1).sum(1) for a in self.arr])
            self._mass = (mass, float(mass.sum()), t, nmass)
        return self._mass

