"""Figure for the exact-sampling submission. Run from the repository root: python born/figures/make_fig.py (needs matplotlib)."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
WORK, BORN, D = ROOT / 'work' / 'rgs_attack', ROOT / 'born', 2.0 ** 70


def zs(path, key, marker):
    return np.array([D * r[key] for r in map(json.loads, Path(path).read_text().splitlines()) if marker in r])


batch1 = np.concatenate([zs(BORN / 'batch1' / f'born_group{g}_fp64.jsonl', 'probability', 'index') for g in range(4)])
batch2 = np.concatenate([zs(BORN / 'batch2' / f'born2_group{g}_fp64.jsonl', 'probability', 'index') for g in range(4)])
ibm = zs(WORK / 'cap1_archive' / 'ibm_scored_fp64.jsonl', 'probability', 'index')
uniform = np.concatenate([zs(WORK / p, 'probability_raw', 'proposal') for p in
                          ('cap1_archive/capped1726.jsonl', 'cap8_archive/cap8_part0.jsonl', 'cap8_archive/cap8_part1.jsonl')])
ours = np.concatenate([batch1, batch2])

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 12, 'xtick.direction': 'in', 'ytick.direction': 'in'})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'wspace': 0.25})
red, orange, blue, gray = '#c0392b', '#e67e22', '#2471a3', '#7f8c8d'
bins = np.linspace(0, 10, 26)
centers, widths = 0.5 * (bins[1:] + bins[:-1]), np.diff(bins)
for data, color, label, marker in ((ours, red, 'This work, 3,452 samples (two batches)', 'o'),
                                   (ibm, blue, 'IBM hardware, 1,726 samples', 's'),
                                   (uniform, gray, 'Uniform bitstrings, 16,442', '^')):
    counts, _ = np.histogram(data, bins)
    keep = counts > 0
    ax.errorbar(centers[keep], (counts / (len(data) * widths))[keep], yerr=(np.sqrt(counts) / (len(data) * widths))[keep],
                fmt=marker, ms=5, color=color, label=label, lw=1)
x = np.linspace(0, 10, 400)
ax.plot(x, x * np.exp(-x), color=red, lw=1.2, label=r'$z\,e^{-z}$ (ideal sampler)')
ax.plot(x, 0.466 * x * np.exp(-x) + 0.534 * np.exp(-x), color=blue, lw=1.0, ls='--', label=r'$Fze^{-z}+(1-F)e^{-z}$, $F=0.466$')
ax.plot(x, np.exp(-x), color=gray, lw=1.0, ls=':', label=r'$e^{-z}$ (uniform)')
ax.set(yscale='log', xlim=(0, 10), ylim=(1e-4, 8), xlabel=r'$z = 2^{70}\,p(x)$', ylabel='probability density')
ax.legend(fontsize=9.5, frameon=False, loc='upper right')
ax.text(0.03, 0.93, '(a)', transform=ax.transAxes)
for data, color, ls, label in ((batch1, red, '-', 'This work, batch 1 (timed)'), (batch2, orange, '-', 'This work, batch 2'),
                               (ibm, blue, '-', 'IBM hardware')):
    n = np.arange(1, len(data) + 1)
    mean = np.cumsum(data) / n - 1
    se = np.array([data[:i].std(ddof=1) if i > 1 else np.nan for i in n]) / np.sqrt(n)
    bx.plot(n, mean, color=color, ls=ls, lw=1.4, label=label)
    bx.fill_between(n, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.15, lw=0)
bx.axhline(1.0, color='k', lw=0.8, ls='--')
bx.text(1740, 1.0, ' ideal', va='center', fontsize=10)
bx.axhline(0.092, color='k', lw=0.8, ls=':')
bx.text(1740, 0.092, ' 0.092', va='center', fontsize=10)
bx.set(xlim=(0, 1726), ylim=(-0.1, 1.8), xlabel='number of samples', ylabel='linear XEB (cumulative)')
bx.legend(fontsize=10, frameon=False, loc='upper right')
bx.text(0.03, 0.93, '(b)', transform=bx.transAxes)
out = BORN / 'figures' / 'rgs70_born_xeb.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print('saved', out)
