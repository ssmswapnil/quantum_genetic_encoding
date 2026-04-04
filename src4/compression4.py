"""
Step 1: Classical Bit Register -- Codon division, frequency counting,
weight vector, and normalised target distribution for AAE training.
Data source: Human beta-globin (HBB) gene, 1,608 bp.
"""
import os, numpy as np
from collections import Counter, OrderedDict

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
_SEQ_FILE = os.path.join(_DATA_DIR, 'hbb_sequence.txt')
with open(_SEQ_FILE, 'r') as _f:
    DNA_SEQUENCE = _f.read().strip().replace('\n','').replace('\r','').replace(' ','')

def divide_into_codons(sequence):
    return [sequence[i:i+3] for i in range(0, len(sequence), 3)]

def build_classical_register(sequence):
    codon_sequence = divide_into_codons(sequence)
    freq = Counter(codon_sequence)
    seen = OrderedDict()
    for codon in codon_sequence:
        if codon not in seen:
            seen[codon] = len(seen)
    n_unique = len(seen)
    n_qubits = int(np.ceil(np.log2(max(n_unique, 2))))
    n_states = 2 ** n_qubits

    unique_register = []
    for codon, idx in seen.items():
        unique_register.append({
            'unique_index': idx, 'codon': codon,
            'weight': freq[codon], 'binary': format(idx, f'0{n_qubits}b'),
        })
    position_register = []
    for pos, codon in enumerate(codon_sequence):
        position_register.append({
            'position': pos, 'codon': codon,
            'unique_index': seen[codon],
            'binary': format(seen[codon], f'0{n_qubits}b'),
        })

    weight_vector = np.zeros(n_states, dtype=float)
    for entry in unique_register:
        weight_vector[entry['unique_index']] = entry['weight']
    d = weight_vector.copy()
    norm = np.linalg.norm(d)
    if norm > 0: d /= norm
    p_comp = d ** 2

    d_H = np.zeros(n_states, dtype=float)
    for j in range(n_states):
        val = 0.0
        for k in range(n_states):
            val += d[k] * ((-1) ** bin(j & k).count('1'))
        d_H[j] = val / np.sqrt(n_states)
    p_hadamard = d_H ** 2

    return {
        'sequence': sequence, 'codon_sequence': codon_sequence,
        'num_codons': len(codon_sequence), 'unique_codons': seen,
        'num_unique': n_unique, 'weights': dict(freq),
        'unique_register': unique_register, 'position_register': position_register,
        'num_qubits': n_qubits, 'weight_vector': weight_vector,
        'd_normalized': d, 'p_comp': p_comp,
        'd_hadamard': d_H, 'p_hadamard': p_hadamard,
    }

def print_step1(result):
    seq, n_q, d = result['sequence'], result['num_qubits'], result['d_normalized']
    print("=" * 70)
    print("STEP 1: CLASSICAL BIT REGISTER  (HBB beta-globin)")
    print("=" * 70)
    print(f"\n  Sequence:       {seq[:60]}...")
    print(f"  Length:          {len(seq)} bases")
    print(f"  Total codons:   {result['num_codons']}")
    print(f"  Unique codons:  {result['num_unique']}")
    print(f"  Qubits needed:  {n_q}   (Hilbert space 2^{n_q} = {2**n_q})")
    print(f"  Unused states:  {2**n_q - result['num_unique']}")
    print(f"  Qubit reduction: {(1 - n_q / (len(seq)*2))*100:.1f}%")
    sorted_reg = sorted(result['unique_register'], key=lambda e: e['weight'], reverse=True)
    print(f"\n  Top-15 codons by frequency:")
    print(f"  {'Rank':>4}  {'Codon':>6}  {'Binary':>{n_q}}  {'Freq':>5}  {'p(j)':>9}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*n_q}  {'-'*5}  {'-'*9}")
    for rank, e in enumerate(sorted_reg[:15], 1):
        idx = e['unique_index']
        print(f"  {rank:4d}  {e['codon']:>6}  {e['binary']}  {e['weight']:5d}  {d[idx]**2:9.6f}")
    if len(sorted_reg) > 15:
        print(f"  ... ({len(sorted_reg)-15} more unique codons)")
