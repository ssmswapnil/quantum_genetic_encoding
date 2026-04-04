"""
Step 3b: Reconstruction
========================
Rebuild the DNA sequence from quantum measurement counts
and the classical position register.
"""

import numpy as np


def reconstruct_dna(counts, step1_result, num_qubits, shots):
    """Reconstruct the DNA string from measurement results."""
    n_states = 2 ** num_qubits
    probs = np.zeros(n_states, dtype=float)
    total = sum(counts.values())
    for bs, c in counts.items():
        idx = int(bs, 2)
        if idx < n_states:
            probs[idx] = c / total

    reconstructed = ''.join(
        e['codon'] for e in step1_result['position_register']
    )

    return {
        'reconstructed_dna': reconstructed,
        'probabilities': probs,
    }


def compute_accuracy(original, reconstructed):
    """Compare original vs reconstructed DNA strings."""
    min_len = min(len(original), len(reconstructed))
    matches = sum(1 for a, b in zip(original[:min_len], reconstructed[:min_len])
                  if a == b)
    accuracy = matches / len(original) if len(original) > 0 else 0.0

    return {
        'exact_match': reconstructed == original,
        'char_accuracy': accuracy,
        'char_matches': matches,
        'original_length': len(original),
        'reconstructed_length': len(reconstructed),
    }
