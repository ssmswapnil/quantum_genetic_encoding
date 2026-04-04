"""
Approximate Amplitude Encoding (AAE) Pipeline -- HBB Beta-Globin
================================================================
PhysisTechne Symposium 2026 -- Quantum Computing Track

Pipeline:
  Step 1  Classical bit register (codon division + frequency counting)
  Step 2  AAE training (brickwall PQC + L-BFGS multi-restart)
  Step 3  Dual simulation (Aer ideal + FakeSherbrooke noisy)
          + DNA reconstruction + multi-level fidelity

Target sequence: Human beta-globin (HBB) locus, ~1.6 kb
"""

import os
import sys
import json
import time
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src4.compression4 import DNA_SEQUENCE, build_classical_register, print_step1
from src4.aae_encoding4 import aae_encode, print_step2
from src4.simulation4 import run_dual_simulation
from src4.reconstruction4 import reconstruct_dna, compute_accuracy
from src4.fidelity4 import compute_all_fidelities

# -- Hyperparameters -------------------------------------------------------
N_LAYERS = 6          # brickwall ansatz depth
N_TRIALS = 3          # random restarts for L-BFGS
MAXITER = 5000        # max iterations per trial
SHOTS = 8192          # measurement shots
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')


# -- Step 3 wrapper --------------------------------------------------------

def run_step3(s1, s2):
    """Simulate, reconstruct, and report."""
    print("\n" + "=" * 70)
    print("STEP 3: SIMULATION + RECONSTRUCTION")
    print("=" * 70)

    sim = run_dual_simulation(s2, shots=SHOTS)

    # Top-5 measurement outcomes
    for label, key in [("Aer (ideal)", 'aer'),
                       ("FakeSherbrooke (noisy)", 'sherbrooke')]:
        top5 = Counter(sim[key]['counts']).most_common(5)
        print(f"\n  {label} -- Top 5: {dict(top5)}")

    # Reconstruction
    n_q = s2['num_qubits']
    aer_recon = reconstruct_dna(sim['aer']['counts'], s1, n_q, SHOTS)
    sher_recon = reconstruct_dna(sim['sherbrooke']['counts'], s1, n_q, SHOTS)
    aer_acc = compute_accuracy(DNA_SEQUENCE, aer_recon['reconstructed_dna'])
    sher_acc = compute_accuracy(DNA_SEQUENCE, sher_recon['reconstructed_dna'])

    print(f"\n  Reconstruction:")
    print(f"    Aer:        {'PASS' if aer_acc['exact_match'] else 'FAIL'} "
          f"({aer_acc['char_accuracy']:.2%})")
    print(f"    Sherbrooke: {'PASS' if sher_acc['exact_match'] else 'FAIL'} "
          f"({sher_acc['char_accuracy']:.2%})")

    return {
        'aer': sim['aer'],
        'sherbrooke': sim['sherbrooke'],
        'aer_recon': {**aer_recon, **aer_acc},
        'sherbrooke_recon': {**sher_recon, **sher_acc},
    }


# -- Main ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  APPROXIMATE AMPLITUDE ENCODING -- HBB (beta-globin)")
    print("  PhysisTechne Symposium 2026")
    print("=" * 70)
    print(f"\n  Target: {DNA_SEQUENCE[:60]}...")
    print(f"  Length: {len(DNA_SEQUENCE)} bases")
    print(f"  Ansatz: Brickwall ({N_LAYERS} layers) | "
          f"Optimizer: L-BFGS | Trials: {N_TRIALS}\n")

    t0 = time.time()

    # Step 1
    s1 = build_classical_register(DNA_SEQUENCE)
    print_step1(s1)

    # Step 2
    print("\n" + "=" * 70)
    print("STEP 2: AAE TRAINING")
    print("=" * 70)
    s2 = aae_encode(s1, n_layers=N_LAYERS, n_trials=N_TRIALS, maxiter=MAXITER)
    print_step2(s1, s2)

    # Step 3
    s3 = run_step3(s1, s2)

    # Fidelity
    fid = compute_all_fidelities(s1, s2, s3)

    elapsed = time.time() - t0
    m = s3['sherbrooke']['metrics']

    # -- Final summary -----------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"""
  DNA:        {len(DNA_SEQUENCE)} bases | {s1['num_codons']} codons | {s1['num_unique']} unique

  Encoding:   {s2['num_qubits']} qubits, {s2['n_layers']} layers, {s2['num_qubits'] * s2['n_layers']} params
  Gates:      {s2['logical_ry']} Ry + {s2['logical_cnot']} CNOT = {s2['logical_total']} logical
  Transpiled: depth {m['depth']}, {m['total_gates']} gates, {m['two_qubit_gates']} two-qubit

  Overlap:                  {s2['overlap']:.8f}
  F(target,  trained):      {fid['f_target_trained']:.6f}
  F(trained, Sherbrooke):   {fid['f_trained_sherbrooke']:.6f}
  F(target,  Sherbrooke):   {fid['f_target_sherbrooke']:.6f}
  Noise drop:               {fid['noise_drop']:.6f}

  Reconstruction:
    Aer:        {'PASS' if s3['aer_recon']['exact_match'] else 'FAIL'}
    Sherbrooke: {'PASS' if s3['sherbrooke_recon']['exact_match'] else 'FAIL'}

  Runtime: {elapsed:.1f}s
""")

    # -- Save results ------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {
        'pipeline': 'src4_aae_hbb',
        'sequence_length': len(DNA_SEQUENCE),
        'num_codons': s1['num_codons'],
        'unique_codons': s1['num_unique'],
        'qubits': s2['num_qubits'],
        'ansatz_layers': s2['n_layers'],
        'params': s2['num_qubits'] * s2['n_layers'],
        'logical_gates': s2['logical_total'],
        'transpiled_depth': m['depth'],
        'transpiled_gates': m['total_gates'],
        'transpiled_2q': m['two_qubit_gates'],
        'overlap': float(s2['overlap']),
        'best_cost': float(s2['best_cost']),
        'f_target_trained': float(fid['f_target_trained']),
        'f_trained_aer': float(fid['f_trained_aer']),
        'f_trained_sherbrooke': float(fid['f_trained_sherbrooke']),
        'f_target_sherbrooke': float(fid['f_target_sherbrooke']),
        'noise_drop': float(fid['noise_drop']),
        'recon_aer_pass': s3['aer_recon']['exact_match'],
        'recon_sherbrooke_pass': s3['sherbrooke_recon']['exact_match'],
        'shots': SHOTS,
        'runtime_seconds': round(elapsed, 1),
    }

    out_path = os.path.join(RESULTS_DIR, 'summary_aae_hbb.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
