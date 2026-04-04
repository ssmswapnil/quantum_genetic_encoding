"""
Step 3a: Simulation
====================
Transpile the trained AAE circuit and run on:
  1. AerSimulator (ideal, noiseless)
  2. FakeSherbrooke (IBM Eagle r3 noise model)

Outputs measurement counts plus density matrices for fidelity analysis.
"""

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit.quantum_info import Statevector, DensityMatrix


def get_circuit_metrics(transpiled_circuit):
    """Extract gate counts and depth from a transpiled circuit."""
    gc = dict(transpiled_circuit.count_ops())
    two_q = sum(v for k, v in gc.items()
                if k in ('cx', 'cnot', 'ecr', 'cz', 'swap', 'iswap'))
    return {
        'depth': transpiled_circuit.depth(),
        'total_gates': sum(gc.values()),
        'gate_counts': gc,
        'two_qubit_gates': two_q,
        'swap_count': gc.get('swap', 0),
    }


def _dm_from_counts(counts, num_qubits):
    """Build a diagonal density matrix from measurement counts."""
    n = 2 ** num_qubits
    arr = np.zeros((n, n), dtype=complex)
    total = sum(counts.values())
    for bs, c in counts.items():
        idx = int(bs, 2)
        if idx < n:
            arr[idx, idx] = c / total
    return DensityMatrix(arr)


def run_dual_simulation(encoding_result, shots=8192):
    """
    Run both Aer (ideal) and FakeSherbrooke (noisy) simulations.

    Parameters
    ----------
    encoding_result : dict from aae_encode()
    shots : int

    Returns
    -------
    dict with 'aer' and 'sherbrooke' sub-dicts, each containing
    counts, dm, metrics, shots.
    """
    qc = encoding_result['circuit']
    qc_m = encoding_result['circuit_meas']
    n_q = encoding_result['num_qubits']
    logical = {k: encoding_result[k]
               for k in ('logical_cnot', 'logical_ry', 'logical_total')}

    # Transpile for FakeSherbrooke
    backend = FakeSherbrooke()
    transpiled_meas = transpile(qc_m, backend=backend, optimization_level=3)
    t_metrics = get_circuit_metrics(transpiled_meas)

    print(f"\n  Logical gates: {logical['logical_cnot']} CX + "
          f"{logical['logical_ry']} Ry = {logical['logical_total']}")
    print(f"  Transpiled (Sherbrooke): depth={t_metrics['depth']}, "
          f"gates={t_metrics['total_gates']}, "
          f"2Q={t_metrics['two_qubit_gates']}, "
          f"SWAPs={t_metrics['swap_count']}")

    # Aer (ideal)
    aer_sim = AerSimulator()
    aer_transpiled = transpile(qc_m, backend=aer_sim, optimization_level=0)
    aer_counts = aer_sim.run(aer_transpiled, shots=shots).result().get_counts()

    try:
        aer_dm = DensityMatrix(Statevector.from_instruction(qc))
    except Exception:
        aer_dm = _dm_from_counts(aer_counts, n_q)

    # FakeSherbrooke (noisy)
    sherbrooke_counts = backend.run(transpiled_meas, shots=shots).result().get_counts()

    try:
        noise = NoiseModel.from_backend(backend)
        dm_sim = AerSimulator(method='density_matrix', noise_model=noise)
        qc_dm = qc.copy()
        qc_dm.save_density_matrix()
        dm_transpiled = transpile(qc_dm, backend=dm_sim, optimization_level=3)
        dm_data = dm_sim.run(dm_transpiled).result().data()['density_matrix']
        sherbrooke_dm = DensityMatrix(dm_data)
    except Exception:
        sherbrooke_dm = _dm_from_counts(sherbrooke_counts, n_q)

    metrics = {**logical, **t_metrics}
    return {
        'aer': {
            'counts': aer_counts, 'dm': aer_dm,
            'metrics': metrics, 'shots': shots,
        },
        'sherbrooke': {
            'counts': sherbrooke_counts, 'dm': sherbrooke_dm,
            'metrics': metrics, 'shots': shots,
        },
    }
