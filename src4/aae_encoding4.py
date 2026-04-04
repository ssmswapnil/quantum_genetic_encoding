"""
Step 2: Approximate Amplitude Encoding (AAE)
=============================================
Trains a parameterised quantum circuit (PQC) to approximate the
target amplitude vector produced by Step 1.

Ansatz:  Brickwall layout -- alternating layers of Ry rotations
         followed by staggered CX (CNOT) entangling gates.

Cost:    1 - |<psi_target | psi(theta)>|  (infidelity)

Optimizer: L-BFGS-B with multi-start (random restarts).
"""

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix


def build_brickwall_ansatz(n_qubits, n_layers, params):
    """Construct a brickwall PQC with Ry rotations and staggered CX gates."""
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        if layer % 2 == 0:
            for q in range(0, n_qubits - 1, 2):
                qc.cx(q, q + 1)
        else:
            for q in range(1, n_qubits - 1, 2):
                qc.cx(q, q + 1)
    return qc


def statevector_from_params(params, n_qubits, n_layers):
    """Return the complex statevector produced by the ansatz."""
    qc = build_brickwall_ansatz(n_qubits, n_layers, params)
    return np.array(Statevector.from_instruction(qc).data)


def cost_function(params, n_qubits, n_layers, target_state):
    """Infidelity: 1 - |<target|psi(theta)>|."""
    sv = statevector_from_params(params, n_qubits, n_layers)
    return 1.0 - abs(np.vdot(target_state, sv))


def train_pqc(n_qubits, n_layers, target_state, n_trials=8, maxiter=5000):
    """Train the brickwall PQC with random restarts. Returns (best_params, best_cost)."""
    n_params = n_qubits * n_layers
    best_params = None
    best_cost = float('inf')

    for trial in range(n_trials):
        params_init = np.random.uniform(0, 2 * np.pi, n_params)
        result = minimize(
            cost_function, params_init,
            args=(n_qubits, n_layers, target_state),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10},
        )
        overlap = 1 - result.fun
        tag = " <-- best" if result.fun < best_cost else ""
        print(f"    Trial {trial + 1:2d}/{n_trials}: "
              f"cost={result.fun:.10f}  overlap={overlap:.8f}  "
              f"iters={result.nit}{tag}")
        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x.copy()

    return best_params, best_cost


def aae_encode(step1_result, n_layers=6, n_trials=8, maxiter=5000):
    """Run AAE training and return the encoding result dictionary."""
    n_q = step1_result['num_qubits']
    d = step1_result['d_normalized']
    n_params = n_q * n_layers

    print(f"\n  Config: {n_q} qubits, {n_layers} layers, "
          f"{n_params} parameters, {n_trials} restarts")
    print(f"  Target state norm: {np.linalg.norm(d):.10f}")

    best_params, best_cost = train_pqc(n_q, n_layers, d, n_trials, maxiter)

    trained_circuit = build_brickwall_ansatz(n_q, n_layers, best_params)

    trained_circuit_meas = QuantumCircuit(n_q, n_q)
    trained_circuit_meas.compose(trained_circuit, inplace=True)
    trained_circuit_meas.measure(range(n_q), range(n_q))

    trained_sv = Statevector.from_instruction(trained_circuit)
    overlap = abs(np.vdot(d, trained_sv.data))

    gc = dict(trained_circuit.count_ops())

    return {
        'encoding_type': 'aae',
        'circuit': trained_circuit,
        'circuit_meas': trained_circuit_meas,
        'initial_sv': trained_sv,
        'initial_dm': DensityMatrix(trained_sv),
        'target_sv': Statevector(d),
        'target_dm': DensityMatrix(Statevector(d)),
        'num_qubits': n_q,
        'best_params': best_params,
        'best_cost': best_cost,
        'overlap': overlap,
        'n_layers': n_layers,
        'logical_cnot': gc.get('cx', 0),
        'logical_ry': gc.get('ry', 0),
        'logical_total': gc.get('cx', 0) + gc.get('ry', 0),
    }


def print_step2(step1_result, step2_result):
    """Print a readable summary of the AAE encoding."""
    n_q = step2_result['num_qubits']
    d = step1_result['d_normalized']
    sv = step2_result['initial_sv']

    print("\n" + "=" * 70)
    print("STEP 2: APPROXIMATE AMPLITUDE ENCODING (AAE)")
    print("=" * 70)
    print(f"\n  Qubits:    {n_q}")
    print(f"  Layers:    {step2_result['n_layers']}")
    print(f"  Params:    {n_q * step2_result['n_layers']}")
    print(f"  Gates:     {step2_result['logical_ry']} Ry + "
          f"{step2_result['logical_cnot']} CX = "
          f"{step2_result['logical_total']} logical")
    print(f"  Depth:     {step2_result['circuit'].depth()}")
    print(f"  Cost:      {step2_result['best_cost']:.10f}")
    print(f"  Overlap:   {step2_result['overlap']:.8f}")

    probs_target = d ** 2
    probs_actual = np.abs(sv.data) ** 2

    idx_to_codon = {e['unique_index']: e['codon']
                    for e in step1_result['unique_register']}

    indices = np.argsort(-probs_target)[:15]
    print(f"\n  {'Basis':>10}  {'p_target':>9}  {'p_actual':>9}  "
          f"{'|dp|':>9}  Codon")
    print(f"  {'-' * 10}  {'-' * 9}  {'-' * 9}  {'-' * 9}  {'-' * 6}")

    for idx in indices:
        delta = abs(probs_target[idx] - probs_actual[idx])
        codon = idx_to_codon.get(idx, '(pad)')
        binary = format(idx, f'0{n_q}b')
        print(f"  |{binary}>  {probs_target[idx]:9.6f}  "
              f"{probs_actual[idx]:9.6f}  {delta:9.6f}  {codon}")

    print(f"\n  Max |dp| (all states): "
          f"{np.max(np.abs(probs_target - probs_actual)):.6f}")
