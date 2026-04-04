"""
Fidelity & Distribution Metrics
================================
Four-level fidelity chain plus KL-divergence and total-variation
distance between target and measured distributions.
"""

import numpy as np
from qiskit.quantum_info import state_fidelity


def _kl_divergence(p, q, eps=1e-12):
    """KL(p || q) with smoothing to avoid log(0)."""
    p_s = np.clip(p, eps, None)
    q_s = np.clip(q, eps, None)
    p_s = p_s / p_s.sum()
    q_s = q_s / q_s.sum()
    return float(np.sum(p_s * np.log(p_s / q_s)))


def _tvd(p, q):
    """Total variation distance: 0.5 * sum|p_i - q_i|."""
    return float(0.5 * np.sum(np.abs(p - q)))


def compute_all_fidelities(step1_result, step2_result, step3_result):
    """
    Compute the full fidelity chain and distribution metrics.

    Fidelity chain:
      F1: F(target, trained)      -- training quality
      F2: F(trained, Aer)         -- sanity (should ~ 1)
      F3: F(trained, Sherbrooke)  -- noise impact
      F4: F(target, Sherbrooke)   -- end-to-end

    Distribution metrics (computational basis):
      KL(target || Aer),  KL(target || Sherbrooke)
      TVD(target, Aer),   TVD(target, Sherbrooke)
    """
    trained_dm = step2_result['initial_dm']
    target_dm = step2_result['target_dm']
    aer_dm = step3_result['aer']['dm']
    sherbrooke_dm = step3_result['sherbrooke']['dm']

    f_tt = state_fidelity(target_dm, trained_dm)
    f_ta = state_fidelity(trained_dm, aer_dm) if aer_dm is not None else 0.0
    f_ts = (state_fidelity(trained_dm, sherbrooke_dm)
            if sherbrooke_dm is not None else 0.0)
    f_es = (state_fidelity(target_dm, sherbrooke_dm)
            if sherbrooke_dm is not None else 0.0)

    # Distribution-level metrics
    p_target = step1_result['p_comp']
    p_aer = (step3_result['aer_recon']['probabilities']
             if 'aer_recon' in step3_result else None)
    p_sher = (step3_result['sherbrooke_recon']['probabilities']
              if 'sherbrooke_recon' in step3_result else None)

    kl_aer = _kl_divergence(p_target, p_aer) if p_aer is not None else None
    kl_sher = _kl_divergence(p_target, p_sher) if p_sher is not None else None
    tvd_aer = _tvd(p_target, p_aer) if p_aer is not None else None
    tvd_sher = _tvd(p_target, p_sher) if p_sher is not None else None

    print("\n" + "=" * 70)
    print("FIDELITY & DISTRIBUTION METRICS")
    print("=" * 70)
    print(f"\n  Quantum state fidelity:")
    print(f"    F(target, trained)       = {f_tt:.8f}   (training quality)")
    print(f"    F(trained, Aer)          = {f_ta:.8f}   (sanity check)")
    print(f"    F(trained, Sherbrooke)   = {f_ts:.8f}   (noise impact)")
    print(f"    F(target, Sherbrooke)    = {f_es:.8f}   (end-to-end)")
    print(f"    Noise drop               = {f_ta - f_ts:.8f}")

    if kl_aer is not None:
        print(f"\n  Distribution divergence:")
        print(f"    KL(target || Aer)        = {kl_aer:.6f}")
        print(f"    KL(target || Sherbrooke) = {kl_sher:.6f}")
        print(f"    TVD(target, Aer)         = {tvd_aer:.6f}")
        print(f"    TVD(target, Sherbrooke)  = {tvd_sher:.6f}")

    return {
        'f_target_trained': f_tt,
        'f_trained_aer': f_ta,
        'f_trained_sherbrooke': f_ts,
        'f_target_sherbrooke': f_es,
        'noise_drop': f_ta - f_ts,
        'overlap': step2_result['overlap'],
        'kl_aer': kl_aer,
        'kl_sherbrooke': kl_sher,
        'tvd_aer': tvd_aer,
        'tvd_sherbrooke': tvd_sher,
    }
