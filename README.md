# 🧬 Quantum Encoding of Genetic Information

<p align="center">
  <b>PhysisTechne Symposium 2026 — Quantum Computing Track</b><br>
  <i>Encoding DNA sequences into quantum circuits using three encoding strategies,<br>
  benchmarked on IBM's FakeSherbrooke 127-qubit Eagle processor</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Qiskit-%3E%3D1.0-6929C4?logo=qiskit" alt="Qiskit">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Backend-FakeSherbrooke-054ADA" alt="Backend">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## 📌 Overview

This project implements a complete pipeline for encoding DNA sequences into optimized quantum circuits. DNA is divided into **codons** (triplets), their frequencies are computed, and the resulting weight distribution is encoded into quantum states using three different strategies — each with different qubit, gate, and fidelity tradeoffs.

Two pipelines are provided:

| Pipeline | Entry Point | Encodings | Target Sequence |
|:---|:---|:---|:---|
| **Pipeline 1** | `main.py` | Amplitude + Angle | 50-base sequence |
| **Pipeline 2** | `main_aae.py` | Approximate Amplitude (AAE) | 12,001-base Rhesus macaque chr16 |

---

## 🏗️ Pipeline Architecture

### High-Level Flow

```mermaid
flowchart LR
    A[🧬 DNA Sequence] --> B[Step 1: Codon Division]
    B --> C[Classical Bit Register]
    C --> D[Step 2: Quantum Encoding]
    D --> E[Step 3: Simulation]
    E --> F[Aer - Ideal]
    E --> G[FakeSherbrooke - Noisy]
    F --> H[Fidelity Comparison]
    G --> H
    H --> I[📊 Results]
```

### Step 1 — Classical Bit Register

```mermaid
flowchart TD
    A["ATGCGTACG...TTAGC (50 bases)"] --> B["Divide into codons (size 3)"]
    B --> C["['ATG', 'CGT', 'ACG', 'TTA', ...]"]
    C --> D["Count frequencies (weights)"]
    D --> E["CGT: 3, ACG: 2, TTA: 2, GAT: 2, ..."]
    E --> F["Unique Register (12 entries)\nindex → codon → weight"]
    E --> G["Position Register (17 entries)\nposition → codon → index"]
```

### Step 2 — Three Encoding Strategies

```mermaid
flowchart TD
    subgraph AMP["Amplitude Encoding"]
        A1["Weights → amplitudes of basis states"]
        A2["|ψ⟩ = (1/N) Σᵢ wᵢ|i⟩"]
        A3["4 qubits | 8 CNOT + 8 Ry = 16 gates"]
    end
    
    subgraph ANG["Angle Encoding"]
        B1["Each codon → own qubit with Ry rotation"]
        B2["|ψ⟩ = ⊗ᵢ Ry(θᵢ)|0⟩"]
        B3["12 qubits | 12 Ry + 0 CNOT | depth 1"]
    end
    
    subgraph AAE["Approximate Amplitude Encoding"]
        C1["Train shallow PQC variationally"]
        C2["Brickwall ansatz + L-BFGS optimizer"]
        C3["7 qubits | 42 Ry + 18 CNOT | depth 12"]
    end
```

### Step 3 — Dual Backend Simulation

```mermaid
flowchart LR
    A[Encoded Circuit] --> B[Transpile for Sherbrooke]
    B --> C["Aer Simulator\n(ideal, no noise)"]
    B --> D["FakeSherbrooke\n(127-qubit Eagle noise)"]
    C --> E["Density Matrix\n(exact statevector)"]
    D --> F["Density Matrix\n(noisy simulation)"]
    E --> G["State Fidelity\nF(ρ_ideal, ρ_noisy)"]
    F --> G
```

---

## 🔬 Encoding Strategies

### Amplitude Encoding

Encodes codon weights as **amplitudes of computational basis states**. Uses `initialize()` which decomposes into `2^(n-1)` CNOT gates and `2^(n-1)` Ry rotation gates.

```
|ψ⟩ = (1/N)(w₀|0000⟩ + w₁|0001⟩ + w₂|0010⟩ + ...)
```

| Property | Value |
|:---|:---|
| Qubits | `ceil(log₂(N_unique))` |
| CNOT gates | `2^(n-1)` |
| Ry gates | `2^(n-1)` |
| Encoding | Exact |
| Entanglement | Yes |

### Angle Encoding

Encodes each codon weight as an **Ry rotation angle** on its own dedicated qubit. Weights are rescaled to `(0, 2π]` to prevent information loss.

```
|ψ⟩ = Ry(θ₀)|0⟩ ⊗ Ry(θ₁)|0⟩ ⊗ ... ⊗ Ry(θₙ)|0⟩
```

| Property | Value |
|:---|:---|
| Qubits | `N_unique` (one per codon) |
| CNOT gates | 0 |
| Ry gates | `N_unique` |
| Depth | 1 |
| Entanglement | None (product state) |

### Approximate Amplitude Encoding (AAE)

Trains a **shallow parameterized quantum circuit** (brickwall ansatz) to approximate the target amplitude distribution using variational optimization.

```
U(θ)|0⟩ ≈ |target⟩     minimizing C(θ) = 1 - Re⟨target|U(θ)|0⟩
```

| Property | Value |
|:---|:---|
| Ansatz | Brickwall (alternating CNOT pairs) |
| Optimizer | L-BFGS (quasi-Newton) |
| Training | Statevector simulation (exact) |
| Depth | O(poly(log N)) |
| Scalable | Yes |

**Brickwall Ansatz Structure:**

```
Layer 1:  ─Ry─╥─Ry─╥─Ry─╥─Ry─╥─Ry─╥─Ry─╥─Ry─
               ║    ║    ║    ║    ║    ║    ║
          ─────╨────╨────╨────╨────╨────╨────╨──
          CNOT: (0,1)  (2,3)  (4,5)          [even pairs]

Layer 2:  ─Ry──Ry──Ry──Ry──Ry──Ry──Ry─
          CNOT:   (1,2)  (3,4)  (5,6)        [odd pairs]
```

---

## 📊 Results

### Pipeline 1 — Amplitude vs Angle Encoding (50 bases)

| Metric | Amplitude | Angle |
|:---|---:|---:|
| Qubits | 4 | 12 |
| Logical CNOT gates | 8 | 0 |
| Logical Ry gates | 8 | 12 |
| Total logical gates | 16 | 12 |
| Transpiled depth | 67 | 5 |
| Two-qubit gates (transpiled) | 15 | 0 |
| **F(initial, Aer)** | **1.000** | **1.000** |
| **F(initial, Sherbrooke)** | **0.959** | **0.985** |
| Noise drop | 0.041 | 0.015 |
| Reconstruction | 100% | 100% |

> Angle encoding achieves higher fidelity (0.985 vs 0.959) due to zero two-qubit gates and depth 1, but requires 3× more qubits.

### Pipeline 2 — AAE on 12,001-Base Sequence

| Metric | Value |
|:---|---:|
| Sequence length | 12,001 bases |
| Total codons | 4,001 |
| Unique codons | 65 |
| Qubits | 7 |
| Ansatz layers | 6 |
| Trainable parameters | 42 |
| Logical gates | 60 (42 Ry + 18 CNOT) |
| Transpiled depth | 39 |
| Two-qubit gates (transpiled) | 18 |
| **Overlap O** | **0.973** |
| **F(target, trained)** | **0.947** |
| **F(trained, Sherbrooke)** | **0.941** |
| **F(target, Sherbrooke)** | **0.890** |
| Noise drop | 0.060 |
| Reconstruction | 100% |
| Runtime | 362s |

> 12,001 bases encoded into a **7-qubit, depth-39 circuit** with **89% end-to-end fidelity** on a noisy 127-qubit backend.

### Fidelity Breakdown (AAE)

```mermaid
flowchart LR
    A["Target |Data⟩"] -- "F = 0.947\n(training quality)" --> B["Trained U(θ)|0⟩"]
    B -- "F = 1.000\n(ideal)" --> C["Aer Output"]
    B -- "F = 0.941\n(noise impact)" --> D["Sherbrooke Output"]
    A -- "F = 0.890\n(end-to-end)" --> D
```

---

## 📁 Project Structure

```
├── main.py                    # Pipeline 1: Amplitude + Angle encoding (50 bases)
├── main_aae.py                # Pipeline 2: AAE encoding (12,001 bases)
├── requirements.txt
├── LICENSE
│
├── src/                       # Pipeline 1 modules
│   ├── compression.py         #   Step 1: Codon division + classical register
│   ├── encoding.py            #   Step 2: Amplitude & angle encoding
│   ├── simulation.py          #   Step 3: Aer + FakeSherbrooke simulation
│   ├── reconstruction.py      #   DNA reconstruction via classical register
│   └── fidelity.py            #   Fidelity calculations
│
├── src2/                      # Pipeline 2 modules (AAE)
│   ├── compression2.py        #   Step 1: Codon division + target distributions
│   ├── aae_encoding.py        #   Step 2: Brickwall ansatz + L-BFGS training
│   ├── simulation2.py         #   Step 3: Dual backend simulation
│   ├── reconstruction2.py     #   DNA reconstruction
│   └── fidelity2.py           #   Fidelity (target vs trained vs noisy)
│
├── data/
│   └── dna_12000.txt          # 12,001-base Rhesus macaque chr16 fragment
│
└── results/
    ├── summary.json            # Pipeline 1 output
    └── summary_aae.json        # Pipeline 2 output
```

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/quantum-dna-encoding.git
cd quantum-dna-encoding
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r requirements.txt

# Run Pipeline 1: Amplitude + Angle encoding (50 bases)
python main.py

# Run Pipeline 2: AAE encoding (12,001 bases)
python main_aae.py
```

---

## 🧪 DNA Sequences

**Pipeline 1 (50 bases):**
```
ATGCGTACGTTAGCGTACGATCGTAGCTAGCTTGACGATCGTACGTTAGC
```

**Pipeline 2 (12,001 bases):**
Rhesus macaque (*Macaca mulatta*) chromosome 16 fragment — `NC_133421.1:91056922-91068922`, gene LOC144335571

---

## 📚 References

1. IBM Quantum Learning — [Data Encoding](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/data-encoding)
2. Nakaji et al. — [Approximate Amplitude Encoding in Shallow Parameterized Quantum Circuits](https://doi.org/10.1103/PhysRevResearch.4.023136), Phys. Rev. Research **4**, 023136 (2022)
3. IBM Qiskit — [FakeSherbrooke Backend](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake_provider)

---

## 📄 License

[MIT](LICENSE)
