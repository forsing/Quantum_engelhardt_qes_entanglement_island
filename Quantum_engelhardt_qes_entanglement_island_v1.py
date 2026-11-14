# Quantum_engelhardt_qes_entanglement_island 



import random
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_2981.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
RADIATION_WINDOW = 2
ISLAND_WINDOW = 6
TOTAL_HORIZON = RADIATION_WINDOW + ISLAND_WINDOW
AREA_NEWTON_G_GRID = (0.35, 0.75, 1.35)
PAGE_SCRAMBLE_GRID = (0.40, 0.85)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < TOTAL_HORIZON + 35:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def build_hawking_scrambling_unitaries() -> tuple[np.ndarray, np.ndarray]:
    qc_qft = QuantumCircuit(NUM_QUBITS)
    qc_qft.append(QFTGate(NUM_QUBITS), list(range(NUM_QUBITS)))
    u_qft = Operator(qc_qft).data

    qc_ent = QuantumCircuit(NUM_QUBITS)
    for q in range(NUM_QUBITS - 1):
        qc_ent.cx(q, q + 1)
    qc_ent.cx(NUM_QUBITS - 1, 0)
    u_ent = Operator(qc_ent).data

    return u_qft, u_ent


def fit_horizon_area_tension(draws_train: np.ndarray) -> np.ndarray:
    n_tr = len(draws_train)
    occ = np.zeros((n_tr, MAX_NUM), dtype=np.float64)
    occ[np.arange(n_tr)[:, None], draws_train - 1] = 1.0

    freq = np.mean(occ, axis=0)
    area_tension = -np.log(np.maximum(freq, 1e-4))
    area_tension = area_tension / max(float(np.mean(area_tension)), 1e-12)
    return area_tension


def compute_qes_entanglement_wedge_scores(
    draws: np.ndarray,
    area_tension: np.ndarray,
    u_qft: np.ndarray,
    u_ent: np.ndarray,
    inv_four_g: float,
    scramble_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_total = len(draws)
    n_samples = n_total - TOTAL_HORIZON
    n_all_win = n_samples + 1

    occ_all = np.zeros((n_total, MAX_NUM), dtype=np.float64)
    occ_all[np.arange(n_total)[:, None], draws - 1] = 1.0

    rad_weights = np.linspace(0.6, 1.0, num=RADIATION_WINDOW, dtype=np.float64)
    rad_weights = rad_weights / np.sum(rad_weights)

    isl_weights = np.exp(np.linspace(-1.4, 0.0, num=ISLAND_WINDOW, dtype=np.float64))
    isl_weights = isl_weights / np.sum(isl_weights)

    psi_rad = np.zeros((n_all_win, STATE_DIM), dtype=np.complex128)
    psi_isl = np.zeros((n_all_win, STATE_DIM), dtype=np.complex128)

    rad_occ_batch = np.empty((n_all_win, MAX_NUM), dtype=np.float64)
    isl_occ_batch = np.empty((n_all_win, MAX_NUM), dtype=np.float64)

    mode_grid = np.arange(MAX_NUM, dtype=np.float64) / float(MAX_NUM)
    phase_i = np.exp(1j * scramble_strength * np.pi * mode_grid)
    phase_r = np.exp(-1j * scramble_strength * np.pi * mode_grid)

    for idx in range(n_all_win):
        isl_slice = occ_all[idx : idx + ISLAND_WINDOW]
        rad_slice = occ_all[idx + ISLAND_WINDOW : idx + TOTAL_HORIZON]

        isl_profile = np.sum(isl_slice * isl_weights[:, None], axis=0)
        rad_profile = np.sum(rad_slice * rad_weights[:, None], axis=0)

        isl_occ_batch[idx] = isl_profile
        rad_occ_batch[idx] = rad_profile

        psi_isl[idx, :MAX_NUM] = np.sqrt(np.maximum(isl_profile, 0.0)) * phase_i
        psi_rad[idx, :MAX_NUM] = np.sqrt(np.maximum(rad_profile, 0.0)) * phase_r

    psi_isl = psi_isl / np.maximum(np.linalg.norm(psi_isl, axis=1, keepdims=True), 1e-12)
    psi_rad = psi_rad / np.maximum(np.linalg.norm(psi_rad, axis=1, keepdims=True), 1e-12)

    u_mix = u_ent @ u_qft
    scrambled_isl = np.einsum("ij,nj->ni", u_mix, psi_isl)
    scrambled_rad = np.einsum("ij,nj->ni", u_mix, psi_rad)

    joint_amp = np.sqrt(0.5) * (scrambled_isl + np.exp(1j * scramble_strength * np.pi) * scrambled_rad)
    joint_amp = joint_amp / np.maximum(np.linalg.norm(joint_amp, axis=1, keepdims=True), 1e-12)

    p_rad_modes = np.abs(scrambled_rad[:, :MAX_NUM]) ** 2
    p_isl_modes = np.abs(scrambled_isl[:, :MAX_NUM]) ** 2
    p_joint_modes = np.abs(joint_amp[:, :MAX_NUM]) ** 2

    p_rad_modes = p_rad_modes / np.maximum(np.sum(p_rad_modes, axis=1, keepdims=True), 1e-12)
    p_isl_modes = p_isl_modes / np.maximum(np.sum(p_isl_modes, axis=1, keepdims=True), 1e-12)
    p_joint_modes = p_joint_modes / np.maximum(np.sum(p_joint_modes, axis=1, keepdims=True), 1e-12)

    s_rad_local = -p_rad_modes * np.log(np.maximum(p_rad_modes, 1e-12))
    s_isl_local = -p_isl_modes * np.log(np.maximum(p_isl_modes, 1e-12))
    s_joint_local = -p_joint_modes * np.log(np.maximum(p_joint_modes, 1e-12))

    mutual_info_wedge = np.maximum(s_rad_local + s_isl_local - s_joint_local, 0.0)
    boundary_area_term = inv_four_g * area_tension[None, :] * (1.0 - 0.5 * rad_occ_batch)
    island_recovery_term = -1.5 * isl_occ_batch * (1.0 - 0.5 * isl_occ_batch)

    s_gen = boundary_area_term + island_recovery_term - 2.5 * mutual_info_wedge
    wedge_weights = np.exp(-s_gen - np.max(-s_gen, axis=1, keepdims=True))
    wedge_weights = wedge_weights * (0.45 + 0.55 * p_joint_modes)
    wedge_probs = wedge_weights / np.maximum(np.sum(wedge_weights, axis=1, keepdims=True), 1e-12)

    qc_verify = QuantumCircuit(NUM_QUBITS)
    qc_verify.append(QFTGate(NUM_QUBITS), list(range(NUM_QUBITS)))
    for q in range(NUM_QUBITS - 1):
        qc_verify.cx(q, q + 1)
    qc_verify.cx(NUM_QUBITS - 1, 0)
    _ = Statevector(joint_amp.__getitem__(n_samples)).evolve(qc_verify)

    y_targets = draws[TOTAL_HORIZON:].copy()
    return wedge_probs[:n_samples], y_targets, wedge_probs.__getitem__(n_samples)


def evaluate_qes_nll(
    probs_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    prob_mat = 0.86 * probs_eval + 0.14 * (1.0 / float(MAX_NUM))
    prob_mat = prob_mat / np.sum(prob_mat, axis=1, keepdims=True)

    n_eval = len(y_eval)
    target_mask = np.zeros((n_eval, MAX_NUM), dtype=bool)
    row_idx = np.arange(n_eval)[:, None]
    target_mask[row_idx, y_eval - 1] = True

    log_p = np.log(np.maximum(prob_mat, 1e-12))
    nll_per_draw = -np.sum(np.where(target_mask, log_p, 0.0), axis=1)
    return float(np.mean(nll_per_draw))


def decode_entanglement_wedge_support(wedge_prob_vec: np.ndarray) -> list[int]:
    top_indices = np.argsort(wedge_prob_vec)[::-1][:NUM_POSITIONS]
    combo = sorted([int(idx) + 1 for idx in top_indices])
    if len(set(combo)) != NUM_POSITIONS:
        raise RuntimeError("Neuspesna rekonstrukcija 7 modova iz QES klina isprepletenosti.")
    return combo


def main() -> None:
    print("Metoda: Netta Engelhardt Quantum Extremal Surface (QES) & Entanglement Island Regressor (1x6 qubita)")
    print("Posebnost: Generalisana entropija S_gen = Area/4G + S_bulk - I(R:I) na Qiskit simulatoru")

    u_qft, u_ent = build_hawking_scrambling_unitaries()

    for file_path in FILES:
        if not file_path.exists():
            print(f"Fajl nije pronadjen: {file_path}")
            continue

        draws = load_draws(file_path)
        n_total = len(draws)
        n_samples = n_total - TOTAL_HORIZON

        train_end = TOTAL_HORIZON + int(0.75 * n_samples)
        val_end = TOTAL_HORIZON + int(0.88 * n_samples)
        val_slice_start = train_end - TOTAL_HORIZON
        val_slice_end = val_end - TOTAL_HORIZON

        area_tr = fit_horizon_area_tension(draws[:train_end])

        best_inv_4g = float(AREA_NEWTON_G_GRID.__getitem__(0))
        best_scram = float(PAGE_SCRAMBLE_GRID.__getitem__(0))
        best_val_nll = float("inf")

        for inv_4g in AREA_NEWTON_G_GRID:
            for scram in PAGE_SCRAMBLE_GRID:
                probs_all, y_all, _ = compute_qes_entanglement_wedge_scores(
                    draws,
                    area_tension=area_tr,
                    u_qft=u_qft,
                    u_ent=u_ent,
                    inv_four_g=inv_4g,
                    scramble_strength=scram,
                )
                val_nll = evaluate_qes_nll(
                    probs_all[val_slice_start:val_slice_end],
                    y_all[val_slice_start:val_slice_end],
                )
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_inv_4g = inv_4g
                    best_scram = scram

        probs_eval_all, y_eval_all, _ = compute_qes_entanglement_wedge_scores(
            draws,
            area_tension=area_tr,
            u_qft=u_qft,
            u_ent=u_ent,
            inv_four_g=best_inv_4g,
            scramble_strength=best_scram,
        )
        test_nll = evaluate_qes_nll(
            probs_eval_all[val_slice_end:],
            y_eval_all[val_slice_end:],
        )

        area_full = fit_horizon_area_tension(draws)
        _, _, next_wedge_probs = compute_qes_entanglement_wedge_scores(
            draws,
            area_tension=area_full,
            u_qft=u_qft,
            u_ent=u_ent,
            inv_four_g=best_inv_4g,
            scramble_strength=best_scram,
        )
        next_combo = decode_entanglement_wedge_support(next_wedge_probs)

        print(f"CSV: {file_path.name}")
        print(f"Validacija NLL (inv_4G={best_inv_4g}, scramble={best_scram}): {best_val_nll:.4f}")
        print(f"Test NLL: {test_nll:.4f}")
        print(f"NEXT: {next_combo}")


if __name__ == "__main__":
    main()



"""
Metoda: Netta Engelhardt Quantum Extremal Surface (QES) & Entanglement Island Regressor (1x6 qubita)
Posebnost: Generalisana entropija S_gen = Area/4G + S_bulk - I(R:I) na Qiskit simulatoru

CSV: loto7_4718_k90.csv
Validacija NLL (inv_4G=0.35, scramble=0.85): 25.8691
Test NLL: 25.8648
NEXT: [1, 6, 9, 14, 17, 22, 34]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (inv_4G=0.35, scramble=0.85): 25.8538
Test NLL: 25.8875
NEXT: [1, 6, 9, 14, 17, 18, 22]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (inv_4G=0.35, scramble=0.85): 25.9346
Test NLL: 25.8332
NEXT: [1, 7, 22, 26, 32, 34, 39]
"""



"""
Netta Engelhardt: The Physicist Helping to Unlock Hawking's Black Hole Paradox

Oblast kvantnih ekstremalnih površi (QES), generalisane entropije \(S_{\text{gen}}\) i rekonstrukcije informacija iz ostrva isprepletenosti (Entanglement Islands / Page curve).

• Osnovna ideja (Netta Engelhardt — QES i Entanglement Islands):
	• Pre Pageovog vremena, entropija Hawkingovog zračenja raste jer su fotoni izvan horizonta isprepleteni sa partnerima unutar crne rupe.
	• Nakon Pageovog vremena, formira se kvantna ekstremalna površ (QES) unutar horizonta koja obuhvata ostrvo \(I\) — skup unutrašnjih modova čija se informacija kroz rekonstrukciju klina isprepletenosti (entanglement wedge reconstruction) direktno preslikava u izlazno zračenje, smanjujući \(S_{\text{gen}}\) i vraćajući izgubljenu informaciju.

	1. Podela sistema na radijaciju \(R\) (poslednje kolo) i unutrašnjost crne rupe \(B\) (dublja istorija kola): Kvantno kolo na 1x6 qubita na Qiskit simulatoru spaja radijacioni registar i unutrašnji registar kroz Hawking–Page ispreplićući operator (UnitaryGate / QFTGate + sprezanje), generišući matricu gustine \(\rho _{R\cup I}\).
	2. Ekstremizacija generalisane entropije \(S_{\text{gen}}(k)\) po kandidatima ostrva: Za svaki od 39 kanala računa se ravnoteža između površinskog člana \(\frac{\text{Area}(\partial I_{k})}{4G_{N}}\) (informaciona barijera prelaza iz istorije) i kvantne von Neumannove entropije isprepletenosti \(S_{\text{vN}}(\rho_{R \cup I_k})\) (uzajamna informacija sa najnovijim zračenjem).
	3. Entanglement Wedge Reconstruction dekoder: NEXT kombinacija od 7 različitih brojeva dobija se direktnim izborom 7 modova sa najvećim informacionim oporavkom (maksimalnom međusobnom informacijom i minimalnom generalisanom entropijom \(S_{\text{gen}}\)) bez ikakvih pozicionih bias matrica.

Koristi kvantnu uzajamnu informaciju i ekstremizaciju von Neumannove entropije između dublje istorije (ostrva \(I\)) i najnovijih izvlačenja (radijacije \(R\)).
"""
