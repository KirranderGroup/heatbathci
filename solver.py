"""
Reusable heat-bath CI implementation that can serve as a PySCF ``FCISolver``.

The code is reorganised from the standalone ``toy_shci`` scripts so the solver
can be plugged into ``mc.fcisolver`` when running CASCI calculations.  The
algorithm remains intentionally simple and is still best suited for educational
and small active-space problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from pyscf import ao2mo, lib
from pyscf.fci import cistring, direct_spin1, spin_op


@dataclass
class HCIVector:
    """Compact container that stores the selected determinant space."""

    determinants: Tuple[int, ...]
    coeffs: np.ndarray
    norb: int
    nelec: Tuple[int, int]

    def __post_init__(self) -> None:
        self.coeffs = np.asarray(self.coeffs, dtype=np.complex128)
        norm = np.linalg.norm(self.coeffs)
        if norm == 0:
            raise ValueError("Coefficient vector must be non-zero.")
        self.coeffs /= norm

    @property
    def nso(self) -> int:
        return 2 * self.norb


class HeatBathCISolver(direct_spin1.FCISolver):
    """
    Minimal SHCI-style CI solver that satisfies PySCF's ``FCISolver`` API.

    Parameters
    ----------
    mol : pyscf.gto.Mole, optional
        Molecule used only for logging context.
    epsilon : float, optional
        Screening threshold ``|H_ai c_i| >= epsilon``.
    start_with_singles : bool, optional
        If true, seed the variational space with all single excitations of the
        Hartree-Fock determinant before applying the heat-bath selection.
    max_macro : int, optional
        Cap on macro-iterations; ``None`` keeps iterating until no new
        determinants pass the threshold.
    verbose : int, optional
        PySCF logger verbosity.
    """

    def __init__(
        self,
        mol=None,
        *,
        epsilon: float = 5e-4,
        start_with_singles: bool = False,
        max_macro: int | None = None,
        verbose: int = lib.logger.NOTE,
    ) -> None:
        super().__init__(mol)
        self.epsilon = epsilon
        self.start_with_singles = start_with_singles
        self.max_macro = max_macro
        self.verbose = verbose
        self._logger = lib.logger.new_logger(mol, verbose)
        self.nroots = 1
        self._last_vector: HCIVector | None = None

    def kernel(
        self,
        h1: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int] | int,
        ci0: HCIVector | None = None,
        **kwargs,
    ) -> Tuple[float, HCIVector]:
        """Run the heat-bath CI selection for the provided active space."""
        nalpha, nbeta = self._parse_nelec(nelec)
        h1 = np.asarray(h1, dtype=float)
        eri = np.asarray(eri, dtype=float)
        if eri.ndim == 2:
            eri = ao2mo.restore(1, eri, norb)
        elif eri.ndim == 1:
            eri = ao2mo.restore(1, eri, norb)
        engine = _HeatBathCIEngine(
            h1=h1,
            eri=eri,
            norb=norb,
            nalpha=nalpha,
            nbeta=nbeta,
            epsilon=self.epsilon,
            start_with_singles=self.start_with_singles,
            max_macro=self.max_macro,
            logger=self._logger,
        )
        energy, vector = engine.run(ci0=ci0)
        self._last_vector = vector
        self.converged = True
        return energy, vector

    # ------------------------------------------------------------------
    # RDM interface expected by PySCF's CASCI driver
    # ------------------------------------------------------------------
    def make_rdm1(
        self, ci: HCIVector, norb: int, nelec: Tuple[int, int] | int
    ) -> np.ndarray:
        vector = self._ensure_vector(ci, norb, nelec)
        dm1_spin = compute_spin_orbital_1rdm(vector.determinants, vector.coeffs, vector.nso)
        return spinfree_1rdm(dm1_spin, vector.norb)

    def make_rdm12(
        self, ci: HCIVector, norb: int, nelec: Tuple[int, int] | int
    ) -> Tuple[np.ndarray, np.ndarray]:
        vector = self._ensure_vector(ci, norb, nelec)
        dm1_spin = compute_spin_orbital_1rdm(vector.determinants, vector.coeffs, vector.nso)
        dm2_spin = compute_spin_orbital_2rdm(vector.determinants, vector.coeffs, vector.nso)
        dm1 = spinfree_1rdm(dm1_spin, vector.norb)
        dm2 = spinfree_2rdm(dm2_spin, vector.norb)
        return dm1, dm2

    def spin_square(self, ci: HCIVector, norb: int, nelec: Tuple[int, int] | int):
        """Return <S^2> by embedding the HCIVector into PySCF's full CI basis."""
        vector = self._ensure_vector(ci, norb, nelec)
        nalpha, nbeta = self._parse_nelec(nelec)
        na = cistring.num_strings(norb, nalpha)
        nb = cistring.num_strings(norb, nbeta)
        fcivec = np.zeros((na, nb), dtype=np.complex128)
        mask = (1 << norb) - 1
        for det, coeff in zip(vector.determinants, vector.coeffs):
            alpha_bits = det & mask
            beta_bits = (det >> norb) & mask
            ia = cistring.str2addr(norb, nalpha, alpha_bits)
            ib = cistring.str2addr(norb, nbeta, beta_bits)
            fcivec[ia, ib] = coeff
        fcivec_use = np.asarray(fcivec.real if np.iscomplexobj(fcivec) else fcivec, order="C")
        value, mult = spin_op.spin_square0(fcivec_use, norb, (nalpha, nbeta))
        return float(value.real), mult

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _ensure_vector(
        self, ci: HCIVector, norb: int, nelec: Tuple[int, int] | int
    ) -> HCIVector:
        if isinstance(ci, HCIVector):
            return ci
        raise TypeError(
            "HeatBathCISolver expects HCIVector objects for CI input/output; "
            "did you accidentally provide a plain numpy array?"
        )

    def _parse_nelec(self, nelec) -> Tuple[int, int]:
        """
        Handle PySCF's ``nelec`` argument which may be an int or a (na, nb) tuple.

        When only the total electron count is provided we use ``self.spin`` to
        recover ``nalpha``/``nbeta``; for singlet solvers ``self.spin`` defaults
        to zero so the electrons are split evenly.
        """
        if isinstance(nelec, (tuple, list)):
            nalpha, nbeta = map(int, nelec)
        else:
            tot = int(nelec)
            spin = getattr(self, "spin", 0)
            nalpha = (tot + spin) // 2
            nbeta = tot - nalpha
        if nalpha < 0 or nbeta < 0:
            raise ValueError(f"Invalid electron numbers (nalpha={nalpha}, nbeta={nbeta}).")
        return nalpha, nbeta


# -----------------------------------------------------------------------------
# Internal engine that mirrors the structure of toy_shci.py but is reusable.
# -----------------------------------------------------------------------------
class _HeatBathCIEngine:
    def __init__(
        self,
        *,
        h1: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nalpha: int,
        nbeta: int,
        epsilon: float,
        start_with_singles: bool,
        max_macro: int | None,
        logger: lib.logger.Logger,
    ) -> None:
        self.h1_mo = np.asarray(h1, dtype=float)
        self.eri_mo = np.asarray(eri, dtype=float)
        self.norb = norb
        self.nalpha = nalpha
        self.nbeta = nbeta
        self.nso = 2 * norb
        self.epsilon = epsilon
        self.start_with_singles = start_with_singles
        self.max_macro = max_macro
        self.logger = logger

        self.h1_so = np.zeros((self.nso, self.nso))
        self.h1_so[:self.norb, :self.norb] = self.h1_mo
        self.h1_so[self.norb :, self.norb :] = self.h1_mo

        self.det_space = tuple(self._enumerate_determinants())

    # ---------------------------- selection loop -----------------------------
    def run(self, ci0: HCIVector | None = None) -> Tuple[float, HCIVector]:
        if ci0 is not None:
            if not isinstance(ci0, HCIVector):
                raise TypeError("Initial guess must be an HCIVector.")
            V = list(ci0.determinants)
            coeffs = np.array(ci0.coeffs, copy=True)
            energy = self._expectation(V, coeffs)
        else:
            V = [self._reference_det()]
            coeffs = np.array([1.0], dtype=np.complex128)
            energy = self.diagonal_energy(V[0])

        if self.start_with_singles:
            singles = [d for d in self._all_singles(V[0]) if d not in V]
            if singles:
                V.extend(singles)
                coeffs = np.ones(len(V), dtype=np.complex128)
                coeffs[0] = 1.0
                coeffs[1:] = 1.0 / len(singles)
                energy, coeffs = self._diagonalize_subspace(V)

        macro = 1
        while True:
            if self.max_macro is not None and macro > self.max_macro:
                self.logger.info("Reached max macro-iterations (%d).", self.max_macro)
                break
            candidates = [det for det in self.det_space if det not in V]
            if not candidates:
                self.logger.info("Full determinant space reached.")
                break
            report, picked = self.hci_screen(candidates, V, coeffs)
            self.logger.info(
                "Macro %d: %d candidate(s) above epsilon=%.2e",
                macro,
                len(picked),
                self.epsilon,
            )
            if not picked:
                break
            V.extend(picked)
            energy, coeffs = self._diagonalize_subspace(V)
            macro += 1

        vector = HCIVector(tuple(V), coeffs, self.norb, (self.nalpha, self.nbeta))
        self.logger.note(
            "Selected %d determinants; final HCI energy = %.10f Eh",
            len(V),
            energy,
        )
        return energy, vector

    def _diagonalize_subspace(self, V: Sequence[int]) -> Tuple[float, np.ndarray]:
        H = self.build_hamiltonian(V)
        eigvals, eigvecs = np.linalg.eigh(H)
        idx = np.argmin(eigvals.real)
        energy = float(eigvals[idx].real)
        coeffs = eigvecs[:, idx]
        return energy, coeffs

    # ---------------------------- determinant utils -------------------------
    def so_index(self, spatial: int, spin: int) -> int:
        return spatial if spin == 0 else spatial + self.norb

    def so_to_spatial_spin(self, idx: int) -> Tuple[int, int]:
        return (idx, 0) if idx < self.norb else (idx - self.norb, 1)

    def _enumerate_determinants(self) -> Iterable[int]:
        for alpha_occ in combinations(range(self.norb), self.nalpha):
            alpha_bits = 0
            for p in alpha_occ:
                alpha_bits |= 1 << self.so_index(p, 0)
            for beta_occ in combinations(range(self.norb), self.nbeta):
                bits = alpha_bits
                for q in beta_occ:
                    bits |= 1 << self.so_index(q, 1)
                yield bits

    def _reference_det(self) -> int:
        bits = 0
        for i in range(self.nalpha):
            bits |= 1 << self.so_index(i, 0)
        for i in range(self.nbeta):
            bits |= 1 << self.so_index(i, 1)
        return bits

    def occ_list(self, bits: int) -> List[int]:
        return [p for p in range(self.nso) if (bits >> p) & 1]

    def excitation_from_to(self, i_bits: int, a_bits: int):
        diff = i_bits ^ a_bits
        if diff == 0:
            return 0, [], []
        removed = [p for p in range(self.nso) if (i_bits >> p) & 1 and not (a_bits >> p) & 1]
        added = [p for p in range(self.nso) if (a_bits >> p) & 1 and not (i_bits >> p) & 1]
        degree = len(removed)
        if degree in (1, 2):
            return degree, sorted(removed), sorted(added)
        return 3, removed, added

    def fermionic_sign(self, i_bits: int, removed: Sequence[int], added: Sequence[int]) -> int:
        sign = 1
        cur = i_bits
        for r in sorted(removed, reverse=True):
            below = cur & ((1 << r) - 1)
            if bin(below).count("1") % 2:
                sign = -sign
            cur &= ~(1 << r)
        for p in sorted(added):
            below = cur & ((1 << p) - 1)
            if bin(below).count("1") % 2:
                sign = -sign
            cur |= 1 << p
        return sign

    def eri_as_so(self, p: int, q: int, r: int, s: int) -> float:
        ip, sp = self.so_to_spatial_spin(p)
        iq, sq = self.so_to_spatial_spin(q)
        ir, sr = self.so_to_spatial_spin(r)
        is_, ss = self.so_to_spatial_spin(s)
        coul = 0.0
        exch = 0.0
        if sp == sr and sq == ss:
            coul = self.eri_mo[ip, ir, iq, is_]
        if sp == ss and sq == sr:
            exch = self.eri_mo[ip, is_, iq, ir]
        return coul - exch

    def diagonal_energy(self, bits: int) -> float:
        occ = self.occ_list(bits)
        e1 = sum(self.h1_so[p, p] for p in occ)
        e2 = 0.0
        for a, p in enumerate(occ):
            for q in occ[:a]:
                e2 += self.eri_as_so(p, q, p, q)
        return e1 + e2

    def H_element(self, i_bits: int, j_bits: int) -> float:
        kind, removed, added = self.excitation_from_to(i_bits, j_bits)
        if kind == 3:
            return 0.0
        if kind == 0:
            return self.diagonal_energy(i_bits)
        phase = self.fermionic_sign(i_bits, removed, added)
        if kind == 1:
            r = removed[0]
            p = added[0]
            h = self.h1_so[p, r]
            for k in self.occ_list(i_bits):
                if k == r or not (j_bits >> k) & 1:
                    continue
                h += self.eri_as_so(p, k, r, k)
            return phase * h
        r, s = removed
        p, q = added
        return phase * self.eri_as_so(p, q, r, s)

    def hci_screen(
        self, candidates: Sequence[int], V: Sequence[int], coeffs: np.ndarray
    ) -> Tuple[List[Tuple[int, float]], List[int]]:
        report = []
        for det in candidates:
            wmax = 0.0
            for ref, c in zip(V, coeffs):
                wmax = max(wmax, abs(self.H_element(det, ref) * c))
            report.append((det, wmax))
        selected = [det for det, weight in report if weight >= self.epsilon]
        return report, selected

    def build_hamiltonian(self, V: Sequence[int]) -> np.ndarray:
        dim = len(V)
        H = np.zeros((dim, dim))
        for i, Di in enumerate(V):
            for j in range(i, dim):
                Dj = V[j]
                val = self.H_element(Di, Dj)
                H[i, j] = H[j, i] = val
        return H

    def _all_singles(self, det: int) -> List[int]:
        occ = self.occ_list(det)
        virt = [p for p in range(self.nso) if not (det >> p) & 1]
        singles = []
        for r in occ:
            for p in virt:
                if self.so_to_spatial_spin(p)[1] != self.so_to_spatial_spin(r)[1]:
                    continue
                if (det >> p) & 1:
                    continue
                new_det = (det & ~(1 << r)) | (1 << p)
                singles.append(new_det)
        return singles

    def _expectation(self, V: Sequence[int], coeffs: np.ndarray) -> float:
        H = self.build_hamiltonian(V)
        return float(np.real(np.conjugate(coeffs) @ (H @ coeffs)))


# -----------------------------------------------------------------------------
# RDM utilities (adapted from toy_shci.py)
# -----------------------------------------------------------------------------
def popcount(value: int) -> int:
    return bin(value).count("1")


def apply_annihilation(bitstring: int, orb: int) -> Tuple[float, int | None]:
    if not (bitstring >> orb) & 1:
        return 0.0, None
    below = popcount(bitstring & ((1 << orb) - 1))
    phase = -1.0 if below % 2 else 1.0
    new_bits = bitstring & ~(1 << orb)
    return phase, new_bits


def apply_creation(bitstring: int, orb: int) -> Tuple[float, int | None]:
    if (bitstring >> orb) & 1:
        return 0.0, None
    below = popcount(bitstring & ((1 << orb) - 1))
    phase = -1.0 if below % 2 else 1.0
    new_bits = bitstring | (1 << orb)
    return phase, new_bits


def apply_operator_sequence(bitstring: int, ops: Sequence[Tuple[str, int]]):
    phase = 1.0
    state = bitstring
    for op, orb in ops:
        if op == "ann":
            contrib, new_state = apply_annihilation(state, orb)
        else:
            contrib, new_state = apply_creation(state, orb)
        if new_state is None:
            return 0.0, None
        phase *= contrib
        state = new_state
    return phase, state


def compute_spin_orbital_1rdm(
    dets: Sequence[int], coeffs: np.ndarray, nso: int
) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    det_index = {det: idx for idx, det in enumerate(dets)}
    dm1 = np.zeros((nso, nso), dtype=np.complex128)
    for p in range(nso):
        for q in range(nso):
            ops = (("ann", q), ("cre", p))
            acc = 0.0j
            for idx_j, det_j in enumerate(dets):
                phase, transformed = apply_operator_sequence(det_j, ops)
                if transformed is None:
                    continue
                idx_k = det_index.get(transformed)
                if idx_k is None:
                    continue
                acc += np.conjugate(coeffs[idx_k]) * (phase * coeffs[idx_j])
            dm1[p, q] = acc
    return dm1


def compute_spin_orbital_2rdm(
    dets: Sequence[int], coeffs: np.ndarray, nso: int
) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    det_index = {det: idx for idx, det in enumerate(dets)}
    gamma = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
    for p in range(nso):
        for q in range(nso):
            for r in range(nso):
                for s in range(nso):
                    ops = (("ann", r), ("ann", s), ("cre", q), ("cre", p))
                    acc = 0.0j
                    for idx_j, det_j in enumerate(dets):
                        phase, transformed = apply_operator_sequence(det_j, ops)
                        if transformed is None:
                            continue
                        idx_k = det_index.get(transformed)
                        if idx_k is None:
                            continue
                        acc += np.conjugate(coeffs[idx_k]) * (phase * coeffs[idx_j])
                    gamma[p, q, r, s] = acc
    return gamma


def spinfree_1rdm(dm1_spin: np.ndarray, norb: int) -> np.ndarray:
    dm1_spin = np.asarray(dm1_spin, dtype=np.complex128)
    dm1_sf = np.zeros((norb, norb), dtype=float)
    for p in range(norb):
        for q in range(norb):
            alpha = dm1_spin[p, q]
            beta = dm1_spin[p + norb, q + norb]
            dm1_sf[p, q] = (alpha + beta).real
    return dm1_sf


def spinfree_2rdm(gamma_spin: np.ndarray, norb: int) -> np.ndarray:
    gamma_spin = np.asarray(gamma_spin, dtype=np.complex128)
    gamma_sf = np.zeros((norb, norb, norb, norb), dtype=float)
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    val = 0.0
                    for sp in (0, 1):
                        p_idx = p + sp * norb
                        r_idx = r + sp * norb
                        for sq in (0, 1):
                            q_idx = q + sq * norb
                            s_idx = s + sq * norb
                            val += gamma_spin[p_idx, q_idx, r_idx, s_idx].real
                    gamma_sf[p, q, r, s] = val
    return np.swapaxes(gamma_sf, 1, 2)
