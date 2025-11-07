"""
Mini heat-bath CI solver packaged for re-use inside PySCF CASCI calculations.

Example
-------
>>> from pyscf import gto, scf, mcscf
>>> from shci_solver import HeatBathCISolver
>>> mol = gto.M(atom=\"Be 0 0 0\", basis=\"sto-3g\", spin=0)
>>> mf = scf.RHF(mol).run()
>>> mc = mcscf.CASCI(mf, norb=4, nelec=(2, 2))
>>> mc.fcisolver = HeatBathCISolver(mol, epsilon=1e-3, start_with_singles=True)
>>> mc.kernel()  # doctest: +SKIP

The :class:`HeatBathCISolver` mirrors PySCF's ``FCISolver`` API so CASCI can
query energies and reduced density matrices.  The solver returns an
:class:`HCIVector` describing the selected determinant space, which its
``make_rdm1``/``make_rdm12`` helpers know how to interpret.
"""

from .solver import HCIVector, HeatBathCISolver

__all__ = ["HCIVector", "HeatBathCISolver"]
