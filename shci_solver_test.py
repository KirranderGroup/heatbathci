from solver import HeatBathCISolver
from pyscf import gto, scf, mcscf

mol = gto.M(atom = 'Be 0 0 0', basis='3-21g', spin=0, charge=0, verbose=5)
mf = scf.RHF(mol)
mf.kernel()
mc = mcscf.CASCI(mf, 9, (2,2))
mc.fcisolver = HeatBathCISolver(mol, epsilon=1e-3, start_with_singles=True)
mc.kernel()