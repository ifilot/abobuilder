import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abobuilder import AboBuilder
from pyqint import MoleculeBuilder, HF, FosterBoys

def main():
    # perform Hartree-Fock calculation of CO
    co = MoleculeBuilder().from_name('CO')
    res = HF().rhf(co, basis='sto3g', verbose=True)
    
    # construct .abo file for the canonical orbitals of CO
    if not os.path.exists('co.abo'):
        AboBuilder().build_abo_hf('co.abo', 
                                res['nuclei'], 
                                res['cgfs'], 
                                res['orbc'], 
                                res['orbe'])

    # perform Foster-Boys localization
    res_fb = FosterBoys(res).run()

    # construct .abo file for the localized orbitals of CO
    if not os.path.exists('co_fb.abo'):
        AboBuilder().build_abo_hf('co_fb.abo', 
                                res['nuclei'], 
                                res_fb['cgfs'], 
                                res_fb['orbc'], 
                                res_fb['orbe'])

if __name__ == '__main__':
    main()