import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'abobuilder'))

from abobuilder import AboBuilder
from pyqint import MoleculeBuilder, HF, FosterBoys

def main():
    # perform Hartree-Fock calculation of CH4
    ch4 = MoleculeBuilder().from_name('CH4')
    res = HF().rhf(ch4, basis='sto3g', verbose=True)
    
    # CH4nstruct .abo file for the canonical orbitals of CH4
    if not os.path.exists('ch4.abo'):
        AboBuilder().build_abo_hf('ch4.abo', 
                                res['nuclei'], 
                                res['cgfs'], 
                                res['orbc'], 
                                res['orbe'])

    # perform Foster-Boys localization
    res_fb = FosterBoys(res).run()

    # Construct .abo file for the localized orbitals of CH4
    if not os.path.exists('ch4_fb.abo'):
        AboBuilder().build_abo_hf('ch4_fb.abo', 
                                res['nuclei'], 
                                res_fb['cgfs'], 
                                res_fb['orbc'], 
                                res_fb['orbe'])

if __name__ == '__main__':
    main()