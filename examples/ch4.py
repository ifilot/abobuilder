import sys
import os

from abobuilder import AboBuilder
from pyqint import MoleculeBuilder, HF, FosterBoys

def main():
    # perform Hartree-Fock calculation of CH4
    ch4 = MoleculeBuilder().from_name('CH4')
    res = HF(ch4, basis='sto3g').rhf(verbose=True)
    
    # CH4nstruct .abo file for the canonical orbitals of CH4
    if not os.path.exists('ch4.abo'):
        AboBuilder().build_abo_hf_v1('ch4.abo', 
                                res['nuclei'], 
                                res['cgfs'], 
                                res['orbc'], 
                                res['orbe'],
                                nsamples=51,
                                compress=True)

    # perform Foster-Boys localization
    res_fb = FosterBoys(res).run()

    # Construct .abo file for the localized orbitals of CH4
    if not os.path.exists('ch4_fb.abo'):
        AboBuilder().build_abo_hf_v1('ch4_fb.abo', 
                                res['nuclei'], 
                                res_fb['cgfs'], 
                                res_fb['orbc'], 
                                res_fb['orbe'],
                                nsamples=51,
                                compress=True)

if __name__ == '__main__':
    main()
