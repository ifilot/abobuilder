import sys
import os

from abobuilder import AboBuilder
from pyqint import MoleculeBuilder, HF, FosterBoys

def main():
    # perform Hartree-Fock calculation of cubane
    cubane = MoleculeBuilder().from_name('cubane')
    res = HF(cubane, basis='sto3g').rhf(verbose=True)
    
    # CH4nstruct .abo file for the canonical orbitals of CH4
    if not os.path.exists('cubane.abo'):
        AboBuilder().build_abo_hf_v1('cubane.abo', 
                                res['nuclei'], 
                                res['cgfs'], 
                                res['orbc'], 
                                res['orbe'],
                                nsamples=71,
                                compress=True,
                                sz=7,
                                maxmo=28)

    # perform Foster-Boys localization
    res_fb = FosterBoys(res).run()

    # Construct .abo file for the localized orbitals of cubane
    if not os.path.exists('cubane_fb.abo'):
        AboBuilder().build_abo_hf_v1('cubane_fb.abo', 
                                res['nuclei'], 
                                res_fb['cgfs'], 
                                res_fb['orbc'], 
                                res_fb['orbe'],
                                nsamples=71,
                                compress=True,
                                sz=7,
                                maxmo=28)

if __name__ == '__main__':
    main()
