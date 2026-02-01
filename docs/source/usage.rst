Using ABO Builder
=================

ABO Builder is typically used alongside PyQInt to compute electronic
structure data and then serialize it to ``.abo`` files for Managlyph. The
examples below mirror the workflow in the project examples.

Prerequisites
-------------

Install both packages (e.g. via pip) so that the PyQInt wavefunction routines
and ABO Builder writer are available in the same environment.

Canonical orbitals workflow
---------------------------

This example performs a Hartree-Fock calculation for methane and writes a v1
``.abo`` file containing all canonical molecular orbitals.

.. code-block:: python

   import os

   from abobuilder import AboBuilder, clean_multiline
   from pyqint import MoleculeBuilder, HF

   # Build a molecule and run HF.
   ch4 = MoleculeBuilder().from_name('CH4')
   res = HF(ch4, basis='sto3g').rhf(verbose=True)

   # Add a descriptor for the geometry frame.
   desc = clean_multiline(
       """
       Canonical molecular orbitals for CH4 (STO-3G), computed with PyQInt.
       Generated with ABO Builder.
       """
   )

   # Create the ABOF v1 file with compression enabled.
   if not os.path.exists('ch4.abo'):
       AboBuilder().build_abo_hf_v1(
           'ch4.abo',
           res['nuclei'],
           res['cgfs'],
           res['orbc'],
           res['orbe'],
           nsamples=51,
           compress=True,
           geometry_descriptor=desc,
       )

Localized orbitals (Foster–Boys)
--------------------------------

After -omputing canonical orbitals, you can localize them using PyQInt's
Foster–Boys routine and emit a second ``.abo`` file.

.. code-block:: python

   import os

   from abobuilder import AboBuilder, clean_multiline
   from pyqint import MoleculeBuilder, HF, FosterBoys

   ch4 = MoleculeBuilder().from_name('CH4')
   res = HF(ch4, basis='sto3g').rhf(verbose=True)

   # Localize orbitals.
   res_fb = FosterBoys(res).run()

   desc = clean_multiline(
       """
       Foster–Boys localized molecular orbitals for CH4 (STO-3G).
       Generated with ABO Builder.
       """
   )

   if not os.path.exists('ch4_fb.abo'):
       AboBuilder().build_abo_hf_v1(
           'ch4_fb.abo',
           res['nuclei'],
           res_fb['cgfs'],
           res_fb['orbc'],
           res_fb['orbe'],
           nsamples=51,
           compress=True,
           geometry_descriptor=desc,
       )

Tips
----

* Use ``compress=True`` to enable Zstandard compression for v1 files.
* For legacy consumers, ``build_abo_hf_v0`` can be used to emit v0 files.
