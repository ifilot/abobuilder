Using ABO Builder
=================

ABO Builder is typically used alongside PyQInt to compute electronic
structure data and then serialize it to ``.abo`` files for Managlyph. The
examples below mirror the workflow in the project examples.

Prerequisites
-------------

Install both packages (e.g. via pip) so that the PyQInt wavefunction routines
and ABO Builder writer are available in the same environment.

Builder configuration
---------------------

``AboBuilder`` is the single orchestrating class in the package. It encapsulates
the file-writing logic for both legacy ``.abo`` (v0) and modern ``.abof`` (v1)
formats, and exposes convenience methods for turning electronic-structure data
into Managlyph-ready orbital meshes. Each builder instance keeps the orbital
color palette and element table needed for serialization, so you can reuse it
across multiple outputs in the same workflow.

You can customize the builder at construction time. At the moment, the main
configuration points are the colors used for occupied and unoccupied orbital
lobes. Supply two colors for each (positive and negative lobes). Colors can be
RGB or RGBA sequences; when RGB is provided, the builder uses its default alpha.

.. code-block:: python

   import numpy as np
   from abobuilder import AboBuilder

   occupied_colors = [
       np.array([0.2, 0.7, 0.9]),  # positive lobe
       np.array([0.1, 0.4, 0.6]),  # negative lobe
   ]
   unoccupied_colors = [
       np.array([0.9, 0.4, 0.2]),
       np.array([0.7, 0.2, 0.1]),
   ]

   builder = AboBuilder(
       occupied_colors=occupied_colors,
       unoccupied_colors=unoccupied_colors,
   )

   # Later, reuse the configured builder for multiple outputs.
   builder.build_abo_hf_v1(...)

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
           nocc=res['nelec'] // 2,
           nsamples=51,
           compress=True,
           geometry_descriptor=desc,
       )

Localized orbitals (Foster-Boys)
--------------------------------

After computing canonical orbitals, you can localize them using PyQInt's
Foster-Boys routine and emit a second ``.abo`` file.

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
       Foster-Boys localized molecular orbitals for CH4 (STO-3G).
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
           nocc=res['nelec'] // 2,
           nsamples=51,
           compress=True,
           geometry_descriptor=desc,
       )

Tips
----

* Use ``compress=True`` to enable Zstandard compression for v1 files.
* Pass ``nocc`` to control the occupied-orbital cutoff. Use ``None`` to infer
  from electron count or ``"all"`` to mark all orbitals as occupied.
* Supply ``occupied_colors`` and ``unoccupied_colors`` when constructing
  ``AboBuilder`` to customize orbital colors.
* For legacy consumers, ``build_abo_hf_v0`` can be used to emit v0 files.

VASP NEB trajectories
---------------------

You can also build geometry-only reaction trajectories from a VASP NEB folder
layout (e.g. ``00`` .. ``09`` image directories). The writer stores only the
last image from each directory, and always reads endpoints from ``POSCAR``.

.. code-block:: python

   from abobuilder import AboBuilder

   builder = AboBuilder()

   # Legacy .abo file (atoms only, no mesh objects)
   builder.build_abo_neb_vasp(
       'neb_path.abo',
       'my_neb_run',
       lattice_atom_count=24,  # use -N to expand all except the last N atoms
       expand_xy=(3, 3),
   )

   # ABOF v1 variant with the reaction-event header bit enabled.
   builder.build_abof_neb_vasp_v1('neb_path.abof', 'my_neb_run')

All NEB frames are centered around the unit-cell midpoint ``(0.5, 0.5, 0.5)``
in direct coordinates before optional lattice expansion.

``expand_xy`` values must be positive odd integers so the expansion can be
centered around zero (for example ``(3, 3)`` yields offsets ``-1, 0, 1`` in
both X and Y).
