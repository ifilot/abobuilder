import os
import tempfile
import unittest

import numpy as np
from abobuilder import AboBuilder


class TestNebBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from ase import Atoms
            from ase.io import write
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"ASE is required for NEB tests: {exc}")
        cls.Atoms = Atoms
        cls.write = staticmethod(write)

    def _write_neb_image(self, root, name, poscar_atoms, contcar_atoms=None):
        folder = os.path.join(root, name)
        os.makedirs(folder)
        self.write(os.path.join(folder, 'POSCAR'), poscar_atoms, format='vasp')
        if contcar_atoms is not None:
            self.write(os.path.join(folder, 'CONTCAR'), contcar_atoms, format='vasp')

    def _write_vasp4_poscar(self, path, cell, counts, positions_direct):
        with open(path, 'w', encoding='utf8') as handle:
            handle.write('VASP4 test\n')
            handle.write('1.0\n')
            for vec in cell:
                handle.write(f"{vec[0]} {vec[1]} {vec[2]}\n")
            handle.write(' '.join(str(c) for c in counts) + '\n')
            handle.write('Direct\n')
            for pos in positions_direct:
                handle.write(f"{pos[0]} {pos[1]} {pos[2]}\n")

    def _read_v0_frames(self, path):
        frames = []
        with open(path, 'rb') as handle:
            nframes = int.from_bytes(handle.read(2), byteorder='little')
            for _ in range(nframes):
                _frame_idx = int.from_bytes(handle.read(2), byteorder='little')
                dlen = int.from_bytes(handle.read(2), byteorder='little')
                _desc = handle.read(dlen).decode('utf8')
                natoms = int.from_bytes(handle.read(2), byteorder='little')

                atoms = []
                for _ in range(natoms):
                    z = int.from_bytes(handle.read(1), byteorder='little')
                    xyz = np.frombuffer(handle.read(12), dtype=np.float32)
                    atoms.append((z, xyz))

                nmodels = int.from_bytes(handle.read(2), byteorder='little')
                self.assertEqual(nmodels, 0)
                frames.append(atoms)
        return frames

    def test_build_abo_neb_vasp_uses_endpoint_poscar_and_last_intermediate(self):
        builder = AboBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            cell = np.diag([5.0, 5.0, 5.0])
            start = self.Atoms('H2', positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=cell, pbc=True)
            mid_poscar = self.Atoms('H2', positions=[[1, 0, 0], [1.5, 0.5, 0.5]], cell=cell, pbc=True)
            mid_contcar = self.Atoms('H2', positions=[[2, 0, 0], [2.5, 0.5, 0.5]], cell=cell, pbc=True)
            end = self.Atoms('H2', positions=[[3, 0, 0], [3.5, 0.5, 0.5]], cell=cell, pbc=True)

            self._write_neb_image(tmpdir, '00', start)
            self._write_neb_image(tmpdir, '01', mid_poscar, contcar_atoms=mid_contcar)
            self._write_neb_image(tmpdir, '09', end)

            outfile = os.path.join(tmpdir, 'neb.abo')
            builder.build_abo_neb_vasp(outfile, tmpdir, legacy_mode=True)
            frames = self._read_v0_frames(outfile)

            self.assertEqual(len(frames), 3)
            np.testing.assert_allclose(frames[0][0][1], [-2.5, -2.5, -2.5], atol=1e-6)
            np.testing.assert_allclose(frames[1][0][1], [-0.5, -2.5, -2.5], atol=1e-6)
            np.testing.assert_allclose(frames[2][0][1], [0.5, -2.5, -2.5], atol=1e-6)

    def test_build_abo_neb_vasp_lattice_expansion(self):
        builder = AboBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            cell = np.diag([4.0, 4.0, 4.0])
            image = self.Atoms('H2', positions=[[0, 0, 0], [1.0, 0.0, 0.0]], cell=cell, pbc=True)
            self._write_neb_image(tmpdir, '00', image)
            self._write_neb_image(tmpdir, '09', image)

            outfile = os.path.join(tmpdir, 'neb_expand.abo')
            builder.build_abo_neb_vasp(
                outfile,
                tmpdir,
                lattice_atom_count=1,
                expand_xy=(3, 3),
                legacy_mode=True,
            )
            frames = self._read_v0_frames(outfile)

            self.assertEqual(len(frames[0]), 10)

    def test_build_abo_neb_vasp_negative_lattice_atom_count(self):
        builder = AboBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            cell = np.diag([4.0, 4.0, 4.0])
            image = self.Atoms('H3', positions=[[0, 0, 0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=cell, pbc=True)
            self._write_neb_image(tmpdir, '00', image)
            self._write_neb_image(tmpdir, '09', image)

            outfile = os.path.join(tmpdir, 'neb_negative_expand.abo')
            builder.build_abo_neb_vasp(
                outfile,
                tmpdir,
                lattice_atom_count=-1,
                expand_xy=(3, 1),
                legacy_mode=True,
            )
            frames = self._read_v0_frames(outfile)

            # Expand all except the last atom: 2 expanded across 3x1 + 1 unchanged.
            self.assertEqual(len(frames[0]), 7)

    def test_build_abo_neb_vasp_rejects_non_odd_or_negative_expansion(self):
        builder = AboBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            cell = np.diag([4.0, 4.0, 4.0])
            image = self.Atoms('H', positions=[[0, 0, 0]], cell=cell, pbc=True)
            self._write_neb_image(tmpdir, '00', image)
            self._write_neb_image(tmpdir, '09', image)

            outfile = os.path.join(tmpdir, 'neb_invalid_expand.abo')
            with self.assertRaises(ValueError):
                builder.build_abo_neb_vasp(
                    outfile,
                    tmpdir,
                    lattice_atom_count=1,
                    expand_xy=(2, 3),
                    legacy_mode=True,
                )
            with self.assertRaises(ValueError):
                builder.build_abo_neb_vasp(
                    outfile,
                    tmpdir,
                    lattice_atom_count=1,
                    expand_xy=(-3, 3),
                    legacy_mode=True,
                )

    def test_build_abo_neb_vasp_centers_positions_to_cell_midpoint(self):
        builder = AboBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            cell = np.diag([4.0, 4.0, 4.0])
            image = self.Atoms('H', positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
            self._write_neb_image(tmpdir, '00', image)
            self._write_neb_image(tmpdir, '09', image)

            outfile = os.path.join(tmpdir, 'neb_centered.abo')
            builder.build_abo_neb_vasp(outfile, tmpdir, lattice_atom_count=0, legacy_mode=True)
            frames = self._read_v0_frames(outfile)

            np.testing.assert_allclose(frames[0][0][1], [-2.0, -2.0, -2.0], atol=1e-6)

    def test_build_abo_neb_vasp_vasp4_poscar_uses_outcar_symbols(self):
        builder = AboBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, '00'))
            os.makedirs(os.path.join(tmpdir, '09'))

            cell = np.diag([5.0, 5.0, 5.0])
            self._write_vasp4_poscar(
                os.path.join(tmpdir, '00', 'POSCAR'),
                cell,
                counts=[1, 1],
                positions_direct=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            )
            self._write_vasp4_poscar(
                os.path.join(tmpdir, '09', 'POSCAR'),
                cell,
                counts=[1, 1],
                positions_direct=[[0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
            )

            os.makedirs(os.path.join(tmpdir, '01'))
            middle = self.Atoms('HO', positions=[[0.15, 0.0, 0.0], [0.25, 0.0, 0.0]], cell=cell, pbc=True)
            self.write(os.path.join(tmpdir, '01', 'CONTCAR'), middle, format='vasp')
            with open(os.path.join(tmpdir, '01', 'OUTCAR'), 'w', encoding='utf8') as handle:
                handle.write(' TITEL  = PAW_PBE H 08Apr2002\n')
                handle.write(' TITEL  = PAW_PBE O 08Apr2002\n')

            outfile = os.path.join(tmpdir, 'neb_vasp4.abo')
            builder.build_abo_neb_vasp(outfile, tmpdir, legacy_mode=True)
            frames = self._read_v0_frames(outfile)

            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0][0][0], 1)
            self.assertEqual(frames[0][1][0], 8)

    def test_build_abo_neb_vasp_defaults_to_abof_v1(self):
        builder = AboBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            cell = np.diag([4.0, 4.0, 4.0])
            image = self.Atoms('H', positions=[[0, 0, 0]], cell=cell, pbc=True)
            self._write_neb_image(tmpdir, '00', image)
            self._write_neb_image(tmpdir, '09', image)

            outfile = os.path.join(tmpdir, 'neb_default.abof')
            builder.build_abo_neb_vasp(outfile, tmpdir)

            with open(outfile, 'rb') as handle:
                header = handle.read(8)

            self.assertEqual(header[2:6], b'ABOF')
            self.assertEqual(header[6], 1)

    def test_build_abof_neb_vasp_sets_reaction_flag(self):
        builder = AboBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            cell = np.diag([4.0, 4.0, 4.0])
            image = self.Atoms('H', positions=[[0, 0, 0]], cell=cell, pbc=True)
            self._write_neb_image(tmpdir, '00', image)
            self._write_neb_image(tmpdir, '09', image)

            outfile = os.path.join(tmpdir, 'neb.abof')
            builder.build_abof_neb_vasp_v1(outfile, tmpdir)

            with open(outfile, 'rb') as handle:
                header = handle.read(8)

            self.assertEqual(header[2:6], b'ABOF')
            self.assertEqual(header[6], 1)
            self.assertTrue(header[7] & builder._REACTION_EVENT_FLAG_BIT)


if __name__ == '__main__':
    unittest.main()
