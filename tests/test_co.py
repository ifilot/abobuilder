import unittest
import sys
import os
import numpy as np
import tempfile
import zstandard as zstd
from abobuilder import AboBuilder
from pyqint import MoleculeBuilder, HF, FosterBoys

class TestAboOrbitals(unittest.TestCase):

    def test_unitcell_properties(self):
        # perform Hartree-Fock calculation of CO
        co = MoleculeBuilder().from_name('CO')
        res = HF(co, basis='sto3g').rhf(verbose=True)
        
        # construct .abo file for the canonical orbitals of CO
        if not os.path.exists('co.abo'):
            AboBuilder().build_abo_hf('co.abo', 
                                    res['nuclei'], 
                                    res['cgfs'], 
                                    res['orbc'], 
                                    res['orbe'],
                                    nsamples=51)

        self.assertTrue(os.path.exists('co.abo'))

        # perform Foster-Boys localization
        res_fb = FosterBoys(res).run()

        # construct .abo file for the localized orbitals of CO
        if not os.path.exists('co_fb.abo'):
            AboBuilder().build_abo_hf('co_fb.abo', 
                                    res['nuclei'], 
                                    res_fb['cgfs'], 
                                    res_fb['orbc'], 
                                    res_fb['orbe'],
                                    nsamples=51)
        
        self.assertTrue(os.path.exists('co_fb.abo'))

    def _read_v1_header(self, path):
        with open(path, 'rb') as handle:
            header = handle.read(8)
        self.assertEqual(header[0:2], (0).to_bytes(2, byteorder='little'))
        self.assertEqual(header[2:6], b'ABOF')
        return header[6], header[7]

    def _build_simple_model(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        normals = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2], dtype=np.uint32)
        model = {'vertices': vertices, 'normals': normals, 'indices': indices}
        colors = [np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)]
        return model, colors

    def test_model_v1_header(self):
        builder = AboBuilder()
        model, colors = self._build_simple_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model_v1.abo')
            builder.build_abo_model_v1(path, [model], colors)

            version, flags = self._read_v1_header(path)
            self.assertEqual(version, 1)
            self.assertEqual(flags, 0)

            with open(path, 'rb') as handle:
                handle.seek(8)
                nr_frames = int.from_bytes(handle.read(2), byteorder='little')
            self.assertEqual(nr_frames, 1)

    def test_model_v1_compressed_payload(self):
        builder = AboBuilder()
        model, colors = self._build_simple_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model_v1_compressed.abo')
            builder.build_abo_model_v1(path, [model], colors, compress=True)

            version, flags = self._read_v1_header(path)
            self.assertEqual(version, 1)
            self.assertEqual(flags & builder._COMPRESSION_FLAG_BIT, builder._COMPRESSION_FLAG_BIT)

            with open(path, 'rb') as handle:
                handle.seek(8)
                compressed_payload = handle.read()

            decompressor = zstd.ZstdDecompressor()
            payload = decompressor.decompress(compressed_payload)
            nr_frames = int.from_bytes(payload[:2], byteorder='little')
            self.assertEqual(nr_frames, 1)

if __name__ == '__main__':
    unittest.main()
