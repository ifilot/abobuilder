from pyqint import PyQInt, Molecule
from pytessel import PyTessel
from abobuilder.element_table import ElementTable
import io
import numpy as np
import os
import zstandard as zstd

class AboBuilder:
    _COMPRESSION_FLAG_BIT = 0x01
    _ZSTD_COMPRESSION_LEVEL = 22

    def __init__(self):
        # set transparency
        self.alpha = 0.97

        # specify colors for occupied and virtual orbitals
        self.colors = [
            np.array([0.592, 0.796, 0.369, self.alpha], dtype=np.float32),
            np.array([0.831, 0.322, 0.604, self.alpha], dtype=np.float32),
            np.array([1.000, 0.612, 0.000, self.alpha], dtype=np.float32),
            np.array([0.400, 0.831, 0.706, self.alpha], dtype=np.float32)
        ]

        self.orbtemplate = [
            '1s',
            '2s', '2px', '2py', '2pz',
            '3s', '3px', '3py', '3pz', '3dx2', '3dy2', '3dz2', '3dxy', '3dxz', '3dyz',
            '4s', '4px', '4py', '4pz'
        ]

        self.et = ElementTable()

    def _write_file_header(self, f, version=None, flags=0):
        if version is None or version == 0:
            return
        if version != 1:
            raise ValueError(f"Unsupported ABO format version: {version}")
        f.write(int(0).to_bytes(2, byteorder='little'))
        f.write(bytearray('ABOF', encoding='ascii'))
        f.write(int(version).to_bytes(1, byteorder='little'))
        f.write(int(flags).to_bytes(1, byteorder='little'))

    def _write_frame_header(self, f, frame_idx, descriptor):
        f.write(int(frame_idx).to_bytes(2, byteorder='little'))
        f.write(len(descriptor).to_bytes(2, byteorder='little'))
        f.write(bytearray(descriptor, encoding='utf8'))

    def _should_compress_payload(self, flags):
        return bool(flags & self._COMPRESSION_FLAG_BIT)

    def _compress_payload(self, payload):
        compressor = zstd.ZstdCompressor(level=self._ZSTD_COMPRESSION_LEVEL)
        return compressor.compress(payload)

    def _octahedral_encode_normals(self, normals):
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        n = normals / norms
        denom = np.sum(np.abs(n), axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        n = n / denom
        enc = n[:, :2].copy()
        mask = n[:, 2] < 0
        if np.any(mask):
            enc[mask] = (1.0 - np.abs(enc[mask][:, ::-1])) * np.sign(enc[mask])
        scale = 32767.0
        return np.round(enc * scale).astype(np.int16)

    def _write_model(self, f, model_idx, color, vertices, normals, indices, normal_encoding="float32"):
        f.write(int(model_idx).to_bytes(2, byteorder='little'))
        f.write(np.array(color).tobytes())
        f.write(vertices.shape[0].to_bytes(4, byteorder='little'))
        if normal_encoding == "float32":
            vertices_normals = np.hstack([vertices, normals])
            f.write(vertices_normals.tobytes())
        elif normal_encoding == "oct16":
            encoded = self._octahedral_encode_normals(normals)
            vertex_dtype = np.dtype(
                [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<i2'), ('ny', '<i2')]
            )
            vertex_data = np.empty(vertices.shape[0], dtype=vertex_dtype)
            vertex_data['x'] = vertices[:, 0]
            vertex_data['y'] = vertices[:, 1]
            vertex_data['z'] = vertices[:, 2]
            vertex_data['nx'] = encoded[:, 0]
            vertex_data['ny'] = encoded[:, 1]
            f.write(vertex_data.tobytes())
        else:
            raise ValueError(f"Unsupported normal encoding: {normal_encoding}")
        f.write(int(len(indices) / 3).to_bytes(4, byteorder='little'))
        f.write(indices.tobytes())

    def build_abo_model(self, outfile, models, colors):
        self.build_abo_model_v0(outfile, models, colors)

    def build_abo_model_v0(self, outfile, models, colors):
        """
        Build managlyph atom/bonds/orbitals file from raw model data
        """
        # build integrator
        integrator = PyQInt()

        # build pytessel object
        pytessel = PyTessel()

        # build output file
        f = open(outfile, 'wb')

        # write number of frames
        f.write(int(1).to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        self._write_frame_header(f, 1, 'Geometry')

        # write nr_atoms
        f.write(int(0).to_bytes(2, byteorder='little'))

        # write number of models
        f.write(int(len(models)).to_bytes(2, byteorder='little'))

        for i,model in enumerate(models):

            self._write_model(
                f,
                i,
                colors[i],
                model['vertices'],
                model['normals'],
                model['indices'],
                normal_encoding="float32",
            )

        f.close()

        # report filesize
        print("Creating file: %s" % outfile)
        print("Size: %f MB" % (os.stat(outfile).st_size / (1024*1024)))

    def build_abo_model_v1(self, outfile, models, colors, flags=0, compress=False):
        """
        Build version 1 ABOF file using octahedral-encoded normals.
        """
        # build integrator
        integrator = PyQInt()

        # build pytessel object
        pytessel = PyTessel()

        if compress:
            flags |= self._COMPRESSION_FLAG_BIT

        # build output file
        f = open(outfile, 'wb')
        self._write_file_header(f, version=1, flags=flags)

        payload_stream = io.BytesIO() if self._should_compress_payload(flags) else f

        # write number of frames
        payload_stream.write(int(1).to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        self._write_frame_header(payload_stream, 1, 'Geometry')

        # write nr_atoms
        payload_stream.write(int(0).to_bytes(2, byteorder='little'))

        # write number of models
        payload_stream.write(int(len(models)).to_bytes(2, byteorder='little'))

        for i,model in enumerate(models):
            self._write_model(
                payload_stream,
                i,
                colors[i],
                model['vertices'],
                model['normals'],
                model['indices'],
                normal_encoding="oct16",
            )

        if isinstance(payload_stream, io.BytesIO):
            f.write(self._compress_payload(payload_stream.getvalue()))

        f.close()

        # report filesize
        print("Creating file: %s" % outfile)
        print("Size: %f MB" % (os.stat(outfile).st_size / (1024*1024)))

    def build_abo_orbs(self, outfile, nuclei, orbs, isovalue=0.03, overwrite_nuclei = None):
        self.build_abo_orbs_v0(outfile, nuclei, orbs, isovalue=isovalue, overwrite_nuclei=overwrite_nuclei)

    def build_abo_orbs_v0(self, outfile, nuclei, orbs, isovalue=0.03, overwrite_nuclei = None):
        """
        Build managlyph atom/bonds/orbitals file from previous HF calculation
        """
        # build integrator
        integrator = PyQInt()

        # build pytessel object
        pytessel = PyTessel()

        # build output file
        f = open(outfile, 'wb')

        # write number of frames
        nr_frames = len(orbs) + 1
        f.write(nr_frames.to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        self._write_frame_header(f, 1, 'Geometry')

        # write nr_atoms
        f.write(len(nuclei).to_bytes(2, byteorder='little'))
        for i,atom in enumerate(nuclei):
            if overwrite_nuclei:
                f.write(self.et.atomic_number_from_element(overwrite_nuclei[i]).to_bytes(1, byteorder='little'))
            else:
                f.write(self.et.atomic_number_from_element(atom[1]).to_bytes(1, byteorder='little'))
            f.write(np.array(atom[0], dtype=np.float32).tobytes())

        # write number of models
        f.write(int(0).to_bytes(1, byteorder='little'))
        f.write(int(0).to_bytes(1, byteorder='little'))

        #
        # Write the geometry including the orbitals
        #
        for i,key in enumerate(orbs):
            self._write_frame_header(f, i + 1, key)

            # write nr_atoms
            f.write(len(nuclei).to_bytes(2, byteorder='little'))
            for a,atom in enumerate(nuclei):
                if overwrite_nuclei:
                    f.write(self.et.atomic_number_from_element(overwrite_nuclei[a]).to_bytes(1, byteorder='little'))
                else:
                    f.write(self.et.atomic_number_from_element(atom[1]).to_bytes(1, byteorder='little'))
                f.write(np.array(atom[0], dtype=np.float32).tobytes())

            print('Writing MO #%02i' % (i+1))

            # write number of models
            nrorbs = len(orbs[key]['orbitals'])
            f.write(int(nrorbs * 2.0).to_bytes(2, byteorder='little'))
            for j in range(0, nrorbs):
                # grab basis functions
                orb = orbs[key]['orbitals'][j]
                nucleus = nuclei[orb[0]-1]
                at = Molecule()
                at.add_atom(nucleus[1], nucleus[0][0], nucleus[0][1], nucleus[0][2], unit='angstrom')
                cgfs, _ = at.build_basis('sto3g')
                print('    Reading %i CGFS' % len(cgfs))
                sc = orb[2] # scalar coefficient to multiply all coefs with
                coeffvec = np.array([1.0 if self.orbtemplate[i] == orb[1] else 0.0 for i in range(0,len(cgfs))])

                # build the pos and negative isosurfaces from the cubefiles
                usz = 7.0 # unitcell size
                sz = 100
                grid = integrator.build_rectgrid3d(-usz, usz, sz)
                scalarfield = np.reshape(integrator.plot_wavefunction(grid, sc * coeffvec, cgfs), (sz, sz, sz))
                unitcell = np.diag(np.ones(3) * (usz * 2.0))

                for k in range(0,2):
                    vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(), scalarfield.shape, unitcell.flatten(), isovalue if k ==0 else -isovalue)
                    vertices_scaled = vertices * 0.529177

                    self._write_model(
                        f,
                        j * 2 + k,
                        self.colors[k],
                        vertices_scaled,
                        normals,
                        indices,
                        normal_encoding="float32",
                    )

                    if k == 0:
                        print('    Writing positive lobe: %i vertices and %i facets' % (vertices_scaled.shape[0], indices.shape[0] / 3))
                    else:
                        print('    Writing negative lobe: %i vertices and %i facets' % (vertices_scaled.shape[0], indices.shape[0] / 3))

        f.close()

        # report filesize
        print("Creating file: %s" % outfile)
        print("Size: %f MB" % (os.stat(outfile).st_size / (1024*1024)))

    def build_abo_orbs_v1(self, outfile, nuclei, orbs, isovalue=0.03, overwrite_nuclei = None, flags=0, compress=False):
        """
        Build version 1 ABOF file using octahedral-encoded normals.
        """
        # build integrator
        integrator = PyQInt()

        # build pytessel object
        pytessel = PyTessel()

        if compress:
            flags |= self._COMPRESSION_FLAG_BIT

        # build output file
        f = open(outfile, 'wb')
        self._write_file_header(f, version=1, flags=flags)

        payload_stream = io.BytesIO() if self._should_compress_payload(flags) else f

        # write number of frames
        nr_frames = len(orbs) + 1
        payload_stream.write(nr_frames.to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        self._write_frame_header(payload_stream, 1, 'Geometry')

        # write nr_atoms
        payload_stream.write(len(nuclei).to_bytes(2, byteorder='little'))
        for i,atom in enumerate(nuclei):
            if overwrite_nuclei:
                payload_stream.write(self.et.atomic_number_from_element(overwrite_nuclei[i]).to_bytes(1, byteorder='little'))
            else:
                payload_stream.write(self.et.atomic_number_from_element(atom[1]).to_bytes(1, byteorder='little'))
            payload_stream.write(np.array(atom[0], dtype=np.float32).tobytes())

        # write number of models
        payload_stream.write(int(0).to_bytes(1, byteorder='little'))
        payload_stream.write(int(0).to_bytes(1, byteorder='little'))

        #
        # Write the geometry including the orbitals
        #
        for i,key in enumerate(orbs):
            self._write_frame_header(payload_stream, i + 1, key)

            # write nr_atoms
            payload_stream.write(len(nuclei).to_bytes(2, byteorder='little'))
            for a,atom in enumerate(nuclei):
                if overwrite_nuclei:
                    payload_stream.write(self.et.atomic_number_from_element(overwrite_nuclei[a]).to_bytes(1, byteorder='little'))
                else:
                    payload_stream.write(self.et.atomic_number_from_element(atom[1]).to_bytes(1, byteorder='little'))
                payload_stream.write(np.array(atom[0], dtype=np.float32).tobytes())

            print('Writing MO #%02i' % (i+1))

            # write number of models
            nrorbs = len(orbs[key]['orbitals'])
            payload_stream.write(int(nrorbs * 2.0).to_bytes(2, byteorder='little'))
            for j in range(0, nrorbs):
                # grab basis functions
                orb = orbs[key]['orbitals'][j]
                nucleus = nuclei[orb[0]-1]
                at = Molecule()
                at.add_atom(nucleus[1], nucleus[0][0], nucleus[0][1], nucleus[0][2], unit='angstrom')
                cgfs, _ = at.build_basis('sto3g')
                print('    Reading %i CGFS' % len(cgfs))
                sc = orb[2] # scalar coefficient to multiply all coefs with
                coeffvec = np.array([1.0 if self.orbtemplate[i] == orb[1] else 0.0 for i in range(0,len(cgfs))])

                # build the pos and negative isosurfaces from the cubefiles
                usz = 7.0 # unitcell size
                sz = 100
                grid = integrator.build_rectgrid3d(-usz, usz, sz)
                scalarfield = np.reshape(integrator.plot_wavefunction(grid, sc * coeffvec, cgfs), (sz, sz, sz))
                unitcell = np.diag(np.ones(3) * (usz * 2.0))

                for k in range(0,2):
                    vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(), scalarfield.shape, unitcell.flatten(), isovalue if k ==0 else -isovalue)
                    vertices_scaled = vertices * 0.529177

                    self._write_model(
                        payload_stream,
                        j * 2 + k,
                        self.colors[k],
                        vertices_scaled,
                        normals,
                        indices,
                        normal_encoding="oct16",
                    )

                    if k == 0:
                        print('    Writing positive lobe: %i vertices and %i facets' % (vertices_scaled.shape[0], indices.shape[0] / 3))
                    else:
                        print('    Writing negative lobe: %i vertices and %i facets' % (vertices_scaled.shape[0], indices.shape[0] / 3))

        if isinstance(payload_stream, io.BytesIO):
            f.write(self._compress_payload(payload_stream.getvalue()))

        f.close()

        # report filesize
        print("Creating file: %s" % outfile)
        print("Size: %f MB" % (os.stat(outfile).st_size / (1024*1024)))

    def build_abo_hf(self, outfile, nuclei, cgfs, coeff, energies, isovalue=0.03, maxmo=-1, sz=5.0, nsamples=100):
        self.build_abo_hf_v0(outfile, nuclei, cgfs, coeff, energies, isovalue=isovalue, maxmo=maxmo, sz=sz, nsamples=nsamples)

    def build_abo_hf_v0(self, outfile, nuclei, cgfs, coeff, energies, isovalue=0.03, maxmo=-1, sz=5.0, nsamples=100):
        """
        Build managlyph atom/bonds/orbitals file from
        previous HF calculation
        """
        # build integrator
        integrator = PyQInt()

        # build pytessel object
        pytessel = PyTessel()

        # build output file
        f = open(outfile, 'wb')

        # write number of frames
        nr_frames = len(cgfs) + 1 if maxmo == -1 else maxmo + 1
        f.write(nr_frames.to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        self._write_frame_header(f, 1, 'Geometry')

        # write nr_atoms
        f.write(len(nuclei).to_bytes(2, byteorder='little'))
        for atom in nuclei:
            f.write(atom[1].to_bytes(1, byteorder='little'))
            f.write(np.array(atom[0] * 0.529177, dtype=np.float32).tobytes())

        # write number of models
        f.write(int(0).to_bytes(1, byteorder='little'))
        f.write(int(0).to_bytes(1, byteorder='little'))

        # calculate number of electrons
        nelec = np.sum([atom[1] for atom in nuclei])

        #
        # Write the geometry including the orbitals
        #
        for i in range(1, nr_frames):
            descriptor = 'Molecular orbital %i\nEnergy: %.4f eV' % (i,energies[i-1])
            self._write_frame_header(f, i + 1, descriptor)

            # write nr_atoms
            f.write(len(nuclei).to_bytes(2, byteorder='little'))
            for atom in nuclei:
                f.write(atom[1].to_bytes(1, byteorder='little'))
                f.write(np.array(atom[0] * 0.529177, dtype=np.float32).tobytes())

            print('Writing MO #%02i' % i)

            # write number of models
            f.write(int(2).to_bytes(2, byteorder='little'))
            for j in range(0, 2):
                # build the pos and negative isosurfaces from the cubefiles
                grid = integrator.build_rectgrid3d(-sz, sz, nsamples)
                scalarfield = np.reshape(integrator.plot_wavefunction(grid, coeff[:,i-1], cgfs), (nsamples, nsamples, nsamples))
                unitcell = np.diag(np.ones(3) * (sz * 2.0))
                vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(), scalarfield.shape, unitcell.flatten(), isovalue if j==1 else -isovalue)
                vertices_scaled = vertices * 0.529177

                if i <= nelec / 2:
                    color = np.array(self.colors[j])
                else:
                    color = np.array(self.colors[j+2])

                self._write_model(
                    f,
                    j,
                    color,
                    vertices_scaled,
                    normals,
                    indices,
                    normal_encoding="float32",
                )

                if j == 0:
                    print('    Writing positive lobe: %i vertices and %i facets' % (vertices_scaled.shape[0], indices.shape[0] / 3))
                else:
                    print('    Writing negative lobe: %i vertices and %i facets' % (vertices_scaled.shape[0], indices.shape[0] / 3))

        f.close()

        # report filesize
        print("Creating file: %s" % outfile)
        print("Size: %f MB" % (os.stat(outfile).st_size / (1024*1024)))

    def build_abo_hf_v1(self, outfile, nuclei, cgfs, coeff, energies, isovalue=0.03, maxmo=-1, sz=5.0, nsamples=100, flags=0, compress=False):
        """
        Build version 1 ABOF file using octahedral-encoded normals.
        """
        # build integrator
        integrator = PyQInt()

        # build pytessel object
        pytessel = PyTessel()

        if compress:
            flags |= self._COMPRESSION_FLAG_BIT

        # build output file
        f = open(outfile, 'wb')
        self._write_file_header(f, version=1, flags=flags)

        payload_stream = io.BytesIO() if self._should_compress_payload(flags) else f

        # write number of frames
        nr_frames = len(cgfs) + 1 if maxmo == -1 else maxmo + 1
        payload_stream.write(nr_frames.to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        self._write_frame_header(payload_stream, 1, 'Geometry')

        # write nr_atoms
        payload_stream.write(len(nuclei).to_bytes(2, byteorder='little'))
        for atom in nuclei:
            payload_stream.write(atom[1].to_bytes(1, byteorder='little'))
            payload_stream.write(np.array(atom[0] * 0.529177, dtype=np.float32).tobytes())

        # write number of models
        payload_stream.write(int(0).to_bytes(1, byteorder='little'))
        payload_stream.write(int(0).to_bytes(1, byteorder='little'))

        # calculate number of electrons
        nelec = np.sum([atom[1] for atom in nuclei])

        #
        # Write the geometry including the orbitals
        #
        for i in range(1, nr_frames):
            descriptor = 'Molecular orbital %i\nEnergy: %.4f eV' % (i,energies[i-1])
            self._write_frame_header(payload_stream, i + 1, descriptor)

            # write nr_atoms
            payload_stream.write(len(nuclei).to_bytes(2, byteorder='little'))
            for atom in nuclei:
                payload_stream.write(atom[1].to_bytes(1, byteorder='little'))
                payload_stream.write(np.array(atom[0] * 0.529177, dtype=np.float32).tobytes())

            print('Writing MO #%02i' % i)

            # write number of models
            payload_stream.write(int(2).to_bytes(2, byteorder='little'))
            for j in range(0, 2):
                # build the pos and negative isosurfaces from the cubefiles
                grid = integrator.build_rectgrid3d(-sz, sz, nsamples)
                scalarfield = np.reshape(integrator.plot_wavefunction(grid, coeff[:,i-1], cgfs), (nsamples, nsamples, nsamples))
                unitcell = np.diag(np.ones(3) * (sz * 2.0))
                vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(), scalarfield.shape, unitcell.flatten(), isovalue if j==1 else -isovalue)
                vertices_scaled = vertices * 0.529177

                if i <= nelec / 2:
                    color = np.array(self.colors[j])
                else:
                    color = np.array(self.colors[j+2])

                self._write_model(
                    payload_stream,
                    j,
                    color,
                    vertices_scaled,
                    normals,
                    indices,
                    normal_encoding="oct16",
                )

                if j == 0:
                    print('    Writing positive lobe: %i vertices and %i facets' % (vertices_scaled.shape[0], indices.shape[0] / 3))
                else:
                    print('    Writing negative lobe: %i vertices and %i facets' % (vertices_scaled.shape[0], indices.shape[0] / 3))

        if isinstance(payload_stream, io.BytesIO):
            f.write(self._compress_payload(payload_stream.getvalue()))

        f.close()

        # report filesize
        print("Creating file: %s" % outfile)
        print("Size: %f MB" % (os.stat(outfile).st_size / (1024*1024)))
