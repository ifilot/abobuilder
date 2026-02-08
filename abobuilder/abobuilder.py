from __future__ import annotations

from pyqint import PyQInt, Molecule
from pytessel import PyTessel
from abobuilder.element_table import ElementTable
from typing import Any, BinaryIO, Mapping, Optional, Sequence
import io
import numpy as np
import os
import zstandard as zstd

Vector3 = Sequence[float]
ModelData = Mapping[str, np.ndarray]
NucleusSpec = tuple[Vector3, Any]
OrbitalCollection = Mapping[str, Mapping[str, Any]]

class AboBuilder:
    """
    Build ABO/ABOF files containing geometry and orbital mesh data.
    """
    # Bit flag to mark compressed payloads in ABOF v1 headers.
    _COMPRESSION_FLAG_BIT = 0x01
    # Zstandard compression level used for ABOF payloads.
    _ZSTD_COMPRESSION_LEVEL = 22

    def __init__(
        self,
        occupied_colors: Optional[Sequence[Sequence[float]]] = None,
        unoccupied_colors: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """
        Initialize the builder with colors, orbital templates, and element table.

        Args:
            occupied_colors: Optional RGBA/RGB colors for occupied orbital lobes.
                Provide two colors (positive and negative lobe).
            unoccupied_colors: Optional RGBA/RGB colors for unoccupied orbital lobes.
                Provide two colors (positive and negative lobe).
        """
        # Global alpha used for all orbital colors.
        self.alpha: float = 0.97

        default_colors = [
            np.array([0.592, 0.796, 0.369, self.alpha], dtype=np.float32),
            np.array([0.831, 0.322, 0.604, self.alpha], dtype=np.float32),
            np.array([1.000, 0.612, 0.000, self.alpha], dtype=np.float32),
            np.array([0.400, 0.831, 0.706, self.alpha], dtype=np.float32)
        ]

        self.colors: list[np.ndarray] = self._build_orbital_colors(
            occupied_colors,
            unoccupied_colors,
            default_colors,
        )

        # Orbital label template used to map basis functions to CGFs.
        self.orbtemplate: list[str] = [
            '1s',
            '2s', '2px', '2py', '2pz',
            '3s', '3px', '3py', '3pz', '3dx2', '3dy2', '3dz2', '3dxy', '3dxz', '3dyz',
            '4s', '4px', '4py', '4pz'
        ]

        # Element table used for symbol -> atomic number conversions.
        self.et: ElementTable = ElementTable()

    def _normalize_color(self, color: Sequence[float]) -> np.ndarray:
        """Normalize a color sequence to RGBA float32."""
        if len(color) == 3:
            rgba = [color[0], color[1], color[2], self.alpha]
        elif len(color) == 4:
            rgba = list(color)
        else:
            raise ValueError("Colors must be RGB or RGBA sequences.")
        return np.array(rgba, dtype=np.float32)

    def _build_orbital_colors(
        self,
        occupied_colors: Optional[Sequence[Sequence[float]]],
        unoccupied_colors: Optional[Sequence[Sequence[float]]],
        default_colors: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        """Return the orbital colors list in occupied/unoccupied order."""
        if occupied_colors is None and unoccupied_colors is None:
            return list(default_colors)

        if occupied_colors is None or unoccupied_colors is None:
            raise ValueError("Both occupied_colors and unoccupied_colors must be provided.")

        if len(occupied_colors) != 2 or len(unoccupied_colors) != 2:
            raise ValueError("Expected two colors for occupied and two for unoccupied orbitals.")

        return [
            self._normalize_color(occupied_colors[0]),
            self._normalize_color(occupied_colors[1]),
            self._normalize_color(unoccupied_colors[0]),
            self._normalize_color(unoccupied_colors[1]),
        ]

    def _resolve_occupied_orbitals(
        self,
        nocc: Optional[int | str],
        nuclei: Sequence[NucleusSpec],
        orbital_count: int,
    ) -> int:
        """Determine the number of occupied orbitals."""
        if nocc is None:
            nelec = np.sum([atom[1] for atom in nuclei])
            return int(nelec / 2)
        if isinstance(nocc, str):
            if nocc == "all":
                return orbital_count
            raise ValueError("nocc must be an integer, None, or 'all'.")
        if nocc < 0:
            raise ValueError("nocc must be non-negative.")
        return int(nocc)

    def _write_file_header(self, f: BinaryIO, version: Optional[int] = None, flags: int = 0) -> None:
        """
        Write the "ABOF" header for the specified file format version.
        """
        if version is None or version == 0:
            return
        if version != 1:
            raise ValueError(f"Unsupported ABO format version: {version}")
        f.write(int(0).to_bytes(2, byteorder='little'))
        f.write(bytearray('ABOF', encoding='ascii'))
        f.write(int(version).to_bytes(1, byteorder='little'))
        f.write(int(flags).to_bytes(1, byteorder='little'))

    def _write_frame_header(self, f: BinaryIO, frame_idx: int, descriptor: str) -> None:
        """
        Write a frame header including the frame index and descriptor text.
        """
        f.write(int(frame_idx).to_bytes(2, byteorder='little'))
        f.write(len(descriptor).to_bytes(2, byteorder='little'))
        f.write(bytearray(descriptor, encoding='utf8'))

    def _should_compress_payload(self, flags: int) -> bool:
        """Return True when the payload compression flag bit is set."""
        return bool(flags & self._COMPRESSION_FLAG_BIT)

    def _compress_payload(self, payload: memoryview) -> bytes:
        """
        Compress payload bytes with Zstandard using the configured level.
        """
        compressor = zstd.ZstdCompressor(level=self._ZSTD_COMPRESSION_LEVEL)
        return compressor.compress(payload)

    def _octahedral_encode_normals(self, normals: np.ndarray) -> np.ndarray:
        """
        Encode normal vectors into octahedral 16-bit representation.
        """
        # Normalize input normals to unit vectors.
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        n = normals / norms
        # Project onto octahedral encoding space.
        denom = np.sum(np.abs(n), axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        n = n / denom
        enc = n[:, :2].copy()
        mask = n[:, 2] < 0
        if np.any(mask):
            enc[mask] = (1.0 - np.abs(enc[mask][:, ::-1])) * np.sign(enc[mask])
        scale = 32767.0
        return np.round(enc * scale).astype(np.int16)

    def _write_model(
        self,
        f: BinaryIO,
        model_idx: int,
        color: np.ndarray,
        vertices: np.ndarray,
        normals: np.ndarray,
        indices: np.ndarray,
        normal_encoding: str = "float32",
    ) -> None:
        """
        Write a single mesh model block (vertices, normals, indices) to the stream.
        """
        f.write(int(model_idx).to_bytes(2, byteorder='little'))
        f.write(np.array(color).tobytes())
        f.write(vertices.shape[0].to_bytes(4, byteorder='little'))
        if normal_encoding == "float32":
            # Interleave vertices and normals as float32 tuples.
            vertices_normals = np.hstack([vertices, normals])
            f.write(vertices_normals.tobytes())
        elif normal_encoding == "oct16":
            # Encode normals to int16 pairs for compact storage.
            encoded = self._octahedral_encode_normals(normals)
            # Pack vertices and encoded normals into a structured array.
            vertex_dtype = np.dtype(
                [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<i2'), ('ny', '<i2')]
            )
            vertex_data = np.empty(vertices.shape[0], dtype=vertex_dtype)
            # Assign vertex positions and encoded normal components.
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

    def build_abo_model(
        self,
        outfile: os.PathLike[str] | str,
        models: Sequence[ModelData],
        colors: Sequence[np.ndarray],
        geometry_descriptor: str = "Geometry",
    ) -> None:
        """
        Build a legacy ABO model file (v0) with provided mesh data.
        """
        self.build_abo_model_v0(outfile, models, colors, geometry_descriptor=geometry_descriptor)

    def build_abo_model_v0(
        self,
        outfile: os.PathLike[str] | str,
        models: Sequence[ModelData],
        colors: Sequence[np.ndarray],
        geometry_descriptor: str = "Geometry",
    ) -> None:
        """
        Build managlyph atom/bonds/orbitals file from raw model data.

        Args:
            outfile: Destination file path.
            models: Iterable of mesh dictionaries with vertices, normals, indices arrays.
            colors: RGBA colors for each model.
            geometry_descriptor: Descriptor text for the initial geometry frame.
        """
        # Wavefunction integrator required by interface (unused for raw models).
        integrator = PyQInt()

        # Marching cubes utility (unused for raw models).
        pytessel = PyTessel()

        # Output file handle for the ABO file.
        f = open(outfile, 'wb')

        # Single frame for the geometry.
        f.write(int(1).to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        self._write_frame_header(f, 1, geometry_descriptor)

        # write nr_atoms
        f.write(int(0).to_bytes(2, byteorder='little'))

        # write number of models
        f.write(int(len(models)).to_bytes(2, byteorder='little'))

        for i, model in enumerate(models):

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

    def build_abo_model_v1(
        self,
        outfile: os.PathLike[str] | str,
        models: Sequence[ModelData],
        colors: Sequence[np.ndarray],
        geometry_descriptor: str = "Geometry",
        flags: int = 0,
        compress: bool = False,
    ) -> None:
        """
        Build version 1 ABOF file using octahedral-encoded normals.

        Args:
            outfile: Destination file path.
            models: Iterable of mesh dictionaries with vertices, normals, indices arrays.
            colors: RGBA colors for each model.
            geometry_descriptor: Descriptor text for the initial geometry frame.
            flags: ABOF header flags.
            compress: Enable payload compression with Zstandard.
        """
        # Wavefunction integrator required by interface (unused for raw models).
        integrator = PyQInt()

        # Marching cubes utility (unused for raw models).
        pytessel = PyTessel()

        if compress:
            flags |= self._COMPRESSION_FLAG_BIT

        # Output file handle for the ABOF file.
        f = open(outfile, 'wb')
        self._write_file_header(f, version=1, flags=flags)

        # Buffer payload when compression is enabled.
        payload_stream: BinaryIO = io.BytesIO() if self._should_compress_payload(flags) else f

        # Single frame for the geometry.
        payload_stream.write(int(1).to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        self._write_frame_header(payload_stream, 1, geometry_descriptor)

        # write nr_atoms
        payload_stream.write(int(0).to_bytes(2, byteorder='little'))

        # write number of models
        payload_stream.write(int(len(models)).to_bytes(2, byteorder='little'))

        for i, model in enumerate(models):
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
            # Compress the payload buffer without duplicating it.
            buffer_view = payload_stream.getbuffer()
            try:
                f.write(self._compress_payload(buffer_view))
            finally:
                buffer_view.release()

        f.close()

        # report filesize
        print("Creating file: %s" % outfile)
        print("Size: %f MB" % (os.stat(outfile).st_size / (1024*1024)))

    def build_abo_orbs(
        self,
        outfile: os.PathLike[str] | str,
        nuclei: Sequence[NucleusSpec],
        orbs: OrbitalCollection,
        isovalue: float = 0.03,
        overwrite_nuclei: Optional[Sequence[str]] = None,
        geometry_descriptor: str = "Geometry",
    ) -> None:
        """Build a legacy ABO orbital file (v0) from orbital data."""
        self.build_abo_orbs_v0(
            outfile,
            nuclei,
            orbs,
            isovalue=isovalue,
            overwrite_nuclei=overwrite_nuclei,
            geometry_descriptor=geometry_descriptor,
        )

    def build_abo_orbs_v0(
        self,
        outfile: os.PathLike[str] | str,
        nuclei: Sequence[NucleusSpec],
        orbs: OrbitalCollection,
        isovalue: float = 0.03,
        overwrite_nuclei: Optional[Sequence[str]] = None,
        geometry_descriptor: str = "Geometry",
    ) -> None:
        """
        Build managlyph atom/bonds/orbitals file from previous HF calculation.

        Args:
            outfile: Destination file path.
            nuclei: Atom positions and element symbols for the molecule.
            orbs: Orbital data including orbitals and coefficients.
            isovalue: Isosurface value for marching cubes.
            overwrite_nuclei: Optional element symbols to override nuclei labels.
            geometry_descriptor: Descriptor text for the initial geometry frame.
        """
        # Wavefunction integrator used to generate orbital scalar fields.
        integrator = PyQInt()

        # Marching cubes utility for generating isosurfaces.
        pytessel = PyTessel()

        # Output file handle for the ABO file.
        f = open(outfile, 'wb')

        # Total frame count includes geometry plus one frame per orbital set.
        nr_frames = len(orbs) + 1
        f.write(nr_frames.to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        self._write_frame_header(f, 1, geometry_descriptor)

        # write nr_atoms
        f.write(len(nuclei).to_bytes(2, byteorder='little'))
        for i, atom in enumerate(nuclei):
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
        for i, key in enumerate(orbs):
            self._write_frame_header(f, i + 1, key)

            # write nr_atoms
            f.write(len(nuclei).to_bytes(2, byteorder='little'))
            for a, atom in enumerate(nuclei):
                if overwrite_nuclei:
                    f.write(self.et.atomic_number_from_element(overwrite_nuclei[a]).to_bytes(1, byteorder='little'))
                else:
                    f.write(self.et.atomic_number_from_element(atom[1]).to_bytes(1, byteorder='little'))
                f.write(np.array(atom[0], dtype=np.float32).tobytes())

            print('Writing MO #%02i' % (i+1))

            # write number of models
            # Number of orbitals in this frame.
            nrorbs = len(orbs[key]['orbitals'])
            f.write(int(nrorbs * 2.0).to_bytes(2, byteorder='little'))
            for j in range(0, nrorbs):
                # grab basis functions
                orb = orbs[key]['orbitals'][j]
                nucleus = nuclei[orb[0] - 1]
                at = Molecule()
                at.add_atom(nucleus[1], nucleus[0][0], nucleus[0][1], nucleus[0][2], unit='angstrom')
                cgfs, _ = at.build_basis('sto3g')
                print('    Reading %i CGFS' % len(cgfs))
                # scalar coefficient to multiply all coefficients with
                sc = orb[2]
                # One-hot coefficient vector for the target orbital.
                coeffvec = np.array([1.0 if self.orbtemplate[i] == orb[1] else 0.0 for i in range(0, len(cgfs))])

                # build the pos and negative isosurfaces from the cubefiles
                # Unit cell size in Bohr.
                usz = 7.0
                # Grid resolution along each axis.
                sz = 100
                # 3D grid of sampling points for the wavefunction.
                grid = integrator.build_rectgrid3d(-usz, usz, sz)
                # Scalar field values at each grid point.
                scalarfield = np.reshape(integrator.plot_wavefunction(grid, sc * coeffvec, cgfs), (sz, sz, sz))
                # Unit cell matrix used by marching cubes.
                unitcell = np.diag(np.ones(3) * (usz * 2.0))

                for k in range(0,2):
                    vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(), scalarfield.shape, unitcell.flatten(), isovalue if k ==0 else -isovalue)
                    # Convert vertices from Bohr to Angstrom.
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

    def build_abo_orbs_v1(
        self,
        outfile: os.PathLike[str] | str,
        nuclei: Sequence[NucleusSpec],
        orbs: OrbitalCollection,
        isovalue: float = 0.03,
        overwrite_nuclei: Optional[Sequence[str]] = None,
        geometry_descriptor: str = "Geometry",
        flags: int = 0,
        compress: bool = False,
    ) -> None:
        """
        Build version 1 ABOF file using octahedral-encoded normals.

        Args:
            outfile: Destination file path.
            nuclei: Atom positions and element symbols for the molecule.
            orbs: Orbital data including orbitals and coefficients.
            isovalue: Isosurface value for marching cubes.
            overwrite_nuclei: Optional element symbols to override nuclei labels.
            geometry_descriptor: Descriptor text for the initial geometry frame.
            flags: ABOF header flags.
            compress: Enable payload compression with Zstandard.
        """
        # Wavefunction integrator used to generate orbital scalar fields.
        integrator = PyQInt()

        # Marching cubes utility for generating isosurfaces.
        pytessel = PyTessel()

        if compress:
            flags |= self._COMPRESSION_FLAG_BIT

        # Output file handle for the ABOF file.
        f = open(outfile, 'wb')
        self._write_file_header(f, version=1, flags=flags)

        # Buffer payload when compression is enabled.
        payload_stream: BinaryIO = io.BytesIO() if self._should_compress_payload(flags) else f

        # Total frame count includes geometry plus one frame per orbital set.
        nr_frames = len(orbs) + 1
        payload_stream.write(nr_frames.to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        self._write_frame_header(payload_stream, 1, geometry_descriptor)

        # write nr_atoms
        payload_stream.write(len(nuclei).to_bytes(2, byteorder='little'))
        for i, atom in enumerate(nuclei):
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
        for i, key in enumerate(orbs):
            self._write_frame_header(payload_stream, i + 1, key)

            # write nr_atoms
            payload_stream.write(len(nuclei).to_bytes(2, byteorder='little'))
            for a, atom in enumerate(nuclei):
                if overwrite_nuclei:
                    payload_stream.write(self.et.atomic_number_from_element(overwrite_nuclei[a]).to_bytes(1, byteorder='little'))
                else:
                    payload_stream.write(self.et.atomic_number_from_element(atom[1]).to_bytes(1, byteorder='little'))
                payload_stream.write(np.array(atom[0], dtype=np.float32).tobytes())

            print('Writing MO #%02i' % (i+1))

            # write number of models
            # Number of orbitals in this frame.
            nrorbs = len(orbs[key]['orbitals'])
            payload_stream.write(int(nrorbs * 2.0).to_bytes(2, byteorder='little'))
            for j in range(0, nrorbs):
                # grab basis functions
                orb = orbs[key]['orbitals'][j]
                nucleus = nuclei[orb[0] - 1]
                at = Molecule()
                at.add_atom(nucleus[1], nucleus[0][0], nucleus[0][1], nucleus[0][2], unit='angstrom')
                cgfs, _ = at.build_basis('sto3g')
                print('    Reading %i CGFS' % len(cgfs))
                # scalar coefficient to multiply all coefficients with
                sc = orb[2]
                # One-hot coefficient vector for the target orbital.
                coeffvec = np.array([1.0 if self.orbtemplate[i] == orb[1] else 0.0 for i in range(0, len(cgfs))])

                # build the pos and negative isosurfaces from the cubefiles
                # Unit cell size in Bohr.
                usz = 7.0
                # Grid resolution along each axis.
                sz = 100
                # 3D grid of sampling points for the wavefunction.
                grid = integrator.build_rectgrid3d(-usz, usz, sz)
                # Scalar field values at each grid point.
                scalarfield = np.reshape(integrator.plot_wavefunction(grid, sc * coeffvec, cgfs), (sz, sz, sz))
                # Unit cell matrix used by marching cubes.
                unitcell = np.diag(np.ones(3) * (usz * 2.0))

                for k in range(0,2):
                    vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(), scalarfield.shape, unitcell.flatten(), isovalue if k ==0 else -isovalue)
                    # Convert vertices from Bohr to Angstrom.
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
            # Compress the payload buffer without duplicating it.
            buffer_view = payload_stream.getbuffer()
            try:
                f.write(self._compress_payload(buffer_view))
            finally:
                buffer_view.release()

        f.close()

        # report filesize
        print("Creating file: %s" % outfile)
        print("Size: %f MB" % (os.stat(outfile).st_size / (1024*1024)))

    def build_abo_hf(
        self,
        outfile: os.PathLike[str] | str,
        nuclei: Sequence[NucleusSpec],
        cgfs: Sequence[Any],
        coeff: np.ndarray,
        energies: Sequence[float],
        nocc: Optional[int | str] = None,
        isovalue: float = 0.03,
        maxmo: int = -1,
        sz: float = 5.0,
        nsamples: int = 100,
        geometry_descriptor: str = "Geometry",
    ) -> None:
        """Build a legacy ABO Hartree-Fock file (v0) from HF results."""
        self.build_abo_hf_v0(
            outfile,
            nuclei,
            cgfs,
            coeff,
            energies,
            nocc=nocc,
            isovalue=isovalue,
            maxmo=maxmo,
            sz=sz,
            nsamples=nsamples,
            geometry_descriptor=geometry_descriptor,
        )

    def build_abo_hf_v0(
        self,
        outfile: os.PathLike[str] | str,
        nuclei: Sequence[NucleusSpec],
        cgfs: Sequence[Any],
        coeff: np.ndarray,
        energies: Sequence[float],
        nocc: Optional[int | str] = None,
        isovalue: float = 0.03,
        maxmo: int = -1,
        sz: float = 5.0,
        nsamples: int = 100,
        geometry_descriptor: str = "Geometry",
    ) -> None:
        """
        Build managlyph atom/bonds/orbitals file from
        previous HF calculation.

        Args:
            outfile: Destination file path.
            nuclei: Atom positions and atomic numbers for the molecule.
            cgfs: Contracted Gaussian basis functions.
            coeff: Molecular orbital coefficient matrix.
            energies: Orbital energies (eV).
            nocc: Number of occupied orbitals. Use None to infer from electron count,
                or "all" to mark all orbitals as occupied.
            isovalue: Isosurface value for marching cubes.
            maxmo: Limit number of molecular orbitals (-1 for all).
            sz: Half-length of the sampling cube in Bohr.
            nsamples: Samples per axis for the grid.
            geometry_descriptor: Descriptor text for the initial geometry frame.
        """
        # Wavefunction integrator used to generate orbital scalar fields.
        integrator = PyQInt()

        # Marching cubes utility for generating isosurfaces.
        pytessel = PyTessel()

        # Output file handle for the ABO file.
        f = open(outfile, 'wb')

        # Total frame count includes geometry plus one frame per orbital.
        nr_frames = len(cgfs) + 1 if maxmo == -1 else maxmo + 1
        f.write(nr_frames.to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        self._write_frame_header(f, 1, geometry_descriptor)

        # write nr_atoms
        f.write(len(nuclei).to_bytes(2, byteorder='little'))
        for atom in nuclei:
            f.write(atom[1].to_bytes(1, byteorder='little'))
            f.write(np.array(atom[0] * 0.529177, dtype=np.float32).tobytes())

        # write number of models
        f.write(int(0).to_bytes(1, byteorder='little'))
        f.write(int(0).to_bytes(1, byteorder='little'))

        orbital_count = nr_frames - 1
        occupied_orbitals = self._resolve_occupied_orbitals(nocc, nuclei, orbital_count)

        # Precompute basis function scalar fields for efficient MO evaluation.
        grid = integrator.build_rectgrid3d(-sz, sz, nsamples)
        basis_fields = np.array(
            [integrator.plot_basis_function(grid, cgf) for cgf in cgfs],
            dtype=np.float64,
        ).reshape(len(cgfs), nsamples, nsamples, nsamples)

        #
        # Write the geometry including the orbitals
        #
        for i in range(1, nr_frames):
            # Frame descriptor includes orbital index and energy.
            descriptor = 'Molecular orbital %i\nEnergy: %.4f eV' % (i, energies[i - 1])
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
                # Scalar field values at each grid point.
                scalarfield = np.einsum(
                    "bxyz,b->xyz",
                    basis_fields,
                    coeff[:, i - 1],
                    optimize=True,
                )
                # Unit cell matrix used by marching cubes.
                unitcell = np.diag(np.ones(3) * (sz * 2.0))
                vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(), scalarfield.shape, unitcell.flatten(), isovalue if j == 1 else -isovalue)
                # Convert vertices from Bohr to Angstrom.
                vertices_scaled = vertices * 0.529177

                if i <= occupied_orbitals:
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

    def build_abo_hf_v1(
        self,
        outfile: os.PathLike[str] | str,
        nuclei: Sequence[NucleusSpec],
        cgfs: Sequence[Any],
        coeff: np.ndarray,
        energies: Sequence[float],
        nocc: Optional[int | str] = None,
        isovalue: float = 0.03,
        maxmo: int = -1,
        sz: float = 5.0,
        nsamples: int = 100,
        geometry_descriptor: str = "Geometry",
        flags: int = 0,
        compress: bool = False,
    ) -> None:
        """
        Build version 1 ABOF file using octahedral-encoded normals.

        Args:
            outfile: Destination file path.
            nuclei: Atom positions and atomic numbers for the molecule.
            cgfs: Contracted Gaussian basis functions.
            coeff: Molecular orbital coefficient matrix.
            energies: Orbital energies (eV).
            nocc: Number of occupied orbitals. Use None to infer from electron count,
                or "all" to mark all orbitals as occupied.
            isovalue: Isosurface value for marching cubes.
            maxmo: Limit number of molecular orbitals (-1 for all).
            sz: Half-length of the sampling cube in Bohr.
            nsamples: Samples per axis for the grid.
            geometry_descriptor: Descriptor text for the initial geometry frame.
            flags: ABOF header flags.
            compress: Enable payload compression with Zstandard.
        """
        # Wavefunction integrator used to generate orbital scalar fields.
        integrator = PyQInt()

        # Marching cubes utility for generating isosurfaces.
        pytessel = PyTessel()

        if compress:
            flags |= self._COMPRESSION_FLAG_BIT

        # Output file handle for the ABOF file.
        f = open(outfile, 'wb')
        self._write_file_header(f, version=1, flags=flags)

        # Buffer payload when compression is enabled.
        payload_stream: BinaryIO = io.BytesIO() if self._should_compress_payload(flags) else f

        # Total frame count includes geometry plus one frame per orbital.
        nr_frames = len(cgfs) + 1 if maxmo == -1 else maxmo + 1
        payload_stream.write(nr_frames.to_bytes(2, byteorder='little'))

        #
        # First write the bare geometry of the molecule
        #

        self._write_frame_header(payload_stream, 1, geometry_descriptor)

        # write nr_atoms
        payload_stream.write(len(nuclei).to_bytes(2, byteorder='little'))
        for atom in nuclei:
            payload_stream.write(atom[1].to_bytes(1, byteorder='little'))
            payload_stream.write(np.array(atom[0] * 0.529177, dtype=np.float32).tobytes())

        # write number of models
        payload_stream.write(int(0).to_bytes(1, byteorder='little'))
        payload_stream.write(int(0).to_bytes(1, byteorder='little'))

        orbital_count = nr_frames - 1
        occupied_orbitals = self._resolve_occupied_orbitals(nocc, nuclei, orbital_count)

        # Precompute basis function scalar fields for efficient MO evaluation.
        grid = integrator.build_rectgrid3d(-sz, sz, nsamples)
        basis_fields = np.array(
            [integrator.plot_basis_function(grid, cgf) for cgf in cgfs],
            dtype=np.float64,
        ).reshape(len(cgfs), nsamples, nsamples, nsamples)

        #
        # Write the geometry including the orbitals
        #
        for i in range(1, nr_frames):
            # Frame descriptor includes orbital index and energy.
            descriptor = 'Molecular orbital %i\nEnergy: %.4f eV' % (i, energies[i - 1])
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
                # Scalar field values at each grid point.
                scalarfield = np.einsum(
                    "bxyz,b->xyz",
                    basis_fields,
                    coeff[:, i - 1],
                    optimize=True,
                )
                # Unit cell matrix used by marching cubes.
                unitcell = np.diag(np.ones(3) * (sz * 2.0))
                vertices, normals, indices = pytessel.marching_cubes(scalarfield.flatten(), scalarfield.shape, unitcell.flatten(), isovalue if j == 1 else -isovalue)
                # Convert vertices from Bohr to Angstrom.
                vertices_scaled = vertices * 0.529177

                if i <= occupied_orbitals:
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
            # Compress the payload buffer without duplicating it.
            buffer_view = payload_stream.getbuffer()
            try:
                f.write(self._compress_payload(buffer_view))
            finally:
                buffer_view.release()

        f.close()

        # report filesize
        print("Creating file: %s" % outfile)
        print("Size: %f MB" % (os.stat(outfile).st_size / (1024*1024)))
