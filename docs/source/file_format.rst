ABO file format
===============

ABO Builder writes binary ``.abo`` files for Managlyph. The package supports
legacy v0 files and the newer ``ABOF`` container formats (v1 and v2).

Overview
--------

* **v0**: a legacy layout without a file header. Normals are stored as
  float32 vectors, and payloads are always uncompressed.
* **v1**: an ``ABOF`` container with a fixed header, optional Zstandard
  compression, and octahedral-encoded normal vectors for more compact mesh
  storage.
* **v2**: extends the payload frame layout with optional per-frame metadata
  sections (currently a unit-cell matrix).

ABOF header (v1 and v2)
--------------

ABOF v1/v2 files begin with a fixed 8-byte header before the payload:

#. ``uint16`` zero padding (reserved, currently ``0``)
#. ASCII ``"ABOF"`` marker
#. ``uint8`` version number (currently ``1`` or ``2``)
#. ``uint8`` flags bitfield

The writer currently uses two bits:

* ``0x01`` indicates payload compression.
* ``0x02`` marks trajectory files that contain a reaction event
  (for example NEB pathways).

Compression
-----------

When the compression flag is set, the writer buffers the entire payload in
memory and compresses it with **Zstandard** at level 22 before writing it to
disk. When the flag is not set, payload bytes are written directly to the file
stream. The compressed payload stores the exact same data order as the
uncompressed payload; only the container differs.

Payload structure
-----------------------------

The v0 and v1 payload layout is identical. v2 adds an optional frame metadata
byte (and optional metadata payloads) immediately after each frame descriptor.

Frame table
~~~~~~~~~~~

The file starts with the total number of frames:

* ``uint16`` frame count

Each frame is written sequentially as:

* ``uint16`` frame index (1-based)
* ``uint16`` descriptor length (bytes)
* UTF-8 descriptor text
* ``uint8`` frame metadata flags (**v2 only**)
* Optional metadata payloads (**v2 only**, see below)
* ``uint16`` number of atoms
* Atom blocks (see below)
* ``uint16`` number of models (meshes)
* Model blocks (see below)


Frame metadata flags (v2)
~~~~~~~~~~~~~~~~~~~~~~~~~

For ABOF v2 payloads, each frame includes a one-byte metadata flag field after
the descriptor text.

* Bit ``0x01``: unit-cell matrix is present and encoded as ``float32[9]``
  (row-major 3x3 matrix).

If no metadata bits are set, the byte is still present and set to ``0``.

Atom blocks
~~~~~~~~~~~

Each atom entry stores:

* ``uint8`` atomic number
* ``float32[3]`` XYZ coordinates

For Hartree-Fock files, coordinates are written in **Angstrom** (the builder
converts Bohr to Angstrom before writing).

Model blocks
~~~~~~~~~~~~

Each model (mesh) entry stores:

* ``uint16`` model index
* ``float32[4]`` RGBA color
* ``uint32`` vertex count
* Vertex payload
* ``uint32`` triangle count
* Triangle indices payload

The vertex payload is encoded as either:

* **v0**: interleaved ``float32`` positions and normals
  ``[x, y, z, nx, ny, nz]`` per vertex.
* **v1**: interleaved ``float32`` positions followed by **octahedral-encoded**
  normal vectors stored as ``int16[2]`` per vertex.

Triangle indices are written as the raw NumPy array bytes emitted by the
marching-cubes pipeline (typically ``int32`` indices). The triangle count is
``len(indices) / 3`` and is stored as ``uint32`` before the index payload.

Normals and octahedral encoding
-------------------------------

Version 1 files compress normal vectors using an octahedral projection. Normals
are first normalized, projected to the octahedron, and then stored as two
signed 16-bit integers. This reduces storage while keeping normals unit-length
when decoded.
