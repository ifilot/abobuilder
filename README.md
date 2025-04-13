# ABO Builder

**ABO Builder** is a Python package designed to build `.abo` files, which are
used by the [Managlyph](https://www.managlyph.nl) software to generate
stereoscopic visualizations, particularly for anaglyph (red/cyan) rendering of
molecular or structural models.

## Purpose

The goal of `abobuilder` is to simplify the creation of `.abo` packages —
structured collections of geometry, metadata, and styling instructions —
enabling users to create vivid anaglyph renderings of 3D scenes using the
Managlyph viewer.

## Features

- Build `.abo` files from electronic structure calculations
- Automatically include atomic and orbital data
- Set colors, transparency, and styling for each element
- Export `.abo` packages ready for use in Managlyph

## Installation

You can install `abobuilder` using pip:

```bash
pip install abobuilder
```

or via Anaconda

```bash
conda install -c ifilot abobuilder
```