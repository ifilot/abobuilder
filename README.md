# ABO Builder

![Version](https://img.shields.io/github/v/tag/ifilot/abobuilder?label=version)
[![PyPI pkg](https://github.com/ifilot/abobuilder/actions/workflows/build_pypi.yml/badge.svg)](https://github.com/ifilot/abobuilder/actions/workflows/build_pypi.yml)
[![PyPI](https://img.shields.io/pypi/v/abobuilder?style=flat-square)](https://pypi.org/project/abobuilder/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**ABO Builder** is a Python package designed to build `.abo` files, which are
used by the [Managlyph](https://ifilot.github.io/managlyph/) software to
generate stereoscopic visualizations, particularly for anaglyph (red/cyan)
rendering of molecular or structural models.

## Purpose

The goal of `abobuilder` is to simplify the creation of `.abo` packages —
structured collections of geometry, metadata, and styling instructions —
enabling users to create vivid anaglyph renderings of 3D scenes using the
Managlyph viewer.

## Features

- Build `.abo` files from electronic structure calculations
- Automatically include atomic and orbital data
- Configure occupied/unoccupied orbital colors and styling
- Export `.abo` packages ready for use in Managlyph

## Installation

You can install `abobuilder` using pip:

```bash
pip install abobuilder
```
