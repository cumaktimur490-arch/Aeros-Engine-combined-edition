# Aeros Engine — combined edition

## What is in this repository

This repository currently contains **aerosGPU**: OpenCL linear solvers for OpenFOAM's
`lduMatrix` (`aerosPCG`, `aerosPBiCGStab`, plus an `aerosGpuInfo` diagnostic), in
[`applications/utilities/aerosGPU`](applications/utilities/aerosGPU/README.md).

The engine sources this workspace was originally branched for are **not** in this
repository: `main` here is a single `README.md` stub, and no Aeros, FluidX3D or OpenFOAM
tree exists in the history. The Aeros-side integration work (the FluidX3D lattice
backend, its UI panel and settings entries) therefore cannot be restored from this
clone, and is not something that can be rebuilt without those sources.

## Building and testing

The accelerator's core — row structure, preconditioners, Krylov solvers, device layer —
does not depend on OpenFOAM at all: it works on plain arrays and indices. That is what
makes it testable on a machine with neither an OpenFOAM installation nor an OpenCL
runtime.

```bash
make          # builds and runs the core suite: 24 checks
make asan     # the same under AddressSanitizer + UndefinedBehaviorSanitizer
```

Building the library for OpenFOAM itself needs an OpenFOAM installation and `wmake`:

```bash
cd applications/utilities/aerosGPU
wmake libso   # libaerosGpuSolvers.so -> $FOAM_USER_LIBBIN
wmake         # aerosGpuInfo          -> $FOAM_USER_APPBIN
```

`aerosGpuFoam.C` is syntax-checked against OpenFOAM-dev headers but has not been linked
or run here, because this environment has no OpenFOAM installation (and no OpenCL
runtime). Everything that can be verified without them is verified in `make`.
