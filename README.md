# Aeros Engine 1.38.1 + aerosGPU

This repository holds two things that are deliberately kept apart:

- **`aeros/`, `libs/`, `docs/`, …** — the Aeros Engine 1.38.1 sources, exactly as
  released (one exception below), including the D3Q19 LBM solver, the CAD/CAE shell,
  the voxeliser, the forces module and the 21 headless test suites.
- **`applications/utilities/aerosGPU/`** — an OpenCL accelerator for OpenFOAM's
  `lduMatrix`, registered as two drop-in solvers, `aerosPCG` and `aerosPBiCGStab`.

`libs/glm/doc` was removed: it is 23 MB of generated Doxygen output that nothing
includes. Everything else is byte-for-byte the release archive.

## What is verified here, and what is not

Being explicit about this matters more than the feature list, because the two halves
have very different levels of evidence behind them.

**`aerosGPU` — verified against a real OpenFOAM build.** `libOpenFOAM.so` and
`libfiniteVolume.so` were compiled from `OpenFOAM-dev-20261113` in this environment,
`libaerosGpuSolvers.so` was linked against them, and
`applications/utilities/aerosGPU/tests/integration` drives it through a real
`lduMatrix`: solvers are selected **by name from a dictionary**, exactly as
`fvSolution` does, the residual is recomputed from `A*x-b` instead of trusted from the
solver's report, and the answer is compared against OpenFOAM's own `PCG`. 14 checks,
0 failures. `aerosGpuInfo` also runs against that build.

**What that does not prove: there was no GPU.** This machine has no OpenCL runtime at
all, so every solve above ran on the CPU reference path and says so in its own log
line. The OpenCL kernels compile-check and the dispatcher's fallback is exercised, but
no kernel has executed on a device. Treat the GPU numbers as unmeasured until
`aerosGpuInfo` reports a device on a machine that has one.

**The Aeros side is imported, not built.** Building it needs GLFW, glad and Vulkan
plus a desktop session; the headless suites need none of that, but the engine itself
was not compiled here. Nothing was changed in `aeros/src`.

**FluidX3D is not integrated.** `docs/` and `CHANGELOG.md` in the Aeros release state
plainly that its LBM solver was written from scratch and that *"third-party solvers,
including FluidX3D, were not used"*. Integrating it anyway would replace a solver with
measured verification (Poiseuille, Rayleigh–Bénard, grid convergence, thread-count
independence) by one that cannot be checked on this machine at all: FluidX3D compiles
its kernels at runtime and refuses to start without an OpenCL device. Substituting it
blind would trade evidence for a claim. The intended shape of such an integration, if
it is wanted, is written up below.

## Building `aerosGPU` (OpenFOAM side)

```sh
source /path/to/OpenFOAM-dev/etc/bashrc
cd applications/utilities/aerosGPU
wmake libso                    # -> $FOAM_USER_LIBBIN/libaerosGpuSolvers.so
wmake                          # -> $FOAM_USER_APPBIN/aerosGpuInfo
./tests/integration            # build + run the lduMatrix checks
```

Then in a case's `fvSolution`:

```c++
solvers
{
    p { solver aerosPCG;        preconditioner DIC;  tolerance 1e-9; relTol 0; }
    U { solver aerosPBiCGStab;  preconditioner DILU; tolerance 1e-9; relTol 0; }
}

aerosGPU
{
    gpuDevice   -2;       // -2 auto, -1 CPU only, >= 0 a device index from aerosGpuInfo
    gpuMinCells 20000;
    gpuVerbose  true;
}
```

A complete example is `applications/utilities/aerosGPU/tests/fvSolution`.

Two build notes that cost time to discover, both now fixed in `Make/options`:

- `-fopenmp` must sit in **`EXE_INC`**, not in `PROJECT_CFLAGS`. `PROJECT_CFLAGS` no
  longer exists in OpenFOAM-dev's `wmake` and is silently ignored, so every `#pragma
  omp` compiled with `-Wunknown-pragmas` and the reference path ran single-threaded
  while still producing correct answers.
- No `-lOpenCL`. The accelerator resolves OpenCL with `dlopen`, so the library loads
  and runs on machines without a driver — which is exactly where the CPU fallback
  matters.

## Running the Aeros headless suites (no GPU, no GLFW, no Vulkan)

```sh
cd aeros/tests
make            # all 21 suites
make lbm        # one suite
EXTRA_CXXFLAGS="-fsanitize=address,undefined -g" ./run-suites.sh
```

Measured in this environment: `make lbm` → 98 checks, 0 failures.

## Standalone aerosGPU harness

The kernels are also testable without OpenFOAM at all, which is how they were
developed:

```sh
make          # build/aerosgpu_core — 24 checks
make asan     # the same under ASan + UBSan
```

Latest result: 24 checks, 0 failures, both plain and under sanitizers. PCG converges in
19 iterations to a residual of 5.585e-13 (dense reference 1.93e-12); PBiCGStab in 10
iterations to 2.824e-14 (dense 1.82e-13).

## If the FluidX3D backend is wanted after all

The honest version of that job, in order, and none of it can be finished here:

1. A build with an OpenCL device attached, so the backend can be measured rather than
   assumed. Without that there is no way to tell a working kernel from a plausible one.
2. A bridge layer of our own in `aeros/src/`, mapping voxelised geometry onto
   FluidX3D's `flags` and reading `rho`/`u` back through its host buffers. FluidX3D's
   cells carry 93 bytes (55 on CPU/iGPU), against Aeros's 19 `float`s per cell plus
   fields, so this is a real data-layout question, not a swap.
3. A UI choice next to the existing CUDA/CPU switch, reported through the same
   diagnostics Aeros already prints: which backend ran, why, and what it cost.
4. The existing `make lbm` / `make physref` / `make thermal` suites re-run against the
   new backend. If they cannot pass, the backend is not ready — those suites are what
   distinguishes this solver from a demo.
