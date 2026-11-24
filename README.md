# Aeros Engine 1.38.1 + aerosGPU

This repository holds two things that are deliberately kept apart:

- **`aeros/`, `libs/`, `docs/`, …** — the Aeros Engine 1.38.1 sources, including the
  D3Q19 LBM solver, the CAD/CAE shell, the voxeliser, the forces module and the headless
  test suites. Two things were added to this tree rather than shipped with it:
  `aeros/src/fx3d_backend.{h,cpp}` (a FluidX3D bridge, described below) and
  `aeros/tests/fx3d_headless.cpp`. Nothing else in `aeros/src` was touched.
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

**The Aeros engine itself is imported, not built.** Building it needs GLFW, glad and
Vulkan plus a desktop session; the headless suites need none of that. Of those suites,
`make lbm` (98 checks) and `make fx3d` (36 checks) were run here; the rest were not.

**FluidX3D: a bridge exists, and it is honestly labelled.** The user asked for a
FluidX3D backend for personal, non-commercial use, which is what FluidX3D 3.9's licence
permits outright ("public research, education or personal use").

`aeros/src/fx3d_backend.{h,cpp}` is that bridge. It maps Aeros geometry onto FluidX3D's
boundary flags, supplies the initial flow, runs steps and reads `rho`/`u` back. It is
built in two halves on purpose:

- **Testable everywhere** (`make fx3d`, 36 checks, 0 failures): index layout, voxels to
  `TYPE_S`/`TYPE_E`, inlet placement and direction, `TYPE_F` never written by hand,
  memory estimate, and the refusal paths. This half is verified without any device
  because the mapping is what fails silently — mistranslate it and you get a plausible
  field that is simply not yours.
- **Compile-verified against FluidX3D itself:** the file compiles against 3.9's real
  headers, so every API call and the `static_assert`s on the flag values are checked
  against the source rather than recalled.

**What is still not proven: no kernel ran.** There is no OpenCL device in this
environment, so `begin()` correctly refuses and the device path is unexecuted. That is
also why the bridge is not yet wired into `lbm.cpp` and the UI: switching the engine's
solver to a path that cannot be run would remove the measured verification the Aeros
solver already has, without replacing it with anything measurable. The switch belongs
on a machine with a GPU.

Two findings worth keeping, both from running the thing rather than reading it:

- `get_devices()` in FluidX3D `opencl.hpp` ends in **`exit(1)`** when no device exists —
  not an exception. So `try`/`catch` around it is useless, and a probe built the naive
  way printed FluidX3D's install-your-drivers wall and died with code 1. `available()`
  now probes with `dlopen` + `clGetPlatformIDs` first and `begin()`/`describeDevice()`
  gate on it. Measured after the fix: both return false with a reason, process exits 0.
- The index orders are *not* different. Aeros stores `(z*Ny+y)*Nx+x`, FluidX3D stores
  `x + Nx*(y + Ny*z)` — the same formula. My first test asserted they differ and
  failed; the test was wrong, not the code. What does differ is field layout
  (`[component*N + n]` versus separate arrays).

Licence note for the record, since it constrains distribution rather than use: Aeros is
GPL-3.0 and FluidX3D forbids commercial and military use and requires modified source to
be published. Personal use is fine; shipping the combination as a product is not.

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
make            # every suite
make lbm        # one suite
make fx3d       # the FluidX3D bridge, no device needed
EXTRA_CXXFLAGS="-fsanitize=address,undefined -g" ./run-suites.sh
```

Measured in this environment: `make lbm` → 98 checks, 0 failures;
`make fx3d` → 36 checks, 0 failures.

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

## Wiring the FluidX3D backend in, on a machine that has a device

The bridge is finished and tested; what remains is the switch, and each step needs a
GPU to be worth doing:

1. `aerosFx3d::Backend` must be able to `begin()` and survive a real run. Until a
   device is present this is unmeasured, and wiring it in would be untested code on
   the physics path.
2. Map the engine's live geometry: `g_voxelData`/`g_voxNx..` into
   `aerosFx3d::mapGeometry` every time the voxel grid is rebuilt, not once at start.
3. Read `rho`/`u` back into `lbmRho`/`lbmUx..lbmUz` after each step so forces,
   streamlines and the pressure map keep working unchanged.
4. Re-run `make lbm`, `make physref`, `make thermal` against the new backend. Those
   suites are what distinguish this solver from a demo; if they cannot pass on
   FluidX3D, the backend is not ready.

