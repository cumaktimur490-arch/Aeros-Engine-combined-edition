# aerosGPU — OpenCL linear solvers for OpenFOAM's `lduMatrix`

Two drop-in replacements for OpenFOAM's own solvers, with the same discretisation and
the same answers, computed on a GPU when there is a usable one:

| `solver` in `fvSolution` | replaces     | preconditioner |
|--------------------------|--------------|----------------|
| `aerosPCG`               | `PCG`        | IC(0) (DIC)    |
| `aerosPBiCGStab`         | `PBiCGStab`  | ILU(0) (DILU)  |

plus `aerosGpuInfo`, which prints what the accelerator can use on this machine and why.

## Why this exists, and what the design is defending against

The device part is the obvious motivation. The rest is about not turning an
accelerator into a source of wrong answers:

* **Both triangles of the row structure are always built.** A face joins cells `i` and
  `j`, and row `i` needs the coefficient it owns *and* the one that belongs to row
  `j`. Building only "the upper triangle" because the values are symmetric makes a
  constant field have a non-zero interior row sum. That was a real bug here; it is now
  a test.
* **The pattern is cached, the coefficients are not.** The row structure depends only
  on the mesh, so it is built once (a counting sort) and reused, while the
  coefficients are gathered from `lduMatrix` on every solve. A cache keyed on
  cell/face counts alone would be reused by a re-meshed case of the same size — so the
  addressing is compared, and `generation()` bumps when it is rebuilt.
* **The first device product is checked against the CPU.** One SpMV; a relative
  disagreement beyond 1e-9 disables the device for the rest of the run with the reason
  printed. A device that disagrees once will disagree again, and a converged-looking
  field computed from a transposed matrix is the worst possible outcome.
* **The preconditioners follow OpenFOAM's face-order sweep.** `upperAddr[f] >
  lowerAddr[f]`, so when face `f` is processed the lower cell's diagonal is final. A
  row-order rewrite is a *different* factorisation — it converges differently, and the
  difference looks like a numerical quirk rather than a bug.
* **`PBiCGStab` uses OpenFOAM's recurrence, not the textbook one.** `omega =
  (tA.sA)/(tA.tA)` with `tA = A*(M⁻¹sA)`, and `zA = M⁻¹sA` is what multiplies `omega`
  in the update; `alpha` uses the preconditioned direction `yA = M⁻¹pA`. The common
  variant that preconditions only `r` diverges on real matrices and is fine on toy ones.
* **Reductions are deterministic** (fixed tree, group partials summed in order), so CPU
  and device residuals are comparable instead of differing in the last bits.
* **Every failure falls back downward, loudly**: no OpenCL loader, no fp64, a CPU
  OpenCL runtime, too small a mesh, a failed build, an OOM buffer — all end on the CPU
  path with a one-line reason.

## Building

```bash
cd applications/utilities/aerosGPU
wmake libso     # -> $FOAM_USER_LIBBIN/libaerosGpuSolvers.so
wmake           # -> $FOAM_USER_APPBIN/aerosGpuInfo
```

Nothing links `libOpenCL`: the loader is resolved with `dlopen`, so the library builds
and loads on a machine without a driver (where it then uses the CPU path).
`-fopenmp` is in `PROJECT_CFLAGS` because `wmake` has no per-target flag variable and
the CPU reference is threaded.

## Using

`solver aerosPCG;` (or `aerosPBiCGStab`) in `fvSolution`, plus one optional
sub-dictionary — see `tests/fvSolution`:

```
aerosGPU
{
    gpuDevice   -2;      // -2 auto, -1 CPU only, >= 0 device index
    gpuMinCells 20000;   // below this the transfer cost dominates
    gpuVerbose  true;    // device table + why the device was not used
}
```

Run `aerosGpuInfo` first: it prints the device table with the rejection reason for each
unusable device, and the dispatcher's verdict.

## Verification

`tests/aerosgpu_core.cpp` (buildable standalone, no OpenFOAM needed — see the root
`Makefile`) checks the parts that can be checked without a GPU:

```
make aerosgpu      # 24 checks
```

* both row directions stored per face, every face exactly once per triangle;
* the cache rebuilds for a re-meshed case of equal size;
* a constant field has zero interior row sum (the dropped-triangle test);
* the product matches a hand-built **dense asymmetric** reference matrix;
* PCG and PBiCGStab recover a known solution to ~1e-13, and the dense residual of the
  answer is ~1e-13;
* an unreachable tolerance is reported as `MAX_ITER`, not as convergence;
* a singular system produces no NaN;
* on a machine without OpenCL, the device path reports the reason verbatim and the
  dispatcher still converges on the CPU.

`aerosGpuFoam.C` (the OpenFOAM glue) is syntax-checked against OpenFOAM-dev headers;
it cannot be linked or run here because no OpenFOAM installation or `lnInclude` tree
exists in this environment.

## Limits, stated rather than discovered

* Single domain. Processor and cyclic interface coefficients are not applied here; the
  outer solver loop handles coupling as it does for any `lduMatrix` solver.
* The preconditioner sweeps run as one work-item, in face order. That order is the
  algorithm; level scheduling would parallelise them and is a separate step.
* Float64 only. A device without `cl_khr_fp64` is rejected — an FP32 PCG cannot hold a
  1e-9 residual, so there is no FP32 path by design.
* 32-bit device indices, so meshes up to 2³¹−1 cells; the conversion from OpenFOAM's
  64-bit `label` is cached per mesh and checked rather than truncated.
* `smoothSolver`, `GAMG` and the coupled solvers are untouched.
