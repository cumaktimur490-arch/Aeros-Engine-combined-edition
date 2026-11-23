/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | OpenCL C source for the sparse solvers
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef aerosGpuKernels_H
#define aerosGpuKernels_H

namespace Foam
{
namespace aerosGpu
{

// The kernel source is a string, not a .cl file with a build rule: the library is
// shipped as a .so into an OpenFOAM install, where a data file next to it would have
// to be found at run time. One string, built when the device is opened, cannot get
// separated from the binary.
//
// Double precision throughout. The device is only used when cl_khr_fp64 is present
// (see suitable() in aerosGpuDevice.C): an FP32 solver cannot hold a 1e-9 residual,
// so there is no FP32 path here by design.
const char* kernelSource();

namespace kernelNames
{
    extern const char* amul;
    extern const char* residual;
    extern const char* dot;
    extern const char* axpy;    // y += alpha*x
    extern const char* xpay;    // x += alpha*y
    extern const char* copy;
    extern const char* scale;   // x *= alpha
    extern const char* dicFactorise;
    extern const char* dicSolve;
    extern const char* diluFactorise;
    extern const char* diluSolve;
}

} // namespace aerosGpu
} // namespace Foam

#endif
