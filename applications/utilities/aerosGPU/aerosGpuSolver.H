/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | PCG and PBiCGStab with IC(0)/ILU(0), CPU reference
    \\  /    A nd           | and device implementations
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

// The arithmetic on the device mirrors the CPU reference here deliberately, down to
// the reduction order of each dot product. Two implementations that "should" agree
// and do not are the usual failure of a port like this, so the CPU path is also what
// the device result is checked against before it is trusted.

#ifndef aerosGpuSolver_H
#define aerosGpuSolver_H

#include "aerosGpuMatrix.H"

#include <string>
#include <vector>

namespace Foam
{
namespace aerosGpu
{

// Why a solve did not finish, in words a user can act on.
enum Status
{
    CONVERGED,
    MAX_ITER,       // residual stayed above tolerance: report it, do not hide it
    DIVERGED,       // residual grew: the matrix or the preconditioner is wrong
    DEVICE_FAILED   // device gave up mid-solve; the caller must fall back to CPU
};


struct SolverControls
{
    int    maxIter;
    double tolerance;   // relative to the source norm
    double minIter;
    bool   verbose;
    bool   useDevice;

    SolverControls()
    :
        maxIter(1000),
        tolerance(1e-9),
        minIter(0),
        verbose(false),
        useDevice(true)
    {}
};


struct SolverResult
{
    Status status;
    int    iterations;
    double residual;
    std::string note;

    SolverResult() : status(MAX_ITER), iterations(0), residual(0) {}

    bool converged() const { return status == CONVERGED; }
};


// Entry points: device when controls.useDevice and a device passed every check, the
// CPU implementation otherwise — including mid-solve, if the device fails.
SolverResult solveCG
(
    std::vector<double>& psi,
    const std::vector<double>& source,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& lower,
    const Pattern& p,
    const SolverControls& controls
);

SolverResult solveBiCGStab
(
    std::vector<double>& psi,
    const std::vector<double>& source,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& lower,
    const Pattern& p,
    const SolverControls& controls
);


// CPU implementations, exposed because they are the reference the device is checked
// against and the tests exercise them directly.
SolverResult solveCGCPU
(
    std::vector<double>& psi,
    const std::vector<double>& source,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& lower,
    const Pattern& p,
    const SolverControls& controls
);

SolverResult solveBiCGStabCPU
(
    std::vector<double>& psi,
    const std::vector<double>& source,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& lower,
    const Pattern& p,
    const SolverControls& controls
);

// Device-only variants: return DEVICE_FAILED with the reason in note when the device
// is not usable, so the dispatcher can fall back.
SolverResult solveCGDevice
(
    std::vector<double>& psi,
    const std::vector<double>& source,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& lower,
    const Pattern& p,
    const SolverControls& controls
);

SolverResult solveBiCGStabDevice
(
    std::vector<double>& psi,
    const std::vector<double>& source,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& lower,
    const Pattern& p,
    const SolverControls& controls
);


// Device state shared by both solvers: opened once, disabled permanently after the
// first failure (retrying a device that produced a wrong answer would cost more than
// the accelerator ever saves, and would hide the failure).
bool deviceReady(std::string& why);
void disableDevice(const std::string& why);
bool deviceDisabled();
const std::string& deviceDisabledReason();

} // namespace aerosGpu
} // namespace Foam

#endif
