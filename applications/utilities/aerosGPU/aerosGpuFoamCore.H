/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | The part of the lduMatrix solver wrapper that does
    \\  /    A nd           | not depend on OpenFOAM's types
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef aerosGpuFoamCore_H
#define aerosGpuFoamCore_H

#include <string>
#include <vector>

#include "aerosGpuMatrix.H"
#include "aerosGpuSolver.H"

namespace Foam
{
namespace aerosGpu
{

// Settings from the "aerosGPU" sub-dictionary: -2 automatic, -1 CPU only, >= 0 a
// device index.
struct Settings
{
    int  device;
    int  minCells;
    bool verbose;

    Settings() : device(-2), minCells(20000), verbose(false) {}
};


// The matrix pulled apart into the arrays the accelerator works with. Only the
// diagonal and the two coefficient arrays are needed: boundary conditions reach the
// solver through the source vector, as they do for any lduMatrix solver, and
// processor coupling is resolved by the outer loop.
struct Assembled
{
    std::vector<double> diag;
    std::vector<double> upper;
    std::vector<double> lower;
    std::vector<double> b;
    std::vector<double> x;
    const Pattern* pattern;
};


// `hasLower` decides whether the matrix is asymmetric. A matrix without a lower
// triangle is symmetric by construction (OpenFOAM stores one coefficient array), and
// that array then serves both row directions.
void assemble
(
    const double* diag, int nCells,
    const double* upper, const double* lower,
    const int* upperAddr, const int* lowerAddr, int nFaces,
    bool hasLower,
    const double* source,
    const double* psi,
    Assembled& a
);


// Residual of the assembled system, normalised by the source norm — the same
// normalisation the accelerator uses, so initial and final numbers are comparable.
double initialResidual(const Assembled& a);


// OpenFOAM stops when the residual falls below max(tolerance, relTol*initial), so the
// accelerator has to be given the *looser* of the two to stop where the reference
// solver would. The stricter one would make every run iterate further than OpenFOAM
// would — slower, and different residual histories in a log comparison.
SolverControls controlsFrom
(
    const Settings& s,
    int maxIter,
    int minIter,
    double tolerance,
    double relTol
);


// Summary of a solve, in OpenFOAM's terms.
struct Outcome
{
    bool   converged;
    double initialResidual;
    double finalResidual;
    int    iterations;
    std::string note;

    Outcome() : converged(false), initialResidual(0), finalResidual(0), iterations(0) {}
};


// Copies the answer into psi and decides convergence from the residual actually
// reached, not from the accelerator's own opinion: the two agree when it is right, and
// if they ever do not, the residual is the part a user can check.
Outcome finish
(
    const SolverResult& r,
    double initRes,
    double tolerance,
    double relTol,
    std::vector<double>& psi,
    const Assembled& a
);

} // namespace aerosGpu
} // namespace Foam

#endif
