/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | Shared logic behind the two lduMatrix solvers
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "aerosGpuFoamCore.H"

#include <cmath>

namespace Foam
{
namespace aerosGpu
{

void assemble
(
    const double* diag, int nCells,
    const double* upper, const double* lower,
    const int* upperAddr, const int* lowerAddr, int nFaces,
    bool hasLower,
    const double* source,
    const double* psi,
    Assembled& a
)
{
    // A matrix without a lower triangle is symmetric by construction: one coefficient
    // array, used for both row directions.
    const bool symmetric = !hasLower;

    a.diag.assign(diag, diag + nCells);
    a.upper.assign(upper, upper + nFaces);

    if (hasLower)
    {
        a.lower.assign(lower, lower + nFaces);
    }
    else
    {
        a.lower = a.upper;
    }

    a.b.assign(source, source + nCells);
    a.x.assign(psi, psi + nCells);

    a.pattern = &pattern(upperAddr, lowerAddr, nCells, nFaces, symmetric);
}


double initialResidual(const Assembled& a)
{
    const size_t n = a.diag.size();

    std::vector<double> ax(n, 0.0);
    amul(ax.data(), a.x.data(), a.diag.data(), a.upper.data(), a.lower.data(), *a.pattern);

    double rNorm = 0;
    double bNorm = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const double r = a.b[i] - ax[i];
        rNorm += r*r;
        bNorm += a.b[i]*a.b[i];
    }

    const double b = std::sqrt(bNorm);
    return std::sqrt(rNorm)/(b > 0 ? b : 1.0);
}


SolverControls controlsFrom
(
    const Settings& s,
    int maxIter,
    int minIter,
    double tolerance,
    double relTol
)
{
    SolverControls c;
    c.maxIter = maxIter;
    c.minIter = minIter;
    c.tolerance = (relTol > tolerance ? relTol : tolerance);
    c.useDevice = (s.device >= -1);
    c.verbose = s.verbose;
    return c;
}


Outcome finish
(
    const SolverResult& r,
    double initRes,
    double tolerance,
    double relTol,
    std::vector<double>& psi,
    const Assembled& a
)
{
    for (size_t i = 0; i < psi.size(); ++i)
    {
        psi[i] = a.x[i];
    }

    Outcome o;
    o.initialResidual = initRes;
    o.finalResidual = r.residual;
    o.iterations = r.iterations;
    o.note = r.note;

    // max(tolerance, relTol*initial), the same rule SolverPerformance::checkConvergence
    // applies.
    const double target = (relTol > 0 ? relTol*initRes : 0);
    o.converged = r.residual <= (target > tolerance ? target : tolerance);

    return o;
}

} // namespace aerosGpu
} // namespace Foam
