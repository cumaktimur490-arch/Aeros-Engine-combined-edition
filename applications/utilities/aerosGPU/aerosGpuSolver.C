/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | PCG and PBiCGStab, CPU reference implementation
    \\  /    A nd           | and the dispatcher
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "aerosGpuSolver.H"

#include <algorithm>
#include <cmath>

namespace Foam
{
namespace aerosGpu
{

namespace
{
    bool devDisabled = false;
    std::string devDisabledWhy;
}


// * * * * * * * * * * * * dot products * * * * * * * * * * * * * * * * * * * //

// A plain sequential sum, deliberately.
//
// A parallel reduction changes the order of additions, so the last bits of every dot
// product differ between thread counts and between CPU and device. For a converged
// solution that is harmless; for a convergence check against a tolerance it is not —
// it makes residuals irreproducible and a device that disagrees with the CPU
// impossible to diagnose. The device kernel sums with a fixed tree for the same
// reason, so the two are comparable.
static double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    const size_t n = a.size();
    double s = 0;
    for (size_t i = 0; i < n; ++i)
    {
        s += a[i]*b[i];
    }
    return s;
}


static double norm2(const std::vector<double>& a)
{
    return std::sqrt(dot(a, a));
}


// * * * * * * * * * * * preconditioners * * * * * * * * * * * * * * * * * * * //

// DIC and DILU follow OpenFOAM's own implementations, including the point that is
// easy to get wrong when porting them to a row-based matrix:
//
//   OpenFOAM sweeps FACES in index order and updates only the upper cell's diagonal.
//   Because upperAddr[f] > lowerAddr[f], that is a valid triangular factorisation:
//   by the time face f is used, the diagonal of its lower cell has received every
//   contribution it will ever get. A row-based rewrite must reproduce that order;
//   "process row r, then row r+1" is a different factorisation and converges
//   differently.
//
// Neither preconditioner stores its factors. An incomplete factorisation of an
// lduMatrix keeps only the diagonal; the off-diagonal factors are the matrix
// coefficients themselves divided by it, which is what makes IC(0)/ILU(0) cheap.
//
// rD holds the diagonal on the way in and its reciprocal on the way out (OpenFOAM
// stores 1/D and multiplies, removing a division from the inner sweep).

static void dicFactorise
(
    std::vector<double>& rD,
    const Pattern& p,
    const std::vector<double>& upper
)
{
    const Triangle& t = p.upperRow;

    for (int k = 0; k < t.nnz(); ++k)
    {
        const int row = t.faceOwnerRow(k);
        const double u = upper[size_t(t.face[size_t(k)])];
        rD[size_t(row)] -= u*u/rD[size_t(t.col[size_t(k)])];
    }

    for (int row = 0; row < p.nCells; ++row)
    {
        // A non-positive pivot cannot be factorised. Clamping instead of dividing by
        // it keeps the run alive and turns the problem into slow convergence, which
        // the caller reports as MAX_ITER — diagnosable, instead of NaNs spreading
        // through the field.
        const double d = rD[size_t(row)];
        rD[size_t(row)] = 1.0/(d > 1e-300 ? d : 1e-300);
    }
}


static void dicSolve
(
    std::vector<double>& x,
    const std::vector<double>& rD,
    const Pattern& p,
    const std::vector<double>& upper
)
{
    const Triangle& t = p.upperRow;

    for (int row = 0; row < p.nCells; ++row)
    {
        x[size_t(row)] *= rD[size_t(row)];
    }

    // Forward, in face order: the entry updated is the face's upper cell.
    for (int k = 0; k < t.nnz(); ++k)
    {
        const int row = t.faceOwnerRow(k);
        x[size_t(row)] -=
            rD[size_t(row)]*upper[size_t(t.face[size_t(k)])]
          * x[size_t(t.col[size_t(k)])];
    }

    // Backward: the face's lower cell, reverse face order.
    for (int k = t.nnz() - 1; k >= 0; --k)
    {
        const int col = t.col[size_t(k)];
        x[size_t(col)] -=
            rD[size_t(col)]*upper[size_t(t.face[size_t(k)])]
          * x[size_t(t.faceOwnerRow(k))];
    }
}


static void diluFactorise
(
    std::vector<double>& rD,
    const Pattern& p,
    const std::vector<double>& upper,
    const std::vector<double>& lower
)
{
    const Triangle& t = p.upperRow;

    for (int k = 0; k < t.nnz(); ++k)
    {
        const size_t f = size_t(t.face[size_t(k)]);
        const int row = t.faceOwnerRow(k);
        rD[size_t(row)] -= upper[f]*lower[f]/rD[size_t(t.col[size_t(k)])];
    }

    for (int row = 0; row < p.nCells; ++row)
    {
        const double d = rD[size_t(row)];
        rD[size_t(row)] =
            1.0/(std::fabs(d) > 1e-300 ? d : (d < 0 ? -1e-300 : 1e-300));
    }
}


static void diluSolve
(
    std::vector<double>& x,
    const std::vector<double>& rD,
    const Pattern& p,
    const std::vector<double>& upper,
    const std::vector<double>& lower
)
{
    const Triangle& t = p.upperRow;

    for (int row = 0; row < p.nCells; ++row)
    {
        x[size_t(row)] *= rD[size_t(row)];
    }

    for (int k = 0; k < t.nnz(); ++k)
    {
        const size_t f = size_t(t.face[size_t(k)]);
        const int row = t.faceOwnerRow(k);
        x[size_t(row)] -= rD[size_t(row)]*lower[f]*x[size_t(t.col[size_t(k)])];
    }

    for (int k = t.nnz() - 1; k >= 0; --k)
    {
        const size_t f = size_t(t.face[size_t(k)]);
        const int col = t.col[size_t(k)];
        x[size_t(col)] -= rD[size_t(col)]*upper[f]*x[size_t(t.faceOwnerRow(k))];
    }
}


// * * * * * * * * * * * * normalisation * * * * * * * * * * * * * * * * * * * //

// OpenFOAM normalises the residual by the source norm on a fresh solve; a zero source
// (psi already exact) must terminate rather than divide by zero.
static double normaliser(const std::vector<double>& source)
{
    const double n = norm2(source);
    return (n > 0 ? n : 1.0);
}


// * * * * * * * * * * * * PCG * * * * * * * * * * * * * * * * * * * * * * * //

SolverResult solveCGCPU
(
    std::vector<double>& psi,
    const std::vector<double>& source,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& lower,
    const Pattern& p,
    const SolverControls& controls
)
{
    SolverResult result;
    const int n = p.nCells;
    if (n == 0)
    {
        result.status = CONVERGED;
        return result;
    }

    const double normFactor = normaliser(source);

    std::vector<double> rD(diag.begin(), diag.begin() + n);
    dicFactorise(rD, p, upper);

    std::vector<double> r(n), w(n), pVec(n), z(n);

    amul(w.data(), psi.data(), diag.data(), upper.data(), lower.data(), p);
    for (int i = 0; i < n; ++i)
    {
        r[size_t(i)] = source[size_t(i)] - w[size_t(i)];
    }

    // psi = 0 is the usual initial guess, but a case re-solving an existing field is
    // not — so the residual above is the real one, not `source`.
    result.residual = norm2(r)/normFactor;
    const double initialResidual = result.residual;

    // A zero right-hand side is already the solution. Without this the loop enters
    // with r == p == 0, the first dot product is exactly 0 and the solve is reported as
    // "zero curvature (singular matrix)", which is the wrong reason for the right
    // answer. Measured on an lduMatrix: 1 iteration and a false breakdown message.
    if (!(initialResidual > controls.tolerance))
    {
        result.status = CONVERGED;
        return result;
    }

    if (controls.maxIter <= 0)
    {
        result.status = MAX_ITER;
        return result;
    }

    z = r;
    dicSolve(z, rD, p, upper);
    pVec = z;
    double rz = dot(r, z);

    for (int iter = 1; iter <= controls.maxIter; ++iter)
    {
        amul(w.data(), pVec.data(), diag.data(), upper.data(), lower.data(), p);
        const double pw = dot(pVec, w);

        if (!(std::fabs(pw) > 0))
        {
            // Zero curvature: the search direction has no component in the matrix
            // range, i.e. a singular system (usually a floating region).
            result.status = DIVERGED;
            result.iterations = iter;
            result.note = "CG: zero curvature (singular matrix or disconnected region)";
            return result;
        }

        const double alpha = rz/pw;

        for (int i = 0; i < n; ++i)
        {
            psi[size_t(i)] += alpha*pVec[size_t(i)];
            r[size_t(i)] -= alpha*w[size_t(i)];
        }

        result.residual = norm2(r)/normFactor;
        result.iterations = iter;

        if (result.residual < controls.tolerance && iter >= controls.minIter)
        {
            result.status = CONVERGED;
            return result;
        }

        if (!std::isfinite(result.residual))
        {
            result.status = DIVERGED;
            result.note = "CG: residual is not finite";
            return result;
        }
        if (result.residual > 1e6*std::max(initialResidual, 1e-300))
        {
            result.status = DIVERGED;
            result.note = "CG: residual growing (matrix not symmetric?)";
            return result;
        }

        z = r;
        dicSolve(z, rD, p, upper);

        const double rzNew = dot(r, z);
        const double beta = rzNew/rz;
        rz = rzNew;

        for (int i = 0; i < n; ++i)
        {
            pVec[size_t(i)] = z[size_t(i)] + beta*pVec[size_t(i)];
        }
    }

    result.status = MAX_ITER;
    return result;
}


// * * * * * * * * * * * * PBiCGStab * * * * * * * * * * * * * * * * * * * * //

SolverResult solveBiCGStabCPU
(
    std::vector<double>& psi,
    const std::vector<double>& source,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& lower,
    const Pattern& p,
    const SolverControls& controls
)
{
    // Follows lduMatrix::solver::PBiCGStab term for term, because the details are the
    // algorithm:
    //   * rA0 is the initial residual and is never updated (two-sided method);
    //   * alpha uses the preconditioned direction: yA = M^-1 pA, AyA = A*yA;
    //   * omega = (tA.sA)/(tA.tA) with tA = A*(M^-1 sA) — not (zA.tA)/(zA.zA) —
    //     and zA, not tA, multiplies omega in the solution update;
    //   * beta = (rA0rA/rA0rAold)*(alpha/omega).
    // Each of those is cheap to get subtly wrong, and each wrong choice gives a
    // method that still converges on easy matrices.
    SolverResult result;
    const int n = p.nCells;
    if (n == 0)
    {
        result.status = CONVERGED;
        return result;
    }

    const double normFactor = normaliser(source);

    std::vector<double> rD(diag.begin(), diag.begin() + n);
    diluFactorise(rD, p, upper, lower);

    std::vector<double> w(n), r(n), rT(n), y(n), Ay(n), pVec(n), s(n), z(n), t(n);

    amul(w.data(), psi.data(), diag.data(), upper.data(), lower.data(), p);
    for (int i = 0; i < n; ++i)
    {
        r[size_t(i)] = source[size_t(i)] - w[size_t(i)];
    }
    rT = r;
    result.residual = norm2(r)/normFactor;
    const double initialResidual = result.residual;

    if (controls.maxIter <= 0)
    {
        result.status = MAX_ITER;
        return result;
    }

    double rT0r = 0;
    double alpha = 0;
    double omega = 0;

    for (int iter = 1; iter <= controls.maxIter; ++iter)
    {
        const double rT0rOld = rT0r;
        rT0r = dot(rT, r);

        if (!(std::fabs(rT0r) > 0))
        {
            result.status = DIVERGED;
            result.iterations = iter - 1;
            result.note = "PBiCGStab: breakdown (rA0.rA == 0)";
            return result;
        }

        if (iter == 1)
        {
            pVec = r;
        }
        else
        {
            if (!(std::fabs(omega) > 0))
            {
                result.status = DIVERGED;
                result.iterations = iter - 1;
                result.note = "PBiCGStab: breakdown (omega == 0)";
                return result;
            }
            const double beta = (rT0r/rT0rOld)*(alpha/omega);
            for (int i = 0; i < n; ++i)
            {
                pVec[size_t(i)] = r[size_t(i)]
                    + beta*(pVec[size_t(i)] - omega*Ay[size_t(i)]);
            }
        }

        y = pVec;
        diluSolve(y, rD, p, upper, lower);
        amul(Ay.data(), y.data(), diag.data(), upper.data(), lower.data(), p);

        const double rT0Ay = dot(rT, Ay);
        if (!(std::fabs(rT0Ay) > 0))
        {
            result.status = DIVERGED;
            result.iterations = iter - 1;
            result.note = "PBiCGStab: breakdown (rA0.A.yA == 0)";
            return result;
        }
        alpha = rT0r/rT0Ay;

        for (int i = 0; i < n; ++i)
        {
            s[size_t(i)] = r[size_t(i)] - alpha*Ay[size_t(i)];
        }

        const double sNorm = norm2(s)/normFactor;
        result.residual = sNorm;
        result.iterations = iter;

        if (sNorm < controls.tolerance && iter >= controls.minIter)
        {
            for (int i = 0; i < n; ++i)
            {
                psi[size_t(i)] += alpha*y[size_t(i)];
            }
            result.status = CONVERGED;
            return result;
        }

        z = s;
        diluSolve(z, rD, p, upper, lower);
        amul(t.data(), z.data(), diag.data(), upper.data(), lower.data(), p);

        const double tAtA = dot(t, t);
        const double tAsA = dot(t, s);
        omega = (std::fabs(tAtA) > 0 ? tAsA/tAtA : 0.0);

        for (int i = 0; i < n; ++i)
        {
            psi[size_t(i)] += alpha*y[size_t(i)] + omega*z[size_t(i)];
            r[size_t(i)] = s[size_t(i)] - omega*t[size_t(i)];
        }

        result.residual = norm2(r)/normFactor;

        if (!std::isfinite(result.residual))
        {
            result.status = DIVERGED;
            result.note = "PBiCGStab: residual is not finite";
            return result;
        }
        if (result.residual > 1e6*std::max(initialResidual, 1e-300))
        {
            result.status = DIVERGED;
            result.note = "PBiCGStab: residual growing";
            return result;
        }
    }

    result.status = MAX_ITER;
    return result;
}


// * * * * * * * * * * * * * dispatch * * * * * * * * * * * * * * * * * * * * //

// Device first, CPU as the safety net. The fallback is not a courtesy: a case that
// runs for hours must not be lost because a driver returned an error on iteration
// 700, and the caller is told which of the two answers it got.
SolverResult solveCG
(
    std::vector<double>& psi,
    const std::vector<double>& source,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& lower,
    const Pattern& p,
    const SolverControls& controls
)
{
    if (controls.useDevice && !deviceDisabled())
    {
        std::vector<double> trial(psi);
        const SolverResult dres =
            solveCGDevice(trial, source, diag, upper, lower, p, controls);

        if (dres.status != DEVICE_FAILED)
        {
            psi.swap(trial);
            return dres;
        }
    }

    return solveCGCPU(psi, source, diag, upper, lower, p, controls);
}


SolverResult solveBiCGStab
(
    std::vector<double>& psi,
    const std::vector<double>& source,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& lower,
    const Pattern& p,
    const SolverControls& controls
)
{
    if (controls.useDevice && !deviceDisabled())
    {
        std::vector<double> trial(psi);
        const SolverResult dres =
            solveBiCGStabDevice(trial, source, diag, upper, lower, p, controls);

        if (dres.status != DEVICE_FAILED)
        {
            psi.swap(trial);
            return dres;
        }
    }

    return solveBiCGStabCPU(psi, source, diag, upper, lower, p, controls);
}


// * * * * * * * * * * * * device state * * * * * * * * * * * * * * * * * * * //

void disableDevice(const std::string& why)
{
    devDisabled = true;
    devDisabledWhy = why;
}


bool deviceDisabled()
{
    return devDisabled;
}


const std::string& deviceDisabledReason()
{
    return devDisabledWhy;
}

} // namespace aerosGpu
} // namespace Foam
