/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | lduMatrix: aerosPCG / aerosPBiCGStab checked
    \\  /    A nd           | against OpenFOAM's own solvers on a real matrix
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

// What this checks, and why it has to exist.
//
// The standalone harness in tests/aerosgpu_core.cpp exercises the kernels on its own
// CSR matrices. That proves the arithmetic, but it does not prove that the two solver
// classes are *reachable from fvSolution* — the part that only works if the runtime
// selection tables, the dictionary parsing, the interface field lists and the
// solverPerformance bookkeeping all line up with this OpenFOAM's actual types.
//
// So this program builds a real lduMatrix (7-point Laplacian on an nx*ny*nz lattice,
// the same structure an fvMesh produces: face f joins lowerAddr[f] to upperAddr[f], every
// face stored once), then:
//
//   1. selects each solver BY NAME from a dictionary, exactly as fvSolution does, so a
//      missing/broken RTS entry fails here instead of in a user's case;
//   2. verifies the linear system is actually solved: the final residual is recomputed
//      here from A*x-b, not taken from the solver's own report;
//   3. verifies the answer agrees with OpenFOAM's own PCG on the same matrix, which is
//      an independent implementation of the same physics;
//   4. checks the asymmetric path with lower != upper, where a symmetric-only
//      implementation would still "converge" but to the wrong answer;
//   5. checks that a zero right-hand side does not produce NaN.
//
// Everything is printed as measured. Nothing is skipped quietly: if the accelerator
// runs on the CPU, its own note says so and is echoed here.

#include "dictionary.H"
#include "lduMatrix.H"
#include "lduMesh.H"
#include "IOstreams.H"
#include "OSspecific.H"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace Foam;

namespace
{

int gChecks = 0;
int gFailed = 0;

void check(bool ok, const char* what)
{
    std::printf("  %s  %s\n", ok ? "ok:  " : "FAIL:", what);
    ++gChecks;
    if (!ok) ++gFailed;
}

// * * * * * * * * * * * * * * * * mesh * * * * * * * * * * * * * * * * * * * //

// A structured lattice as an lduMesh. The accessors are the whole interface
// lduMatrix needs: the addressing (which two cells each face joins), the
// list of coupled patches (empty — everything is within one domain) and a
// communicator index.
class LatticeAddressing
:
    public lduAddressing
{
    labelList lower_;
    labelList upper_;
    lduSchedule schedule_;

public:

    LatticeAddressing
    (
        const label nCells,
        labelList&& lower,
        labelList&& upper
    )
    :
        lduAddressing(nCells),
        lower_(std::move(lower)),
        upper_(std::move(upper))
    {
        // No coupled patches, so there is no inter-process schedule to evaluate.
        schedule_.setSize(0);
    }

    virtual const labelUList& lowerAddr() const { return lower_; }
    virtual const labelUList& upperAddr() const { return upper_; }

    virtual const labelUList& patchAddr(const label) const
    {
        return labelList::null();
    }

    virtual const lduSchedule& patchSchedule() const { return schedule_; }
};


class LatticeMesh
:
    public lduMesh
{
    LatticeAddressing addressing_;

public:

    LatticeMesh(const label nCells, labelList&& lower, labelList&& upper)
    :
        addressing_(nCells, std::move(lower), std::move(upper))
    {}

    virtual const lduAddressing& lduAddr() const { return addressing_; }

    virtual lduInterfacePtrsList interfaces() const
    {
        return lduInterfacePtrsList(0);
    }

    virtual label comm() const { return 0; }
};


// * * * * * * * * * * * * * * * * helpers * * * * * * * * * * * * * * * * * //

// 7-point Laplacian on an nx*ny*nz lattice, cell index = x + nx*(y + ny*z).
// Off-diagonals are -1; the diagonal is set so the matrix is diagonally dominant
// and positive definite for the symmetric case, exactly like an fvMatrix of a
// diffusion term with a Dirichlet boundary on every side.
struct Lattice
{
    label nx, ny, nz, nCells;
    labelList lower, upper;
    scalarField diag;

    Lattice(const label nx_, const label ny_, const label nz_)
    :
        nx(nx_), ny(ny_), nz(nz_), nCells(nx_*ny_*nz_)
    {
        lower.setSize(3*nCells);
        upper.setSize(3*nCells);
        diag.setSize(nCells, Zero);

        label nFaces = 0;
        for (label z = 0; z < nz; ++z)
        {
            for (label y = 0; y < ny; ++y)
            {
                for (label x = 0; x < nx; ++x)
                {
                    const label c = x + nx*(y + ny*z);

                    // +x, +y, +z neighbours only: every interior face is stored once.
                    if (x + 1 < nx)
                    {
                        lower[nFaces] = c; upper[nFaces] = c + 1; ++nFaces;
                        diag[c] += 2; diag[c + 1] += 2;
                    }
                    if (y + 1 < ny)
                    {
                        lower[nFaces] = c; upper[nFaces] = c + nx; ++nFaces;
                        diag[c] += 2; diag[c + nx] += 2;
                    }
                    if (z + 1 < nz)
                    {
                        lower[nFaces] = c; upper[nFaces] = c + nx*ny; ++nFaces;
                        diag[c] += 2; diag[c + nx*ny] += 2;
                    }
                }
            }
        }

        lower.setSize(nFaces);
        upper.setSize(nFaces);
    }

    // A*x for the assembled system; the reference against which the solver's own
    // claim of convergence is checked.
    scalarField multiply(const scalarField& x, const scalarField& lo, const scalarField& up) const
    {
        scalarField r(nCells, Zero);
        for (label c = 0; c < nCells; ++c)
        {
            r[c] = diag[c]*x[c];
        }
        for (label f = 0; f < upper.size(); ++f)
        {
            const label l = lower[f], u = upper[f];
            r[l] += lo[f]*x[u];
            r[u] += up[f]*x[l];
        }
        return r;
    }

    double residual
    (
        const scalarField& x,
        const scalarField& b,
        const scalarField& lo,
        const scalarField& up,
        double* bNorm
    ) const
    {
        const scalarField Ax = multiply(x, lo, up);
        double s = 0.0, sB = 0.0;
        for (label c = 0; c < nCells; ++c)
        {
            const double d = double(Ax[c]) - double(b[c]);
            s += d*d;
            sB += double(b[c])*double(b[c]);
        }
        if (bNorm) *bNorm = std::sqrt(sB);
        return std::sqrt(s);
    }
};


// Build the dictionary a solver is normally handed by fvSolution, and let OpenFOAM
// pick the implementation by name. This is the path that fails first if the runtime
// selection table was not registered.
autoPtr<lduMatrix::solver> select
(
    const word& solverName,
    const word& preconditioner,
    const lduMatrix& matrix,
    const lduMesh& mesh,
    double tolerance,
    double relTol,
    label maxIter,
    const char* gpu = nullptr
)
{
    const lduInterfaceFieldPtrsList interfaces(mesh.interfaces().size());

    dictionary controls;
    controls.add("solver", solverName);
    controls.add("preconditioner", preconditioner);
    controls.add("tolerance", tolerance);
    controls.add("relTol", relTol);
    controls.add("maxIter", maxIter);

    if (gpu)
    {
        dictionary gpuDict;
        gpuDict.add("gpuDevice", -1);     // this machine has no OpenCL runtime
        gpuDict.add("gpuMinCells", 1);    // so the threshold logic is exercised too
        gpuDict.add("gpuVerbose", true);
        controls.add("aerosGPU", gpuDict);
    }

    return lduMatrix::solver::New
    (
        "p",
        matrix,
        Field<Field<scalar>>(0),
        Field<Field<scalar>>(0),
        interfaces,
        controls
    );
}

} // End anonymous namespace


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char* argv[])
{
    const label nx = 16, ny = 16, nz = 16;

    std::printf("aerosGPU x lduMatrix: %ld x %ld x %ld lattice (%ld cells)\n\n",
                long(nx), long(ny), long(nz), long(nx*ny*nz));

    Lattice lattice(nx, ny, nz);
    LatticeMesh mesh(lattice.nCells, labelList(lattice.lower), labelList(lattice.upper));

    // Symmetric matrix: one coefficient array, upper == lower.
    lduMatrix symMatrix(mesh);
    symMatrix.diag() = lattice.diag;
    symMatrix.upper() = scalarField(lattice.upper.size(), -1.0);
    if (!symMatrix.hasLower())
    {
        // Writing lower() would make the matrix asymmetric; for the symmetric
        // operator OpenFOAM stores the single upper array only.
        std::printf("  (matrix reports symmetric storage as expected)\n");
    }

    // Right-hand side: a smooth field so the solution is not dominated by
    // round-off, and the exact answer is bounded.
    scalarField b(lattice.nCells, Zero);
    for (label c = 0; c < lattice.nCells; ++c)
    {
        b[c] = 1.0 + 0.5*std::sin(0.1*double(c));
    }

    const lduInterfaceFieldPtrsList noInterfaces(0);

    // ---------------------------------------------------------------- PCG ----
    std::printf("\n-- aerosPCG selected by name from the dictionary\n");
    scalarField psi(lattice.nCells, Zero);
    autoPtr<lduMatrix::solver> pcg =
        select("aerosPCG", "DIC", symMatrix, mesh, 1e-10, 0.0, 1000, "cpu");

    check(pcg.valid(), "lduMatrix::solver::New returned a solver for aerosPCG");
    if (!pcg.valid())
    {
        std::printf("\n=== NOT LINKED REACHABLE (checks: %d, failures: %d) ===\n",
                    gChecks, gFailed);
        return 1;
    }

    check
    (
        pcg->type() == "aerosPCG",
        "the registered type name is aerosPCG (a wrong entry would collide with PCG)"
    );

    const solverPerformance pcgPerf = pcg->solve(psi, b);

    double bNorm = 0.0;
    const double pcgRes = lattice.residual(psi, b, symMatrix.upper(), symMatrix.upper(), &bNorm);
    std::printf("  reported: iterations %ld, initial %.3e, final %.3e\n",
                long(pcgPerf.nIterations()),
                double(pcgPerf.initialResidual()),
                double(pcgPerf.finalResidual()));
    std::printf("  recomputed here from A*x-b: %.3e (relative %.3e)\n",
                pcgRes, pcgRes/bNorm);

    // OpenFOAM reports a solve as "<preconditioner><solver>": DICPCG, DILUPBiCGStab.
    // Anything else (e.g. "DICaerosPCG") makes the log line read as a different
    // preconditioner, so the convention is part of the contract.
    std::printf("  reported solver name: %s\n", word(pcgPerf.solverName()).c_str());
    check
    (
        pcgPerf.solverName() == word("DICPCG"),
        "solver name follows OpenFOAM's <preconditioner><solver> convention"
    );

    check(pcgPerf.nIterations() > 0, "the solver actually iterated");
    check
    (
        pcgRes/bNorm < 1e-8,
        "residual recomputed from A*x-b is below 1e-8 relative — the system is solved"
    );
    check
    (
        std::fabs(double(pcgPerf.finalResidual()) - pcgRes/bNorm) < 1e-6,
        "the residual the solver reports matches the one recomputed here"
    );

    // ------------------------------------------------- cross-check with OF ----
    std::printf("\n-- same matrix, OpenFOAM's own PCG (independent implementation)\n");
    scalarField psiRef(lattice.nCells, Zero);
    autoPtr<lduMatrix::solver> ofPcg =
        select("PCG", "DIC", symMatrix, mesh, 1e-10, 0.0, 1000);

    check(ofPcg.valid(), "OpenFOAM's own PCG is reachable too (control case)");
    if (ofPcg.valid())
    {
        const solverPerformance refPerf = ofPcg->solve(psiRef, b);
        std::printf("  reported: iterations %ld, final %.3e\n",
                    long(refPerf.nIterations()), double(refPerf.finalResidual()));

        double maxDiff = 0.0, maxVal = 0.0;
        for (label c = 0; c < lattice.nCells; ++c)
        {
            const double d = std::fabs(double(psi[c]) - double(psiRef[c]));
            if (d > maxDiff) maxDiff = d;
            const double v = std::fabs(double(psiRef[c]));
            if (v > maxVal) maxVal = v;
        }
        std::printf("  max |x_aerosGPU - x_OpenFOAM| = %.3e (solution scale %.3e)\n",
                    maxDiff, maxVal);
        check
        (
            maxDiff <= 1e-7*std::max(1.0, maxVal),
            "the accelerated solution agrees with OpenFOAM's PCG on the same matrix"
        );
    }

    // ------------------------------------------------- asymmetric BiCGStab ----
    std::printf("\n-- aerosPBiCGStab on an asymmetric matrix (lower != upper)\n");

    lduMatrix asymMatrix(mesh);
    asymMatrix.diag() = lattice.diag;
    asymMatrix.upper() = scalarField(lattice.upper.size(), -1.0);
    asymMatrix.lower() = scalarField(lattice.lower.size(), -0.5);

    check(asymMatrix.hasLower(), "matrix now reports asymmetric storage");

    scalarField psiA(lattice.nCells, Zero);
    autoPtr<lduMatrix::solver> bicg =
        select("aerosPBiCGStab", "DILU", asymMatrix, mesh, 1e-10, 0.0, 1000, "cpu");

    check(bicg.valid(), "lduMatrix::solver::New returned a solver for aerosPBiCGStab");
    if (bicg.valid())
    {
        const solverPerformance bicgPerf = bicg->solve(psiA, b);

        double bNormA = 0.0;
        const double bicgRes =
            lattice.residual(psiA, b, asymMatrix.lower(), asymMatrix.upper(), &bNormA);
        std::printf("  reported: iterations %ld, final %.3e\n",
                    long(bicgPerf.nIterations()), double(bicgPerf.finalResidual()));
        std::printf("  recomputed here from A*x-b: %.3e (relative %.3e)\n",
                    bicgRes, bicgRes/bNormA);
        check
        (
            bicgRes/bNormA < 1e-8,
            "asymmetric system solved to 1e-8 relative — the lower array is really used"
        );

        // If lower were ignored (duplicated from upper), x would be the solution of the
        // symmetric system and the residual of the real asymmetric system would sit
        // near 1e-1, not 1e-8. Comparing the two solutions makes that explicit.
        double asymDiff = 0.0;
        for (label c = 0; c < lattice.nCells; ++c)
        {
            asymDiff = std::max(asymDiff, std::fabs(double(psiA[c]) - double(psi[c])));
        }
        std::printf("  max |x_asym - x_sym| = %.3e (the two systems do differ)\n", asymDiff);
        check
        (
            asymDiff > 1e-3,
            "asymmetric solution differs from the symmetric one, so lower[] is in use"
        );
    }

    // ------------------------------------------------------- zero RHS ----
    std::printf("\n-- zero right-hand side (must not produce NaN)\n");
    scalarField zeroB(lattice.nCells, Zero);
    scalarField psiZ(lattice.nCells, Zero);
    const solverPerformance zPerf = pcg->solve(psiZ, zeroB);
    bool finite = true;
    for (label c = 0; c < lattice.nCells; ++c)
    {
        if (!std::isfinite(double(psiZ[c]))) { finite = false; break; }
    }
    std::printf("  iterations %ld, final %.3e\n",
                long(zPerf.nIterations()), double(zPerf.finalResidual()));
    check(finite, "the solution stays finite");
    check
    (
        double(zPerf.finalResidual()) < 1e-9,
        "a zero source is reported as already converged"
    );

    // ------------------------------------------------------------- report ----
    std::printf
    (
        "\n=== %s (checks: %d, failures: %d) ===\n",
        gFailed == 0 ? "LDU INTEGRATION IN ORDER" : "LDU INTEGRATION FAILED",
        gChecks, gFailed
    );

    return gFailed == 0 ? 0 : 1;
}


// ************************************************************************* //
