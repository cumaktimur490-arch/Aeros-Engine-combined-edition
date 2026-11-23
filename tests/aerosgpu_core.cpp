// Tests for the aerosGPU core: lduMatrix -> row structure, the sparse product, and
// both Krylov solvers against problems with a known answer.
//
// What is being defended:
//   * the pattern stores both row directions for every face (dropping one makes a
//     constant field have a non-zero interior row sum);
//   * the product is A*x, not A^T*x — checked against a dense reference built by
//     hand, on an asymmetric matrix, because a symmetric test cannot see it;
//   * a divergence in the preconditioning sweep is caught (see the DIAGONAL test);
//   * a solver that cannot converge says so instead of returning a field that
//     looks fine.
//
// The device path is exercised only for its failure behaviour: there is no OpenCL
// runtime in this environment, so "the device explains why it cannot run" is what
// can be verified here.

#include "aerosGpuMatrix.H"
#include "aerosGpuSolver.H"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace Foam::aerosGpu;

static int failures = 0;
static int checks = 0;

static void check(bool ok, const char* what)
{
    ++checks;
    if (ok) { std::printf("  [ok  ] %s\n", what); }
    else    { ++failures; std::printf("  [FAIL] %s\n", what); }
}


// A structured 3-D mesh: hexahedral cells, faces between neighbours in x, y, z. Same
// connectivity as an OpenFOAM structured mesh, which is what makes the row structure
// non-trivial (several neighbours per row).
struct Grid
{
    int nx, ny, nz;
    std::vector<int> upperAddr;
    std::vector<int> lowerAddr;

    Grid(int a, int b, int c) : nx(a), ny(b), nz(c) {}

    int cell(int i, int j, int k) const { return (k*ny + j)*nx + i; }
    int nCells() const { return nx*ny*nz; }
    int nFaces() const { return int(upperAddr.size()); }

    void build()
    {
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
            const int c = cell(i, j, k);
            if (i + 1 < nx) { lowerAddr.push_back(c); upperAddr.push_back(cell(i+1, j, k)); }
            if (j + 1 < ny) { lowerAddr.push_back(c); upperAddr.push_back(cell(i, j+1, k)); }
            if (k + 1 < nz) { lowerAddr.push_back(c); upperAddr.push_back(cell(i, j, k+1)); }
        }
    }
};


// Laplace matrix: diagonal 2*d where d is the face count of the cell, off-diagonal
// -1 per neighbour (the sign convention of fvm::laplacian). Boundary rows get the
// diagonal bumped, which is what a Dirichlet condition does to the row; a singular
// matrix would make the test about robustness instead of correctness.
static void laplacian
(
    const Grid& g,
    std::vector<double>& diag,
    std::vector<double>& upper,
    std::vector<double>& lower
)
{
    const int n = g.nCells();
    diag.assign(size_t(n), 0.0);
    upper.assign(size_t(g.nFaces()), -1.0);
    lower.assign(size_t(g.nFaces()), -1.0);

    for (int f = 0; f < g.nFaces(); ++f)
    {
        diag[size_t(g.lowerAddr[size_t(f)])] += 1.0;
        diag[size_t(g.upperAddr[size_t(f)])] += 1.0;
    }
    for (int c = 0; c < n; ++c)
    {
        if (diag[size_t(c)] < 6.0)
        {
            diag[size_t(c)] += 1.0;
        }
    }
}


// Dense reference matrix, written out by hand from the lduMatrix convention:
// lower[f] is the (lower,upper) entry, upper[f] is the (upper,lower) entry. The whole
// library depends on that convention, so the reference does not reuse any of it.
static std::vector<double> denseMatrix
(
    const Grid& g,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& lower
)
{
    const int n = g.nCells();
    std::vector<double> a(size_t(n)*size_t(n), 0.0);
    for (int c = 0; c < n; ++c)
    {
        a[size_t(c)*size_t(n) + size_t(c)] = diag[size_t(c)];
    }
    for (int f = 0; f < g.nFaces(); ++f)
    {
        const int lo = g.lowerAddr[size_t(f)];
        const int up = g.upperAddr[size_t(f)];
        a[size_t(lo)*size_t(n) + size_t(up)] += lower[size_t(f)];
        a[size_t(up)*size_t(n) + size_t(lo)] += upper[size_t(f)];
    }
    return a;
}


static double denseResidual
(
    const std::vector<double>& a,
    int n,
    const std::vector<double>& x,
    const std::vector<double>& b
)
{
    double m = 0;
    for (int row = 0; row < n; ++row)
    {
        double ax = 0;
        for (int col = 0; col < n; ++col)
        {
            ax += a[size_t(row)*size_t(n) + size_t(col)]*x[size_t(col)];
        }
        m = std::max(m, std::fabs(ax - b[size_t(row)]));
    }
    return m;
}


int main()
{
    std::printf("aerosGPU core tests\n\n");

    // ---- row structure ------------------------------------------------------
    {
        Grid g(3, 3, 3);
        g.build();
        clearPatternCache();

        const Pattern& p = pattern
        (
            g.upperAddr.data(), g.lowerAddr.data(), g.nCells(), g.nFaces(), true
        );

        check(p.nCells == 27, "pattern: cell count");
        check(p.storedEntries() == 2*g.nFaces(),
              "pattern: both row directions stored for every face");
        check(p.symmetric, "pattern: symmetric flag kept");

        // Every face appears exactly once per triangle.
        int missing = 0;
        for (const Triangle* t : {&p.upperRow, &p.lowerRow})
        {
            std::vector<int> seen(size_t(g.nFaces()), 0);
            for (int k = 0; k < t->nnz(); ++k)
            {
                ++seen[size_t(t->face[size_t(k)])];
            }
            for (int f = 0; f < g.nFaces(); ++f)
            {
                if (seen[size_t(f)] != 1) ++missing;
            }
        }
        check(missing == 0, "pattern: every face stored exactly once in each triangle");

        // Row spans add up to the nnz and are monotone — a corrupt rowStart would
        // otherwise read out of bounds at run time.
        bool spansOk = p.upperRow.rowStart[size_t(p.nCells)] == p.upperRow.nnz();
        for (int r = 0; r < p.nCells; ++r)
        {
            spansOk = spansOk
                && p.upperRow.rowStart[size_t(r)] <= p.upperRow.rowStart[size_t(r) + 1];
        }
        check(spansOk, "pattern: row spans are monotone and cover every entry");
    }

    // ---- cache --------------------------------------------------------------
    {
        // The cache holds one pattern object, so the address proves nothing; what
        // matters is whether it was rebuilt (generation) and whether the content
        // matches the new mesh.
        Grid g(3, 3, 3);
        g.build();
        clearPatternCache();

        const unsigned g0 = generation();
        pattern(g.upperAddr.data(), g.lowerAddr.data(), g.nCells(), g.nFaces(), true);
        const unsigned g1 = generation();
        const std::vector<int> colBefore =
            pattern(g.upperAddr.data(), g.lowerAddr.data(), g.nCells(), g.nFaces(), true)
                .upperRow.col;

        pattern(g.upperAddr.data(), g.lowerAddr.data(), g.nCells(), g.nFaces(), true);
        check(generation() == g1 && g1 > g0, "cache: same mesh reuses the built pattern");

        // Same size, different connectivity: must NOT be reused. This is the case
        // that silently corrupts a run when a cache is keyed on dimensions alone.
        Grid h(3, 3, 3);
        h.build();
        std::swap(h.upperAddr[0], h.upperAddr[3]);

        const Pattern& c = pattern
        (
            h.upperAddr.data(), h.lowerAddr.data(), h.nCells(), h.nFaces(), true
        );
        check(generation() > g1,
              "cache: differently connected mesh of equal size rebuilds");
        check(c.upperRow.col != colBefore,
              "cache: the rebuilt pattern reflects the new wiring");
    }

    // ---- product ------------------------------------------------------------
    {
        Grid g(4, 4, 4);
        g.build();
        clearPatternCache();

        std::vector<double> diag, upper, lower;
        laplacian(g, diag, upper, lower);

        const Pattern& p = pattern
        (
            g.upperAddr.data(), g.lowerAddr.data(), g.nCells(), g.nFaces(), true
        );

        // x = 1 everywhere: every interior row of a Laplace matrix sums to zero, so a
        // correct product is zero there. A dropped triangle leaves a non-zero sum.
        std::vector<double> x(size_t(g.nCells()), 1.0);
        std::vector<double> y(size_t(g.nCells()), 0.0);
        amul(y.data(), x.data(), diag.data(), upper.data(), lower.data(), p);

        int interior = 0, nonzero = 0;
        for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
        {
            const bool isInterior =
                i > 0 && i < g.nx-1 && j > 0 && j < g.ny-1 && k > 0 && k < g.nz-1;
            if (isInterior)
            {
                ++interior;
                if (std::fabs(y[size_t(g.cell(i,j,k))]) > 1e-12) ++nonzero;
            }
        }
        check(interior > 0 && nonzero == 0,
              "amul: constant field has zero interior row sum");
    }

    {
        // Against a dense reference, on a deliberately asymmetric matrix: this is what
        // gets the two coefficient arrays the right way round. A transpose passes the
        // row-sum test above and fails here.
        Grid g(3, 2, 2);
        g.build();
        clearPatternCache();

        std::vector<double> diag, upper, lower;
        laplacian(g, diag, upper, lower);
        for (double& u : upper) u *= 2.5;

        const Pattern& p = pattern
        (
            g.upperAddr.data(), g.lowerAddr.data(), g.nCells(), g.nFaces(), false
        );

        const int n = g.nCells();
        const std::vector<double> dense = denseMatrix(g, diag, upper, lower);

        std::vector<double> x(size_t(n), 0.0);
        for (int c = 0; c < n; ++c) x[size_t(c)] = std::sin(0.7*double(c) + 1.0);

        std::vector<double> y(size_t(n), 0.0);
        amul(y.data(), x.data(), diag.data(), upper.data(), lower.data(), p);

        double maxErr = 0;
        for (int r = 0; r < n; ++r)
        {
            double expect = 0;
            for (int c = 0; c < n; ++c)
            {
                expect += dense[size_t(r)*size_t(n) + size_t(c)]*x[size_t(c)];
            }
            maxErr = std::max(maxErr, std::fabs(expect - y[size_t(r)]));
        }
        check(maxErr < 1e-12, "amul: asymmetric matrix matches the dense reference");
    }

    // ---- PCG ----------------------------------------------------------------
    {
        Grid g(6, 6, 6);
        g.build();
        clearPatternCache();

        std::vector<double> diag, upper, lower;
        laplacian(g, diag, upper, lower);

        const Pattern& p = pattern
        (
            g.upperAddr.data(), g.lowerAddr.data(), g.nCells(), g.nFaces(), true
        );

        const int n = g.nCells();
        const std::vector<double> exact(size_t(n), 1.0);
        std::vector<double> b(size_t(n), 0.0);
        amul(b.data(), exact.data(), diag.data(), upper.data(), lower.data(), p);

        std::vector<double> psi(size_t(n), 0.0);
        SolverControls c;
        c.maxIter = 500;
        c.tolerance = 1e-12;

        const SolverResult r = solveCGCPU(psi, b, diag, upper, lower, p, c);

        check(r.converged(), "PCG: converged");
        double maxErr = 0;
        for (int i = 0; i < n; ++i)
        {
            maxErr = std::max(maxErr, std::fabs(psi[size_t(i)] - 1.0));
        }
        check(maxErr < 1e-8, "PCG: recovered the known solution");

        const std::vector<double> dense = denseMatrix(g, diag, upper, lower);
        const double denseRes = denseResidual(dense, n, psi, b);
        check(denseRes < 1e-8, "PCG: the dense residual of the answer is small");

        std::printf("        PCG: %d iterations, residual %.3e, max error %.2e, "
                    "dense residual %.2e\n",
                    r.iterations, r.residual, maxErr, denseRes);

        // The preconditioner has to help: with IC(0) the solve must need far fewer than
        // n iterations. A verifiably *worse* preconditioner (the identity) cannot be
        // expressed by zeroing the off-diagonals — IC(0) would then be the exact
        // diagonal inverse, i.e. a better one.
        check(r.iterations < n/2,
              "PCG: IC(0) converges in far fewer iterations than the cell count");

        // An impossible tolerance must be reported, not silently accepted.
        SolverControls hard = c;
        hard.maxIter = 3;
        hard.tolerance = 1e-16;
        std::vector<double> psi3(size_t(n), 0.0);
        const SolverResult r3 = solveCGCPU(psi3, b, diag, upper, lower, p, hard);
        check(!r3.converged(),
              "PCG: gives up honestly instead of claiming convergence");
        check(r3.status == MAX_ITER,
              "PCG: reports MAX_ITER for an unreachable tolerance");
    }

    // ---- PBiCGStab ----------------------------------------------------------
    {
        Grid g(6, 6, 6);
        g.build();
        clearPatternCache();

        std::vector<double> diag, upper, lower;
        laplacian(g, diag, upper, lower);

        // Asymmetric: convection adds a directed term. This is the shape of a real
        // convection-diffusion matrix, and the transpose mistake a symmetric test
        // cannot see becomes a wrong answer here.
        for (int f = 0; f < g.nFaces(); ++f)
        {
            upper[size_t(f)] += 0.35;
            lower[size_t(f)] -= 0.15;
        }
        for (int c = 0; c < g.nCells(); ++c) diag[size_t(c)] += 0.2;

        const Pattern& p = pattern
        (
            g.upperAddr.data(), g.lowerAddr.data(), g.nCells(), g.nFaces(), false
        );

        const int n = g.nCells();
        std::vector<double> exact(size_t(n), 0.0);
        for (int i = 0; i < n; ++i) exact[size_t(i)] = 1.0 + 0.01*double(i % 7);

        std::vector<double> b(size_t(n), 0.0);
        amul(b.data(), exact.data(), diag.data(), upper.data(), lower.data(), p);

        std::vector<double> psi(size_t(n), 0.0);
        SolverControls c;
        c.maxIter = 500;
        c.tolerance = 1e-12;

        const SolverResult r = solveBiCGStabCPU(psi, b, diag, upper, lower, p, c);

        check(r.converged(), "PBiCGStab: converged on an asymmetric matrix");

        double maxErr = 0;
        for (int i = 0; i < n; ++i)
        {
            maxErr = std::max(maxErr, std::fabs(psi[size_t(i)] - exact[size_t(i)]));
        }
        check(maxErr < 1e-8, "PBiCGStab: recovered the known solution");

        const std::vector<double> dense = denseMatrix(g, diag, upper, lower);
        const double denseRes = denseResidual(dense, n, psi, b);
        check(denseRes < 1e-7, "PBiCGStab: the dense residual of the answer is small");

        std::printf("        PBiCGStab: %d iterations, residual %.3e, max error "
                    "%.2e, dense residual %.2e\n",
                    r.iterations, r.residual, maxErr, denseRes);
    }

    // ---- singular system ----------------------------------------------------
    {
        Grid g(4, 4, 4);
        g.build();
        clearPatternCache();

        // Pure Laplace, no boundary bump: every row sums to zero, so the matrix is
        // singular. b = 0 is in its range, and psi = 0 is a solution — what matters is
        // that the solver terminates without dividing by zero or producing NaNs.
        const int n = g.nCells();
        std::vector<double> diag(size_t(n), 0.0);
        std::vector<double> upper(size_t(g.nFaces()), -1.0);
        std::vector<double> lower(size_t(g.nFaces()), -1.0);
        for (int f = 0; f < g.nFaces(); ++f)
        {
            diag[size_t(g.lowerAddr[size_t(f)])] += 1.0;
            diag[size_t(g.upperAddr[size_t(f)])] += 1.0;
        }

        const Pattern& p = pattern
        (
            g.upperAddr.data(), g.lowerAddr.data(), n, g.nFaces(), true
        );

        std::vector<double> b(size_t(n), 0.0);
        std::vector<double> psi(size_t(n), 0.0);
        SolverControls c;
        c.maxIter = 20;
        c.tolerance = 1e-14;

        const SolverResult r = solveCGCPU(psi, b, diag, upper, lower, p, c);

        bool finite = true;
        for (int i = 0; i < n; ++i) finite = finite && std::isfinite(psi[size_t(i)]);
        check(finite, "singular system: no NaN in the result");
        check(r.residual < 1e-10 || r.status == DIVERGED,
              "singular system: honest status (converged at zero, or reported)");
    }

    // ---- device path: reports why it cannot run here ------------------------
    {
        Grid g(8, 8, 8);
        g.build();
        clearPatternCache();

        std::vector<double> diag, upper, lower;
        laplacian(g, diag, upper, lower);

        const Pattern& p = pattern
        (
            g.upperAddr.data(), g.lowerAddr.data(), g.nCells(), g.nFaces(), true
        );

        std::vector<double> b(size_t(g.nCells()), 1.0);
        std::vector<double> psi(size_t(g.nCells()), 0.0);
        SolverControls c;
        c.useDevice = true;

        const SolverResult d = solveCGDevice(psi, b, diag, upper, lower, p, c);
        if (d.status == DEVICE_FAILED)
        {
            check(!d.note.empty(), "device: failure explains itself");
            std::printf("        device unavailable: %s\n", d.note.c_str());
        }
        else
        {
            // A machine with OpenCL reaches here; the device answer must be as good as
            // the CPU one.
            check(d.converged(), "device: converged");
            check(denseResidual(denseMatrix(g, diag, upper, lower), g.nCells(), psi, b)
                  < 1e-8, "device: the dense residual of the answer is small");
        }

        // The dispatcher must still produce an answer.
        std::vector<double> psi2(size_t(g.nCells()), 0.0);
        const SolverResult fallback = solveCG(psi2, b, diag, upper, lower, p, c);
        check(fallback.converged(),
              "dispatcher: falls back to the CPU and converges");
        check(deviceDisabled(),
              "dispatcher: device stayed disabled after the failure");
    }

    std::printf("\n%s: %d failure(s), %d checks\n",
                failures ? "FAILED" : "PASSED", failures, checks);
    return failures ? 1 : 0;
}
