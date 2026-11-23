/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | lduMatrix -> row structure, cached per mesh
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "aerosGpuMatrix.H"

#include <algorithm>

namespace Foam
{
namespace aerosGpu
{

namespace
{
    Pattern cached;
    bool haveCached = false;
    unsigned cacheGeneration = 0;

    // Counting sort of face indices by row. Writing into the counted positions in
    // increasing face order keeps every row's columns sorted, so the pattern is
    // stable across runs and the result is reproducible.
    void buildTriangle
    (
        Triangle& t,
        const int* rowOfFace,
        const int* colOfFace,
        int nRows,
        int nFaces
    )
    {
        t.rowStart.assign(size_t(nRows) + 1, 0);

        for (int f = 0; f < nFaces; ++f)
        {
            ++t.rowStart[size_t(rowOfFace[f]) + 1];
        }
        for (int r = 0; r < nRows; ++r)
        {
            t.rowStart[size_t(r) + 1] += t.rowStart[size_t(r)];
        }

        if (nFaces == 0)
        {
            return;
        }

        t.col.assign(size_t(nFaces), 0);
        t.face.assign(size_t(nFaces), 0);

        std::vector<int> cursor(t.rowStart.begin(), t.rowStart.end() - 1);

        for (int f = 0; f < nFaces; ++f)
        {
            const int r = rowOfFace[f];
            const int pos = cursor[size_t(r)]++;
            t.col[size_t(pos)] = colOfFace[f];
            t.face[size_t(pos)] = f;
        }
    }
}


int Triangle::faceOwnerRow(int k) const
{
    if (int(ownerRowCache.size()) != nnz())
    {
        ownerRowCache.assign(size_t(nnz()), 0);
        for (int r = 0; r < rows(); ++r)
        {
            for (int e = rowStart[size_t(r)]; e < rowStart[size_t(r) + 1]; ++e)
            {
                ownerRowCache[size_t(e)] = r;
            }
        }
    }
    return ownerRowCache[size_t(k)];
}


void Pattern::clear()
{
    upperRow.clear();
    lowerRow.clear();
    upperAddrCopy.clear();
    lowerAddrCopy.clear();
    nCells = 0;
    nFaces = 0;
    symmetric = true;
}


int Pattern::maxRowLength() const
{
    int m = 0;
    for (int r = 0; r < upperRow.rows(); ++r)
    {
        m = std::max(m, upperRow.rowStart[size_t(r) + 1] - upperRow.rowStart[size_t(r)]);
    }
    for (int r = 0; r < lowerRow.rows(); ++r)
    {
        m = std::max(m, lowerRow.rowStart[size_t(r) + 1] - lowerRow.rowStart[size_t(r)]);
    }
    return m;
}


const Pattern& pattern
(
    const int* upperAddr,
    const int* lowerAddr,
    int nCells,
    int nFaces,
    bool symmetric
)
{
    if (haveCached)
    {
        bool same = cached.nCells == nCells
                 && cached.nFaces == nFaces
                 && cached.symmetric == symmetric;

        // Same shape is not the same pattern: two meshes of equal size can have
        // different addressing, so the face endpoints are compared as well. A cache
        // keyed on counts alone would be reused by a re-meshed case.
        if (same)
        {
            for (int f = 0; f < nFaces && same; ++f)
            {
                same = cached.upperAddrCopy[size_t(f)] == upperAddr[f]
                    && cached.lowerAddrCopy[size_t(f)] == lowerAddr[f];
            }
        }

        if (same)
        {
            return cached;
        }
    }

    cached.clear();
    cached.nCells = nCells;
    cached.nFaces = nFaces;
    cached.symmetric = symmetric;

    // Both triangles, always. See the header: a row needs the entry belonging to the
    // other cell of the face regardless of whether the values are symmetric.
    buildTriangle(cached.upperRow, lowerAddr, upperAddr, nCells, nFaces);
    buildTriangle(cached.lowerRow, upperAddr, lowerAddr, nCells, nFaces);

    cached.upperAddrCopy.assign(upperAddr, upperAddr + nFaces);
    cached.lowerAddrCopy.assign(lowerAddr, lowerAddr + nFaces);

    haveCached = true;
    ++cacheGeneration;

    return cached;
}


unsigned generation()
{
    return cacheGeneration;
}


void clearPatternCache()
{
    cached.clear();
    haveCached = false;
    ++cacheGeneration;
}


namespace
{
    // One row of the product. `own` is the coefficient array whose entries belong to
    // this row (lower for the lower-cell rows), `other` the neighbour's.
    inline void rowDot
    (
        double& acc,
        int row,
        const double* x,
        const double* diag,
        const int* rowStart,
        const int* col,
        const int* face,
        const double* own
    )
    {
        acc = diag[row]*x[row];

        const int end = rowStart[row + 1];
        for (int k = rowStart[row]; k < end; ++k)
        {
            // Gathered: this row owns the coefficient, so no other row writes it.
            // That is what makes the loop parallel-safe, unlike OpenFOAM's face-wise
            // scatter which touches both endpoints.
            acc += own[face[k]]*x[col[k]];
        }
    }
}


void amul
(
    double* y,
    const double* x,
    const double* diag,
    const double* upper,
    const double* lower,
    const Pattern& p
)
{
    // Two passes, one per triangle, because a face's two coefficients belong to two
    // different rows.
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < p.nCells; ++row)
    {
        double acc = 0;
        rowDot
        (
            acc, row, x, diag,
            p.upperRow.rowStart.data(),
            p.upperRow.col.data(),
            p.upperRow.face.data(),
            lower
        );
        y[row] = acc;
    }

    if (p.lowerRow.nnz())
    {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < p.nCells; ++row)
        {
            double acc = 0;
            rowDot
            (
                acc, row, x, diag,
                p.lowerRow.rowStart.data(),
                p.lowerRow.col.data(),
                p.lowerRow.face.data(),
                upper
            );
            // rowDot includes the diagonal; subtract it back out so the second pass
            // only adds the second triangle.
            y[row] += acc - diag[row]*x[row];
        }
    }
}


void amulAdd
(
    double* y,
    double alpha,
    const double* x,
    const double* diag,
    const double* upper,
    const double* lower,
    const Pattern& p
)
{
    std::vector<double> tmp(size_t(p.nCells), 0.0);
    amul(tmp.data(), x, diag, upper, lower, p);

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < p.nCells; ++row)
    {
        y[row] += alpha*tmp[size_t(row)];
    }
}

} // namespace aerosGpu
} // namespace Foam
