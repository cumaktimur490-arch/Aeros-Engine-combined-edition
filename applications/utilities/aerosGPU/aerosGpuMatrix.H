/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | lduMatrix -> row structure, cached per mesh
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

// OpenFOAM stores a face-based matrix: each face carries one coefficient for each
// of the two cells it joins. A device SpMV wants rows. The conversion is a counting
// sort — cheap, but not free — and the *pattern* depends only on the mesh, so it is
// built once and reused while the coefficients keep being read from lduMatrix on
// every solve.
//
// Both triangles are always built, even for a symmetric matrix. A face joins cells
// i and j, and row i needs the entry it owns as well as the entry that belongs to
// row j. Building only "the upper triangle" because the values are symmetric is the
// mistake that gives a constant field a non-zero interior row sum — see the test.

#ifndef aerosGpuMatrix_H
#define aerosGpuMatrix_H

#include <vector>

namespace Foam
{
namespace aerosGpu
{

// Row structure for one triangle.
//
// `col` holds the column (the other cell of the face) of every stored entry and
// `face` the lduMatrix face it came from, so the device gets the coefficients as a
// gather out of the lduMatrix arrays instead of a re-ordered copy.
struct Triangle
{
    std::vector<int> rowStart;  // row r owns entries [rowStart[r], rowStart[r+1])
    std::vector<int> col;
    std::vector<int> face;

    // Row that owns entry k. The preconditioners walk entries in face order and
    // need to know which row each one updates; face order and row order are not the
    // same sequence. Built lazily and cached.
    int faceOwnerRow(int k) const;

    int rows() const { return rowStart.empty() ? 0 : int(rowStart.size()) - 1; }
    int nnz() const { return int(col.size()); }
    void clear() { rowStart.clear(); col.clear(); face.clear(); ownerRowCache.clear(); }

    mutable std::vector<int> ownerRowCache;
};


struct Pattern
{
    int nCells;
    int nFaces;
    bool symmetric;

    // Rows keyed by the lower cell of each face, columns are the upper cells, and
    // the coefficients for these rows are the `lower` array.
    Triangle upperRow;

    // Rows keyed by the upper cell, columns are the lower cells, coefficients the
    // `upper` array. For a symmetric matrix there is one coefficient array and the
    // caller passes it for both; the row structure is still needed.
    Triangle lowerRow;

    // The addressing the pattern was built from, so an equal-sized but differently
    // connected mesh is not mistaken for a cache hit.
    std::vector<int> upperAddrCopy;
    std::vector<int> lowerAddrCopy;

    Pattern() : nCells(0), nFaces(0), symmetric(true) {}

    void clear();

    // Entries the device has to index: every face appears once in each triangle.
    int storedEntries() const { return upperRow.nnz() + lowerRow.nnz(); }

    int maxRowLength() const;
};


// Builds, or returns the cached, pattern. `upperAddr[f] > lowerAddr[f]` must hold,
// as it does for lduMatrix.
const Pattern& pattern
(
    const int* upperAddr,
    const int* lowerAddr,
    int nCells,
    int nFaces,
    bool symmetric
);

// Bumped whenever the cached pattern is rebuilt, so device buffers keyed on it
// cannot silently outlive the mesh they were built for.
unsigned generation();

void clearPatternCache();


// y = A*x. `lower` holds the coefficients of the lower-cell rows and `upper` those
// of the upper-cell rows (the same array for a symmetric matrix).
//
// Row-wise gather rather than OpenFOAM's face-wise scatter: each row writes only its
// own entry, so this is safe to run on several threads and has the same arithmetic
// as the device kernel.
void amul
(
    double* y,
    const double* x,
    const double* diag,
    const double* upper,
    const double* lower,
    const Pattern& p
);

// y += alpha*A*x
void amulAdd
(
    double* y,
    double alpha,
    const double* x,
    const double* diag,
    const double* upper,
    const double* lower,
    const Pattern& p
);

} // namespace aerosGpu
} // namespace Foam

#endif
