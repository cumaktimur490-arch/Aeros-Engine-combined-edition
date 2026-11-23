/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | OpenCL C source for the sparse solvers
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "aerosGpuKernels.H"

namespace Foam
{
namespace aerosGpu
{

// Design notes that are easy to lose in the code below:
//
// * Row-per-work-item SpMV. Rows are gathered, so no two work-items write the same
//   entry. One work-item per face, scattering to both endpoints, needs atomics and
//   loses reproducibility.
//
// * Reductions are deterministic: a fixed-size local tree, then the group partials
//   summed in group order. Not the fastest reduction, but the result depends on the
//   data and the launch geometry only, so a CPU/GPU comparison is meaningful.
//
// * The preconditioner sweeps run as a single work-item, in face order. That order is
//   the algorithm (see aerosGpuSolver.C); a data-parallel version needs level
//   scheduling and is a separate optimisation, and getting it slightly wrong changes
//   the preconditioner silently.

namespace kernelNames
{
    const char* amul         = "aeros_amul";
    const char* residual     = "aeros_residual";
    const char* dot          = "aeros_dot";
    const char* axpy         = "aeros_axpy";
    const char* xpay         = "aeros_xpay";
    const char* copy         = "aeros_copy";
    const char* scale        = "aeros_scale";
    const char* dicFactorise = "aeros_dic_factorise";
    const char* dicSolve     = "aeros_dic_solve";
    const char* diluFactorise= "aeros_dilu_factorise";
    const char* diluSolve    = "aeros_dilu_solve";
}


const char* kernelSource()
{
    return R"CLC(

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// y = A*x, one work-item per row, both triangles in one pass.
//
// own1/rs1/col1/face1 describe the rows keyed by the lower cell of each face; the
// coefficients those rows own are `lower` (or the single array of a symmetric
// matrix). own2/... the rows keyed by the upper cell, coefficients `upper`. Both are
// always bound: a face contributes to two different rows, and a product that walks
// only one triangle loses half of every interior row — which makes even a constant
// field have a non-zero row sum.
__kernel void aeros_amul
(
    __global const double* x,
    __global const double* diag,
    __global const double* own1,
    __global const int*    rs1,
    __global const int*    col1,
    __global const int*    face1,
    __global const double* own2,
    __global const int*    rs2,
    __global const int*    col2,
    __global const int*    face2,
    __global double*       y,
    const int              nRows
)
{
    const int row = get_global_id(0);
    if (row >= nRows) return;

    double acc = diag[row]*x[row];

    for (int k = rs1[row]; k < rs1[row + 1]; ++k)
    {
        acc += own1[face1[k]]*x[col1[k]];
    }
    for (int k = rs2[row]; k < rs2[row + 1]; ++k)
    {
        acc += own2[face2[k]]*x[col2[k]];
    }

    y[row] = acc;
}

// r = b - A*x, recomputed from scratch. Never accumulated recursively: the recursive
// residual drifts, and a solver that reports convergence it no longer has is worse
// than one that reports failure.
__kernel void aeros_residual
(
    __global const double* b,
    __global const double* x,
    __global const double* diag,
    __global const double* own1,
    __global const int*    rs1,
    __global const int*    col1,
    __global const int*    face1,
    __global const double* own2,
    __global const int*    rs2,
    __global const int*    col2,
    __global const int*    face2,
    __global double*       r,
    const int              nRows
)
{
    const int row = get_global_id(0);
    if (row >= nRows) return;

    double acc = diag[row]*x[row];

    for (int k = rs1[row]; k < rs1[row + 1]; ++k)
    {
        acc += own1[face1[k]]*x[col1[k]];
    }
    for (int k = rs2[row]; k < rs2[row + 1]; ++k)
    {
        acc += own2[face2[k]]*x[col2[k]];
    }

    r[row] = b[row] - acc;
}

// Deterministic dot product. Each work-item walks a strided slice (coalesced reads),
// groups reduce through a local tree, and the partials are combined in group order.
__kernel void aeros_dot
(
    __global const double* a,
    __global const double* b,
    __global double*       partials,
    __local double*        scratch,
    const int              n
)
{
    const int lid     = get_local_id(0);
    const int lsize   = get_local_size(0);
    const int nGroups = get_num_groups(0);

    double v = 0.0;
    for (int i = get_global_id(0); i < n; i += get_global_size(0))
    {
        v += a[i]*b[i];
    }

    scratch[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1)
    {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) partials[get_group_id(0)] = scratch[0];

    if (get_group_id(0) == 0 && lid == 0)
    {
        double s = 0.0;
        for (int g = 0; g < nGroups; ++g) s += partials[g];
        partials[0] = s;
    }
}

// y += alpha*x
__kernel void aeros_axpy
(
    __global double*       y,
    __global const double* x,
    const double           alpha,
    const int              n
)
{
    const int i = get_global_id(0);
    if (i < n) y[i] += alpha*x[i];
}

// x += alpha*y
__kernel void aeros_xpay
(
    __global double*       x,
    __global const double* y,
    const double           alpha,
    const int              n
)
{
    const int i = get_global_id(0);
    if (i < n) x[i] += alpha*y[i];
}

__kernel void aeros_copy
(
    __global double*       dst,
    __global const double* src,
    const int              n
)
{
    const int i = get_global_id(0);
    if (i < n) dst[i] = src[i];
}

__kernel void aeros_scale
(
    __global double* x,
    const double     alpha,
    const int        n
)
{
    const int i = get_global_id(0);
    if (i < n) x[i] *= alpha;
}

// IC(0) diagonal, swept in face order exactly as OpenFOAM's DICPreconditioner does.
// The order is the algorithm: because upperAddr[f] > lowerAddr[f], by the time face f
// is processed the diagonal of its lower cell is final. rD comes in as the matrix
// diagonal and leaves as its reciprocal.
__kernel void aeros_dic_factorise
(
    __global double*       rD,
    __global const double* upper,
    __global const int*    upperAddr,
    __global const int*    lowerAddr,
    const int              nFaces,
    const int              nRows
)
{
    for (int f = 0; f < nFaces; ++f)
    {
        const double u = upper[f];
        rD[upperAddr[f]] -= u*u/rD[lowerAddr[f]];
    }

    for (int row = 0; row < nRows; ++row)
    {
        // A non-positive pivot cannot be factorised; clamping keeps the run alive and
        // shows up as slow convergence the caller can report, instead of NaNs
        // spreading through the field.
        const double d = rD[row];
        rD[row] = 1.0/(d > 1e-300 ? d : 1e-300);
    }
}

__kernel void aeros_dic_solve
(
    __global double*       x,
    __global const double* rD,
    __global const double* upper,
    __global const int*    upperAddr,
    __global const int*    lowerAddr,
    const int              nFaces,
    const int              nRows
)
{
    for (int row = 0; row < nRows; ++row)
    {
        x[row] *= rD[row];
    }

    // forward: the face's upper cell, in face order
    for (int f = 0; f < nFaces; ++f)
    {
        x[upperAddr[f]] -= rD[upperAddr[f]]*upper[f]*x[lowerAddr[f]];
    }

    // backward: the face's lower cell, reverse face order
    for (int f = nFaces - 1; f >= 0; --f)
    {
        x[lowerAddr[f]] -= rD[lowerAddr[f]]*upper[f]*x[upperAddr[f]];
    }
}

// DILU: the same sweep with both coefficients, for asymmetric matrices.
__kernel void aeros_dilu_factorise
(
    __global double*       rD,
    __global const double* upper,
    __global const double* lower,
    __global const int*    upperAddr,
    __global const int*    lowerAddr,
    const int              nFaces,
    const int              nRows
)
{
    for (int f = 0; f < nFaces; ++f)
    {
        rD[upperAddr[f]] -= upper[f]*lower[f]/rD[lowerAddr[f]];
    }

    for (int row = 0; row < nRows; ++row)
    {
        const double d = rD[row];
        rD[row] = 1.0/(fabs(d) > 1e-300 ? d : (d < 0.0 ? -1e-300 : 1e-300));
    }
}

__kernel void aeros_dilu_solve
(
    __global double*       x,
    __global const double* rD,
    __global const double* upper,
    __global const double* lower,
    __global const int*    upperAddr,
    __global const int*    lowerAddr,
    const int              nFaces,
    const int              nRows
)
{
    for (int row = 0; row < nRows; ++row)
    {
        x[row] *= rD[row];
    }

    for (int f = 0; f < nFaces; ++f)
    {
        x[upperAddr[f]] -= rD[upperAddr[f]]*lower[f]*x[lowerAddr[f]];
    }

    for (int f = nFaces - 1; f >= 0; --f)
    {
        x[lowerAddr[f]] -= rD[lowerAddr[f]]*upper[f]*x[upperAddr[f]];
    }
}

)CLC";
}

} // namespace aerosGpu
} // namespace Foam
