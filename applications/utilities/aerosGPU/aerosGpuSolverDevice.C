/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | Host side of the device solvers: buffers, kernel
    \\  /    A nd           | dispatch, and the CPU cross-check that gates them
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "aerosGpuSolver.H"
#include "aerosGpuDevice.H"
#include "aerosGpuCl.H"
#include "aerosGpuKernels.H"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace Foam
{
namespace aerosGpu
{

namespace
{
    // One context for the whole run: building the program costs a few hundred
    // milliseconds, which would otherwise be paid again for every field.
    Context* sharedContext = nullptr;

    // Buffers are recreated when the pattern changes and reused otherwise.
    int cachedGeneration = -1;
    int bufCells = 0;
    int bufNnz = 0;

    cl::cl_mem d_x = nullptr;    // solution
    cl::cl_mem d_s = nullptr;    // right-hand side
    cl::cl_mem d_r = nullptr;    // residual
    cl::cl_mem d_y = nullptr;    // rA0 (bi-CGStab) / A*p (CG)
    cl::cl_mem d_z = nullptr;    // preconditioned vector
    cl::cl_mem d_p = nullptr;    // search direction
    cl::cl_mem d_v = nullptr;    // A*(direction)
    cl::cl_mem d_a = nullptr;    // sA
    cl::cl_mem d_b = nullptr;    // zA = M^-1 sA
    cl::cl_mem d_c = nullptr;    // tA = A*zA
    cl::cl_mem d_rD = nullptr;

    cl::cl_mem d_diag = nullptr;
    cl::cl_mem d_upper = nullptr;
    cl::cl_mem d_lower = nullptr;

    cl::cl_mem d_rs1 = nullptr;
    cl::cl_mem d_col1 = nullptr;
    cl::cl_mem d_face1 = nullptr;
    cl::cl_mem d_rs2 = nullptr;
    cl::cl_mem d_col2 = nullptr;
    cl::cl_mem d_face2 = nullptr;

    cl::cl_mem d_upperAddr = nullptr;
    cl::cl_mem d_lowerAddr = nullptr;

    cl::cl_mem d_partials = nullptr;

    struct Kernels
    {
        cl::cl_kernel amul;
        cl::cl_kernel residual;
        cl::cl_kernel dot;
        cl::cl_kernel axpy;
        cl::cl_kernel xpay;
        cl::cl_kernel copy;
        cl::cl_kernel scale;
        cl::cl_kernel dicFactorise;
        cl::cl_kernel dicSolve;
        cl::cl_kernel diluFactorise;
        cl::cl_kernel diluSolve;

        Kernels()
        :
            amul(nullptr), residual(nullptr), dot(nullptr), axpy(nullptr),
            xpay(nullptr), copy(nullptr), scale(nullptr),
            dicFactorise(nullptr), dicSolve(nullptr),
            diluFactorise(nullptr), diluSolve(nullptr)
        {}
    };
    Kernels k;

    // 256 work-items per group; the number of groups is fixed by the problem size so
    // the summation order is reproducible.
    const int kReductionGroup = 256;

    cl::cl_context ctx() { return static_cast<cl::cl_context>(sharedContext->context()); }
    cl::cl_command_queue queue() { return static_cast<cl::cl_command_queue>(sharedContext->queue()); }


    void releaseBuffers()
    {
        cl::cl_mem* all[] =
        {
            &d_x, &d_s, &d_r, &d_y, &d_z, &d_p, &d_v, &d_a, &d_b, &d_c, &d_rD,
            &d_diag, &d_upper, &d_lower,
            &d_rs1, &d_col1, &d_face1, &d_rs2, &d_col2, &d_face2,
            &d_upperAddr, &d_lowerAddr, &d_partials
        };
        for (cl::cl_mem* m : all)
        {
            if (*m)
            {
                cl::clReleaseMemObject(*m);
                *m = nullptr;
            }
        }
        bufCells = 0;
        bufNnz = 0;
        cachedGeneration = -1;
    }


    bool createBuffer
    (
        cl::cl_mem& m,
        size_t bytes,
        cl::cl_bitfield flags,
        void* host,
        std::string& err
    )
    {
        cl::cl_int e = cl::CL_SUCCESS_;
        m = cl::clCreateBuffer(ctx(), flags, bytes, host, &e);
        if (e != cl::CL_SUCCESS_ || !m)
        {
            std::ostringstream os;
            os << "clCreateBuffer failed (" << e << ") for " << bytes << " bytes"
               << (host ? "" : " (is the mesh too large for this device?)");
            err = os.str();
            return false;
        }
        return true;
    }


    bool writeBuffer(cl::cl_mem m, size_t bytes, const void* host, std::string& err)
    {
        const cl::cl_int e =
            cl::clEnqueueWriteBuffer(queue(), m, cl::CL_TRUE_, 0, bytes, host, 0, nullptr, nullptr);
        if (e != cl::CL_SUCCESS_)
        {
            err = "upload failed (" + std::to_string(e) + ")";
            return false;
        }
        return true;
    }


    bool readBuffer(cl::cl_mem m, size_t bytes, void* host, std::string& err)
    {
        const cl::cl_int e =
            cl::clEnqueueReadBuffer(queue(), m, cl::CL_TRUE_, 0, bytes, host, 0, nullptr, nullptr);
        if (e != cl::CL_SUCCESS_)
        {
            err = "download failed (" + std::to_string(e) + ")";
            return false;
        }
        return true;
    }


    bool launch1D(cl::cl_kernel kern, int n, std::string& err)
    {
        if (!kern)
        {
            err = "kernel was not created";
            return false;
        }
        const size_t global = size_t(n > 0 ? n : 1);
        const cl::cl_int e =
            cl::clEnqueueNDRangeKernel(queue(), kern, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (e != cl::CL_SUCCESS_)
        {
            err = "kernel launch failed (" + std::to_string(e) + ")";
            return false;
        }
        return true;
    }


    bool makeKernel(cl::cl_kernel& out, const char* name, std::string& err)
    {
        out = static_cast<cl::cl_kernel>(sharedContext->kernel(name));
        if (!out)
        {
            err = std::string("kernel not found in program: ") + name;
            return false;
        }
        return true;
    }


    bool setMem(cl::cl_kernel kern, cl::cl_uint i, cl::cl_mem m, std::string& err)
    {
        if (cl::clSetKernelArg(kern, i, sizeof(cl::cl_mem), &m) != cl::CL_SUCCESS_)
        {
            err = "clSetKernelArg(buffer) failed";
            return false;
        }
        return true;
    }


    bool setInt(cl::cl_kernel kern, cl::cl_uint i, int v, std::string& err)
    {
        if (cl::clSetKernelArg(kern, i, sizeof(int), &v) != cl::CL_SUCCESS_)
        {
            err = "clSetKernelArg(int) failed";
            return false;
        }
        return true;
    }


    bool setDouble(cl::cl_kernel kern, cl::cl_uint i, double v, std::string& err)
    {
        if (cl::clSetKernelArg(kern, i, sizeof(double), &v) != cl::CL_SUCCESS_)
        {
            err = "clSetKernelArg(double) failed";
            return false;
        }
        return true;
    }


    // Argument layout of aeros_amul / aeros_residual, kept as constants because
    // getting one off by one gives a kernel that runs and reads the wrong buffer — no
    // compiler catches that.
    enum
    {
        AM_X = 0, AM_DIAG, AM_OWN1, AM_RS1, AM_COL1, AM_FACE1,
        AM_OWN2, AM_RS2, AM_COL2, AM_FACE2, AM_Y, AM_NROWS
    };


    bool setMatrixArgs(cl::cl_kernel kern, const Pattern& p, std::string& err)
    {
        // A symmetric matrix stores one coefficient array: the same buffer serves both
        // row directions. The row structures still differ and both are bound.
        return
            setMem(kern, AM_DIAG, d_diag, err)
         && setMem(kern, AM_OWN1, d_lower, err)
         && setMem(kern, AM_RS1, d_rs1, err)
         && setMem(kern, AM_COL1, d_col1, err)
         && setMem(kern, AM_FACE1, d_face1, err)
         && setMem(kern, AM_OWN2, p.symmetric ? d_lower : d_upper, err)
         && setMem(kern, AM_RS2, d_rs2, err)
         && setMem(kern, AM_COL2, d_col2, err)
         && setMem(kern, AM_FACE2, d_face2, err)
         && setInt(kern, AM_NROWS, p.nCells, err);
    }


    bool ensureBuffers
    (
        const std::vector<double>& diag,
        const std::vector<double>& upper,
        const std::vector<double>& lower,
        const Pattern& p,
        std::string& err
    )
    {
        const int nc = p.nCells;
        const size_t bytesCells = sizeof(double)*size_t(nc);

        if
        (
            cachedGeneration == int(generation())
         && bufCells == nc
         && bufNnz == p.storedEntries()
        )
        {
            // Same pattern: only the coefficients changed.
            return
                writeBuffer(d_diag, bytesCells, diag.data(), err)
             && writeBuffer(d_upper, sizeof(double)*upper.size(), upper.data(), err)
             && writeBuffer(d_lower, sizeof(double)*lower.size(), lower.data(), err);
        }

        releaseBuffers();

        bool ok =
            createBuffer(d_x, bytesCells, cl::CL_MEM_READ_WRITE_, nullptr, err)
         && createBuffer(d_s, bytesCells, cl::CL_MEM_READ_WRITE_, nullptr, err)
         && createBuffer(d_r, bytesCells, cl::CL_MEM_READ_WRITE_, nullptr, err)
         && createBuffer(d_y, bytesCells, cl::CL_MEM_READ_WRITE_, nullptr, err)
         && createBuffer(d_z, bytesCells, cl::CL_MEM_READ_WRITE_, nullptr, err)
         && createBuffer(d_p, bytesCells, cl::CL_MEM_READ_WRITE_, nullptr, err)
         && createBuffer(d_v, bytesCells, cl::CL_MEM_READ_WRITE_, nullptr, err)
         && createBuffer(d_a, bytesCells, cl::CL_MEM_READ_WRITE_, nullptr, err)
         && createBuffer(d_b, bytesCells, cl::CL_MEM_READ_WRITE_, nullptr, err)
         && createBuffer(d_c, bytesCells, cl::CL_MEM_READ_WRITE_, nullptr, err)
         && createBuffer(d_rD, bytesCells, cl::CL_MEM_READ_WRITE_, nullptr, err)
         && createBuffer(d_partials, sizeof(double)*size_t(kReductionGroup),
                         cl::CL_MEM_READ_WRITE_, nullptr, err)
         && createBuffer(d_diag, bytesCells,
                         cl::CL_MEM_READ_ONLY_ | cl::CL_MEM_COPY_HOST_PTR_,
                         const_cast<double*>(diag.data()), err)
         && createBuffer(d_upper, sizeof(double)*upper.size(),
                         cl::CL_MEM_READ_ONLY_ | cl::CL_MEM_COPY_HOST_PTR_,
                         const_cast<double*>(upper.data()), err)
         && createBuffer(d_lower, sizeof(double)*lower.size(),
                         cl::CL_MEM_READ_ONLY_ | cl::CL_MEM_COPY_HOST_PTR_,
                         const_cast<double*>(lower.data()), err)
         && createBuffer(d_rs1, sizeof(int)*(size_t(nc) + 1),
                         cl::CL_MEM_READ_ONLY_ | cl::CL_MEM_COPY_HOST_PTR_,
                         const_cast<int*>(p.upperRow.rowStart.data()), err)
         && createBuffer(d_col1, sizeof(int)*size_t(p.upperRow.nnz() ? p.upperRow.nnz() : 1),
                         cl::CL_MEM_READ_ONLY_ | cl::CL_MEM_COPY_HOST_PTR_,
                         const_cast<int*>(p.upperRow.col.data()), err)
         && createBuffer(d_face1, sizeof(int)*size_t(p.upperRow.nnz() ? p.upperRow.nnz() : 1),
                         cl::CL_MEM_READ_ONLY_ | cl::CL_MEM_COPY_HOST_PTR_,
                         const_cast<int*>(p.upperRow.face.data()), err);

        if (!ok)
        {
            return false;
        }

        // The second row structure is needed even for a symmetric matrix: it is not a
        // second copy of the coefficients, it is the other half of the rows.
        ok =
            createBuffer(d_rs2, sizeof(int)*(size_t(nc) + 1),
                         cl::CL_MEM_READ_ONLY_ | cl::CL_MEM_COPY_HOST_PTR_,
                         const_cast<int*>(p.lowerRow.rowStart.data()), err)
         && createBuffer(d_col2, sizeof(int)*size_t(p.lowerRow.nnz() ? p.lowerRow.nnz() : 1),
                         cl::CL_MEM_READ_ONLY_ | cl::CL_MEM_COPY_HOST_PTR_,
                         const_cast<int*>(p.lowerRow.col.data()), err)
         && createBuffer(d_face2, sizeof(int)*size_t(p.lowerRow.nnz() ? p.lowerRow.nnz() : 1),
                         cl::CL_MEM_READ_ONLY_ | cl::CL_MEM_COPY_HOST_PTR_,
                         const_cast<int*>(p.lowerRow.face.data()), err)
         && createBuffer(d_upperAddr, sizeof(int)*size_t(p.nFaces ? p.nFaces : 1),
                         cl::CL_MEM_READ_ONLY_ | cl::CL_MEM_COPY_HOST_PTR_,
                         const_cast<int*>(p.upperAddrCopy.data()), err)
         && createBuffer(d_lowerAddr, sizeof(int)*size_t(p.nFaces ? p.nFaces : 1),
                         cl::CL_MEM_READ_ONLY_ | cl::CL_MEM_COPY_HOST_PTR_,
                         const_cast<int*>(p.lowerAddrCopy.data()), err);

        if (!ok)
        {
            return false;
        }

        cachedGeneration = int(generation());
        bufCells = nc;
        bufNnz = p.storedEntries();
        return true;
    }


    // The CPU cross-check that gates the device: one SpMV compared against the CPU
    // reference. It catches the two mistakes that matter — a transposed matrix (wrong
    // coefficient array bound to a triangle) and a dropped triangle. Both let a solver
    // converge on symmetric toy cases and fail on real ones, so they cannot be left to
    // be discovered later.
    bool crossCheck
    (
        const std::vector<double>& x,
        const std::vector<double>& diag,
        const std::vector<double>& upper,
        const std::vector<double>& lower,
        const Pattern& p,
        std::string& err
    )
    {
        std::vector<double> cpu(size_t(p.nCells), 0.0);
        amul(cpu.data(), x.data(), diag.data(), upper.data(), lower.data(), p);

        const size_t bytes = sizeof(double)*size_t(p.nCells);
        if (!writeBuffer(d_x, bytes, x.data(), err)) return false;

        if
        (
            !setMem(k.amul, AM_X, d_x, err)
         || !setMem(k.amul, AM_Y, d_y, err)
         || !setMatrixArgs(k.amul, p, err)
         || !launch1D(k.amul, p.nCells, err)
        )
        {
            return false;
        }

        std::vector<double> gpu(size_t(p.nCells), 0.0);
        if (!readBuffer(d_y, bytes, gpu.data(), err)) return false;

        double num = 0, den = 0;
        for (int i = 0; i < p.nCells; ++i)
        {
            const double d = gpu[size_t(i)] - cpu[size_t(i)];
            num += d*d;
            den += cpu[size_t(i)]*cpu[size_t(i)];
        }
        const double rel = std::sqrt(num)/(std::sqrt(den) > 0 ? std::sqrt(den) : 1.0);

        if (!std::isfinite(rel) || rel > 1e-9)
        {
            std::ostringstream os;
            os << "device SpMV disagrees with the CPU reference (relative " << rel
               << "); device disabled for this run";
            err = os.str();
            return false;
        }
        return true;
    }


    bool dotProduct(cl::cl_mem a, cl::cl_mem b, int n, double& out, std::string& err)
    {
        const int groups = (n + kReductionGroup - 1)/kReductionGroup;
        const size_t global = size_t(kReductionGroup)*size_t(groups > 0 ? groups : 1);
        const size_t local = size_t(kReductionGroup);

        if
        (
            !setMem(k.dot, 0, a, err)
         || !setMem(k.dot, 1, b, err)
         || !setMem(k.dot, 2, d_partials, err)
        )
        {
            return false;
        }

        // The __local scratch is passed by size, not by pointer (argument index 3).
        if
        (
            cl::clSetKernelArg(k.dot, 3, sizeof(double)*size_t(kReductionGroup), nullptr)
            != cl::CL_SUCCESS_
        )
        {
            err = "clSetKernelArg(local scratch) failed";
            return false;
        }

        if (!setInt(k.dot, 4, n, err)) return false;

        const cl::cl_int le =
            cl::clEnqueueNDRangeKernel
            (
                queue(), k.dot, 1, nullptr, &global, &local, 0, nullptr, nullptr
            );
        if (le != cl::CL_SUCCESS_)
        {
            err = "dot kernel launch failed (" + std::to_string(le) + ")";
            return false;
        }

        return readBuffer(d_partials, sizeof(double), &out, err);
    }


    bool computeResidual
    (
        cl::cl_mem b,
        cl::cl_mem x,
        cl::cl_mem r,
        const Pattern& p,
        std::string& err
    )
    {
        return
            setMem(k.residual, 0, b, err)
         && setMem(k.residual, 1, x, err)
         && setMatrixArgs(k.residual, p, err)
         && setMem(k.residual, AM_Y, r, err)
         && launch1D(k.residual, p.nCells, err);
    }


    bool xpay(cl::cl_mem x, cl::cl_mem y, double alpha, int n, std::string& err)
    {
        return
            setMem(k.xpay, 0, x, err)
         && setMem(k.xpay, 1, y, err)
         && setDouble(k.xpay, 2, alpha, err)
         && setInt(k.xpay, 3, n, err)
         && launch1D(k.xpay, n, err);
    }


    bool axpy(cl::cl_mem y, cl::cl_mem x, double alpha, int n, std::string& err)
    {
        return
            setMem(k.axpy, 0, y, err)
         && setMem(k.axpy, 1, x, err)
         && setDouble(k.axpy, 2, alpha, err)
         && setInt(k.axpy, 3, n, err)
         && launch1D(k.axpy, n, err);
    }


    bool copyVec(cl::cl_mem dst, cl::cl_mem src, int n, std::string& err)
    {
        return
            setMem(k.copy, 0, dst, err)
         && setMem(k.copy, 1, src, err)
         && setInt(k.copy, 2, n, err)
         && launch1D(k.copy, n, err);
    }


    bool scaleVec(cl::cl_mem x, double alpha, int n, std::string& err)
    {
        return
            setMem(k.scale, 0, x, err)
         && setDouble(k.scale, 1, alpha, err)
         && setInt(k.scale, 2, n, err)
         && launch1D(k.scale, n, err);
    }


    // The preconditioner sweeps run as a single work-item: the loop inside is serial by
    // construction (face f depends on faces before it), and more work-items would only
    // race on the dependency chain. `withLower` selects DILU over DIC.
    bool runPreconditioner
    (
        cl::cl_kernel kern,
        cl::cl_mem rD,
        bool withLower,
        const Pattern& p,
        std::string& err
    )
    {
        // argument layout: x/rD, [lower], upper, upperAddr, lowerAddr, nFaces, nRows
        const cl::cl_uint base = withLower ? 2 : 1;

        return
            setMem(kern, 0, rD, err)
         && (!withLower || setMem(kern, 1, d_lower, err))
         && setMem(kern, base, d_upper, err)
         && setMem(kern, base + 1, d_upperAddr, err)
         && setMem(kern, base + 2, d_lowerAddr, err)
         && setInt(kern, base + 3, p.nFaces, err)
         && setInt(kern, base + 4, p.nCells, err)
         && launch1D(kern, 1, err);
    }
}


// * * * * * * * * * * * * * device state * * * * * * * * * * * * * * * * * * //

// Opened once per run, taken away for good the first time it fails. Retrying a device
// that produced a wrong answer on every field would cost more than the accelerator
// ever saves — and would hide the failure.
bool deviceReady(std::string& why)
{
    if (deviceDisabled())
    {
        why = deviceDisabledReason();
        return false;
    }

    if (sharedContext)
    {
        why.clear();
        return true;
    }

    const std::vector<DeviceInfo>& list = devices();
    if (list.empty())
    {
        why = reason();
        disableDevice(why);
        return false;
    }

    const DeviceInfo* chosen = nullptr;
    std::string rejected;
    for (const DeviceInfo& d : list)
    {
        std::string r;
        if (suitable(d, 0, r))
        {
            chosen = &d;
            break;
        }
        rejected += d.name + ": " + r + "; ";
    }

    if (!chosen)
    {
        why = rejected.empty() ? "no suitable device" : rejected;
        disableDevice(why);
        return false;
    }

    Context* c = new Context;
    if (!c->open(*chosen, kernelSource()))
    {
        why = c->error();
        delete c;
        disableDevice(why);
        return false;
    }

    if (!c->selfTest())
    {
        why = c->error();
        c->close();
        delete c;
        disableDevice(why);
        return false;
    }

    sharedContext = c;

    std::string err;
    const bool ok =
        makeKernel(k.amul, kernelNames::amul, err)
     && makeKernel(k.residual, kernelNames::residual, err)
     && makeKernel(k.dot, kernelNames::dot, err)
     && makeKernel(k.axpy, kernelNames::axpy, err)
     && makeKernel(k.xpay, kernelNames::xpay, err)
     && makeKernel(k.copy, kernelNames::copy, err)
     && makeKernel(k.scale, kernelNames::scale, err)
     && makeKernel(k.dicFactorise, kernelNames::dicFactorise, err)
     && makeKernel(k.dicSolve, kernelNames::dicSolve, err)
     && makeKernel(k.diluFactorise, kernelNames::diluFactorise, err)
     && makeKernel(k.diluSolve, kernelNames::diluSolve, err);

    if (!ok)
    {
        why = err;
        disableDevice(why);
        return false;
    }

    why.clear();
    return true;
}


namespace
{
    SolverResult deviceFailure(const std::string& why)
    {
        SolverResult r;
        r.status = DEVICE_FAILED;
        r.note = why;
        disableDevice(why);
        return r;
    }
}


// * * * * * * * * * * * * * CG on the device * * * * * * * * * * * * * * * * //

SolverResult solveCGDevice
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
    result.status = MAX_ITER;

    const int n = p.nCells;
    if (n == 0)
    {
        result.status = CONVERGED;
        return result;
    }

    std::string err;
    if (!deviceReady(err))
    {
        result.status = DEVICE_FAILED;
        result.note = err;
        return result;
    }
    if (!ensureBuffers(diag, upper, lower, p, err))
    {
        return deviceFailure(err);
    }
    if (!crossCheck(psi, diag, upper, lower, p, err))
    {
        return deviceFailure(err);
    }

    const size_t bytes = sizeof(double)*size_t(n);

    if (!writeBuffer(d_x, bytes, psi.data(), err)) return deviceFailure(err);
    if (!writeBuffer(d_s, bytes, source.data(), err)) return deviceFailure(err);

    double bNorm2 = 0;
    if (!dotProduct(d_s, d_s, n, bNorm2, err)) return deviceFailure(err);
    const double normFactor = (std::sqrt(bNorm2) > 0 ? std::sqrt(bNorm2) : 1.0);

    if (!computeResidual(d_s, d_x, d_r, p, err)) return deviceFailure(err);

    double rNorm2 = 0;
    if (!dotProduct(d_r, d_r, n, rNorm2, err)) return deviceFailure(err);
    result.residual = std::sqrt(rNorm2)/normFactor;
    const double initialResidual = result.residual;

    // rD = diag, then factorise in place — once: the factors do not change during the
    // solve, and refactorising every iteration would throw the ILU away.
    if (!writeBuffer(d_rD, bytes, diag.data(), err)) return deviceFailure(err);
    if (!runPreconditioner(k.dicFactorise, d_rD, false, p, err)) return deviceFailure(err);

    if (!copyVec(d_z, d_r, n, err)) return deviceFailure(err);
    if (!runPreconditioner(k.dicSolve, d_z, false, p, err)) return deviceFailure(err);
    if (!copyVec(d_p, d_z, n, err)) return deviceFailure(err);

    double rz = 0;
    if (!dotProduct(d_r, d_z, n, rz, err)) return deviceFailure(err);

    for (int iter = 1; iter <= controls.maxIter; ++iter)
    {
        // v = A*p
        if
        (
            !setMem(k.amul, AM_X, d_p, err)
         || !setMem(k.amul, AM_Y, d_v, err)
         || !setMatrixArgs(k.amul, p, err)
         || !launch1D(k.amul, n, err)
        )
        {
            return deviceFailure(err);
        }

        double pw = 0;
        if (!dotProduct(d_p, d_v, n, pw, err)) return deviceFailure(err);
        if (!(std::fabs(pw) > 0))
        {
            result.status = DIVERGED;
            result.iterations = iter;
            result.note = "CG: zero curvature (singular matrix or disconnected region)";
            return result;
        }

        const double alpha = rz/pw;

        if (!xpay(d_x, d_p, alpha, n, err)) return deviceFailure(err);   // psi += alpha*p
        if (!axpy(d_r, d_v, -alpha, n, err)) return deviceFailure(err);  // r   -= alpha*v

        if (!dotProduct(d_r, d_r, n, rNorm2, err)) return deviceFailure(err);
        result.residual = std::sqrt(rNorm2)/normFactor;
        result.iterations = iter;

        if (result.residual < controls.tolerance && iter >= controls.minIter)
        {
            result.status = CONVERGED;
            break;
        }

        if (!std::isfinite(result.residual))
        {
            result.status = DIVERGED;
            result.note = "CG: residual is not finite";
            break;
        }
        if (result.residual > 1e6*std::max(initialResidual, 1e-300))
        {
            result.status = DIVERGED;
            result.note = "CG: residual growing (matrix not symmetric?)";
            break;
        }

        if (!copyVec(d_z, d_r, n, err)) return deviceFailure(err);
        if (!runPreconditioner(k.dicSolve, d_z, false, p, err)) return deviceFailure(err);

        double rzNew = 0;
        if (!dotProduct(d_r, d_z, n, rzNew, err)) return deviceFailure(err);
        const double beta = rzNew/rz;
        rz = rzNew;

        // p = z + beta*p: scale then add, both in place on the device.
        if (!scaleVec(d_p, beta, n, err)) return deviceFailure(err);
        if (!axpy(d_p, d_z, 1.0, n, err)) return deviceFailure(err);
    }

    // Read the answer back: the caller's psi is the authority, the buffer is scratch.
    if (!readBuffer(d_x, bytes, psi.data(), err)) return deviceFailure(err);
    return result;
}


// * * * * * * * * * * * * * PBiCGStab on the device * * * * * * * * * * * * * //

SolverResult solveBiCGStabDevice
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
    // The same sequence of operations as solveBiCGStabCPU, which is the point: the two
    // must agree iteration for iteration, or a difference in the answer cannot be
    // attributed to rounding.
    SolverResult result;
    result.status = MAX_ITER;

    const int n = p.nCells;
    if (n == 0)
    {
        result.status = CONVERGED;
        return result;
    }

    std::string err;
    if (!deviceReady(err))
    {
        result.status = DEVICE_FAILED;
        result.note = err;
        return result;
    }
    if (!ensureBuffers(diag, upper, lower, p, err))
    {
        return deviceFailure(err);
    }
    if (!crossCheck(psi, diag, upper, lower, p, err))
    {
        return deviceFailure(err);
    }

    const size_t bytes = sizeof(double)*size_t(n);

    // Buffer roles: d_x psi, d_s b, d_r r, d_y rA0 (frozen), d_p pA,
    // d_z yA = M^-1 pA, d_v AyA = A*yA, d_a sA, d_b zA = M^-1 sA, d_c tA = A*zA.
    if (!writeBuffer(d_x, bytes, psi.data(), err)) return deviceFailure(err);
    if (!writeBuffer(d_s, bytes, source.data(), err)) return deviceFailure(err);

    double bNorm2 = 0;
    if (!dotProduct(d_s, d_s, n, bNorm2, err)) return deviceFailure(err);
    const double normFactor = (std::sqrt(bNorm2) > 0 ? std::sqrt(bNorm2) : 1.0);

    if (!computeResidual(d_s, d_x, d_r, p, err)) return deviceFailure(err);
    if (!copyVec(d_y, d_r, n, err)) return deviceFailure(err);   // rA0, never updated

    double rNorm2 = 0;
    if (!dotProduct(d_r, d_r, n, rNorm2, err)) return deviceFailure(err);
    result.residual = std::sqrt(rNorm2)/normFactor;
    const double initialResidual = result.residual;

    if (!writeBuffer(d_rD, bytes, diag.data(), err)) return deviceFailure(err);
    if (!runPreconditioner(k.diluFactorise, d_rD, true, p, err)) return deviceFailure(err);

    double rT0r = 0;
    double alpha = 0;
    double omega = 0;

    for (int iter = 1; iter <= controls.maxIter; ++iter)
    {
        const double rT0rOld = rT0r;
        if (!dotProduct(d_y, d_r, n, rT0r, err)) return deviceFailure(err);

        if (!(std::fabs(rT0r) > 0))
        {
            result.status = DIVERGED;
            result.iterations = iter - 1;
            result.note = "PBiCGStab: breakdown (rA0.rA == 0)";
            return result;
        }

        if (iter == 1)
        {
            if (!copyVec(d_p, d_r, n, err)) return deviceFailure(err);
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
            // pA = rA + beta*(pA - omega*AyA)
            const double beta = (rT0r/rT0rOld)*(alpha/omega);
            if (!axpy(d_p, d_v, -omega, n, err)) return deviceFailure(err);
            if (!scaleVec(d_p, beta, n, err)) return deviceFailure(err);
            if (!axpy(d_p, d_r, 1.0, n, err)) return deviceFailure(err);
        }

        // yA = M^-1 pA, AyA = A*yA
        if (!copyVec(d_z, d_p, n, err)) return deviceFailure(err);
        if (!runPreconditioner(k.diluSolve, d_z, true, p, err)) return deviceFailure(err);
        if
        (
            !setMem(k.amul, AM_X, d_z, err)
         || !setMem(k.amul, AM_Y, d_v, err)
         || !setMatrixArgs(k.amul, p, err)
         || !launch1D(k.amul, n, err)
        )
        {
            return deviceFailure(err);
        }

        double rT0Ay = 0;
        if (!dotProduct(d_y, d_v, n, rT0Ay, err)) return deviceFailure(err);
        if (!(std::fabs(rT0Ay) > 0))
        {
            result.status = DIVERGED;
            result.iterations = iter - 1;
            result.note = "PBiCGStab: breakdown (rA0.A.yA == 0)";
            return result;
        }
        alpha = rT0r/rT0Ay;

        // sA = rA - alpha*AyA
        if (!copyVec(d_a, d_r, n, err)) return deviceFailure(err);
        if (!axpy(d_a, d_v, -alpha, n, err)) return deviceFailure(err);

        double sNorm2 = 0;
        if (!dotProduct(d_a, d_a, n, sNorm2, err)) return deviceFailure(err);
        const double sNorm = std::sqrt(sNorm2)/normFactor;
        result.residual = sNorm;
        result.iterations = iter;

        if (sNorm < controls.tolerance && iter >= controls.minIter)
        {
            if (!xpay(d_x, d_z, alpha, n, err)) return deviceFailure(err);
            result.status = CONVERGED;
            break;
        }

        // zA = M^-1 sA, tA = A*zA
        if (!copyVec(d_b, d_a, n, err)) return deviceFailure(err);
        if (!runPreconditioner(k.diluSolve, d_b, true, p, err)) return deviceFailure(err);
        if
        (
            !setMem(k.amul, AM_X, d_b, err)
         || !setMem(k.amul, AM_Y, d_c, err)
         || !setMatrixArgs(k.amul, p, err)
         || !launch1D(k.amul, n, err)
        )
        {
            return deviceFailure(err);
        }

        double tAtA = 0;
        double tAsA = 0;
        if (!dotProduct(d_c, d_c, n, tAtA, err)) return deviceFailure(err);
        if (!dotProduct(d_c, d_a, n, tAsA, err)) return deviceFailure(err);
        omega = (std::fabs(tAtA) > 0 ? tAsA/tAtA : 0.0);

        // psi += alpha*yA + omega*zA ; rA = sA - omega*tA
        if (!xpay(d_x, d_z, alpha, n, err)) return deviceFailure(err);
        if (!xpay(d_x, d_b, omega, n, err)) return deviceFailure(err);
        if (!copyVec(d_r, d_a, n, err)) return deviceFailure(err);
        if (!axpy(d_r, d_c, -omega, n, err)) return deviceFailure(err);

        double rn2 = 0;
        if (!dotProduct(d_r, d_r, n, rn2, err)) return deviceFailure(err);
        result.residual = std::sqrt(rn2)/normFactor;

        if (!std::isfinite(result.residual))
        {
            result.status = DIVERGED;
            result.note = "PBiCGStab: residual is not finite";
            break;
        }
        // Against the *initial* residual: a method can legitimately grow during the
        // first iterations and still converge.
        if (result.residual > 1e6*std::max(initialResidual, 1e-300))
        {
            result.status = DIVERGED;
            result.note = "PBiCGStab: residual growing";
            break;
        }
    }

    if (!readBuffer(d_x, bytes, psi.data(), err)) return deviceFailure(err);
    return result;
}

} // namespace aerosGpu
} // namespace Foam
