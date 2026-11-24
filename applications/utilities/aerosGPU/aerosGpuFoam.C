/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | lduMatrix::solver entry points
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

// dictionary.H first: lduMatrix.H holds a dictionary by value but only
// forward-declares the type, so it has to be complete before it is included.
#include "dictionary.H"

#include "aerosGpuFoam.H"
#include "aerosGpuDevice.H"   // describe()
#include "aerosGpuFoamCore.H"
#include "aerosGpuMatrix.H"
#include "aerosGpuSolver.H"

namespace Foam
{

namespace
{

using aerosGpu::Settings;
using aerosGpu::Assembled;
using aerosGpu::Outcome;


Settings readSettings(const dictionary& controls)
{
    Settings s;

    if (controls.found("aerosGPU") && controls.isDict("aerosGPU"))
    {
        const dictionary& d = controls.subDict("aerosGPU");
        d.readIfPresent("gpuDevice", s.device);
        d.readIfPresent("gpuMinCells", s.minCells);
        d.readIfPresent("gpuVerbose", s.verbose);
    }

    return s;
}


// The accelerator indexes with int (device buffers hold 32-bit indices), while
// OpenFOAM's label is 64-bit. The conversion is cached and refilled in place, so it
// costs one linear pass per solve — negligible next to the dozens of sparse products a
// Krylov solve performs — and the mesh size is checked rather than silently truncated.
const aerosGpu::Pattern& patternFor(const lduMatrix& matrix)
{
    const UList<label>& up = matrix.lduAddr().upperAddr();
    const UList<label>& lo = matrix.lduAddr().lowerAddr();

    if (matrix.diag().size() > label(2147483647))
    {
        FatalErrorInFunction
            << "mesh has " << matrix.diag().size() << " cells; the accelerator indexes"
            << " with 32-bit integers (limit 2^31-1)" << exit(FatalError);
    }

    static std::vector<int> upInt;
    static std::vector<int> loInt;

    upInt.resize(up.size());
    loInt.resize(lo.size());
    for (label f = 0; f < up.size(); ++f)
    {
        upInt[size_t(f)] = int(up[f]);
        loInt[size_t(f)] = int(lo[f]);
    }

    return aerosGpu::pattern
    (
        upInt.data(), loInt.data(),
        int(matrix.diag().size()), int(up.size()),
        !matrix.hasLower()
    );
}


solverPerformance toPerformance
(
    const Outcome& o,
    const word& name,
    const word& fieldName,
    scalarField& psi,
    const Assembled& a
)
{
    forAll(psi, cell)
    {
        psi[cell] = a.x[size_t(cell)];
    }

    solverPerformance perf(name, fieldName);
    perf.initialResidual() = o.initialResidual;
    perf.finalResidual() = o.finalResidual;
    perf.nIterations() = o.iterations;

    if (!o.note.empty())
    {
        // Always printed, whatever gpuVerbose says: this is the line that explains why
        // an "accelerated" run is running on the CPU.
        Info<< "aerosGPU (" << name << "): " << o.note << endl;
    }

    return perf;
}


// OpenFOAM reports a solve as "<preconditioner><solver>", e.g. "DICPCG". Ours used to
// concatenate the preconditioner with the registered type name and produce
// "DICaerosPCG", which reads like a preconditioner named "DICaeros". Keep the user's
// preconditioner and append the name of the solver actually doing the work.
word solverName(const dictionary& controls, const char* base)
{
    return lduMatrix::preconditioner::getName(controls) + base;
}


void announce(const Settings& s, const lduMatrix& matrix)
{
    if (s.verbose)
    {
        Info<< aerosGpu::describe();
    }

    // A user who set gpuDevice to a specific card and sees it ignored deserves the
    // number that explains it.
    if (s.device >= 0 && matrix.diag().size() < s.minCells)
    {
        Info<< "aerosGPU: " << matrix.diag().size() << " cells is below gpuMinCells ("
            << s.minCells << "); staying on the CPU" << endl;
    }
}

} // namespace


// * * * * * * * * * * * * * * * aerosPCG * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(aerosPCG, 0);

lduMatrix::solver::addsymMatrixConstructorToTable<aerosPCG>
    addaerosPCGSymMatrixConstructorToTable_;


aerosPCG::aerosPCG
(
    const word& fieldName,
    const lduMatrix& matrix,
    const Field<Field<scalar>>& interfaceBouCoeffs,
    const Field<Field<scalar>>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


solverPerformance aerosPCG::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    (void)cmpt;

    const Settings s = readSettings(controlDict_);
    announce(s, matrix_);

    // Fill in the fields, then attach the cached pattern: the addressing conversion
    // belongs to patternFor(), the coefficient copy to assemble().
    aerosGpu::Assembled a;
    a.diag.assign(matrix_.diag().begin(), matrix_.diag().end());
    a.upper.assign(matrix_.upper().begin(), matrix_.upper().end());
    a.lower = a.upper;                     // symmetric: one coefficient array
    a.b.assign(source.begin(), source.end());
    a.x.assign(psi.begin(), psi.end());
    a.pattern = &patternFor(matrix_);

    const double initRes = aerosGpu::initialResidual(a);
    const aerosGpu::SolverControls c =
        aerosGpu::controlsFrom(s, int(maxIter_), int(minIter_), tolerance_, relTol_);

    const aerosGpu::SolverResult r =
        aerosGpu::solveCG(a.x, a.b, a.diag, a.upper, a.lower, *a.pattern, c);

    const Outcome o =
        aerosGpu::finish(r, initRes, tolerance_, relTol_, a.x, a);

    return toPerformance
    (
        o,
        solverName(controlDict_, "PCG"),
        fieldName_,
        psi,
        a
    );
}


// * * * * * * * * * * * * * * aerosPBiCGStab * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(aerosPBiCGStab, 0);

lduMatrix::solver::addasymMatrixConstructorToTable<aerosPBiCGStab>
    addaerosPBiCGStabAsymMatrixConstructorToTable_;


aerosPBiCGStab::aerosPBiCGStab
(
    const word& fieldName,
    const lduMatrix& matrix,
    const Field<Field<scalar>>& interfaceBouCoeffs,
    const Field<Field<scalar>>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


solverPerformance aerosPBiCGStab::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    (void)cmpt;

    const Settings s = readSettings(controlDict_);
    announce(s, matrix_);

    const bool hasLower = matrix_.hasLower();

    aerosGpu::Assembled a;
    a.diag.assign(matrix_.diag().begin(), matrix_.diag().end());
    a.upper.assign(matrix_.upper().begin(), matrix_.upper().end());
    if (hasLower)
    {
        a.lower.assign(matrix_.lower().begin(), matrix_.lower().end());
    }
    else
    {
        a.lower = a.upper;
    }
    a.b.assign(source.begin(), source.end());
    a.x.assign(psi.begin(), psi.end());
    a.pattern = &patternFor(matrix_);

    const double initRes = aerosGpu::initialResidual(a);
    const aerosGpu::SolverControls c =
        aerosGpu::controlsFrom(s, int(maxIter_), int(minIter_), tolerance_, relTol_);

    const aerosGpu::SolverResult r =
        aerosGpu::solveBiCGStab(a.x, a.b, a.diag, a.upper, a.lower, *a.pattern, c);

    const Outcome o =
        aerosGpu::finish(r, initRes, tolerance_, relTol_, a.x, a);

    return toPerformance
    (
        o,
        solverName(controlDict_, "PBiCGStab"),
        fieldName_,
        psi,
        a
    );
}


// ************************************************************************* //

} // End namespace Foam
