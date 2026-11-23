/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | lduMatrix::solver entry points: aerosPCG and
    \\  /    A nd           | aerosPBiCGStab
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

// Drop-in replacements for OpenFOAM's PCG and PBiCGStab, selected from fvSolution
// like any other solver:
//
//   solvers
//   {
//       p
//       {
//           solver          aerosPCG;
//           preconditioner  DIC;      // accepted and ignored: the accelerator has
//                                     // its own IC(0)/ILU(0), because the
//                                     // factorisation has to exist on the device
//           tolerance       1e-8;
//           relTol          0;
//       }
//   }
//
//   aerosGPU
//   {
//       gpuDevice   -2;      // -2 auto, -1 CPU only, >= 0 device index
//       gpuMinCells 20000;   // below this the transfer cost dominates
//       gpuVerbose  false;
//   }

#ifndef aerosGpuFoam_H
#define aerosGpuFoam_H

#include "lduMatrix.H"

namespace Foam
{

class aerosPCG
:
    public lduMatrix::solver
{
public:

    TypeName("aerosPCG");

    aerosPCG
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const Field<Field<scalar>>& interfaceBouCoeffs,
        const Field<Field<scalar>>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    aerosPCG(const aerosPCG&) = delete;

    virtual ~aerosPCG() {}

    virtual solverPerformance solve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt = 0
    ) const;

    void operator=(const aerosPCG&) = delete;
};


class aerosPBiCGStab
:
    public lduMatrix::solver
{
public:

    TypeName("aerosPBiCGStab");

    aerosPBiCGStab
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const Field<Field<scalar>>& interfaceBouCoeffs,
        const Field<Field<scalar>>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    aerosPBiCGStab(const aerosPBiCGStab&) = delete;

    virtual ~aerosPBiCGStab() {}

    virtual solverPerformance solve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt = 0
    ) const;

    void operator=(const aerosPBiCGStab&) = delete;
};

} // End namespace Foam

#endif
