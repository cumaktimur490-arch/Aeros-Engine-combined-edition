/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | aerosGpuInfo: what the accelerator can use here
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

// Prints the device table and the dispatcher's verdict. Run this before editing
// fvSolution: the table says which devices exist and why each unusable one was
// rejected, and the verdict says what a solve would actually do.

#include "aerosGpuDevice.H"
#include "aerosGpuSolver.H"

#include <cstdio>
#include <string>

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    std::printf("%s", Foam::aerosGpu::describe().c_str());

    std::string why;
    const bool ready = Foam::aerosGpu::deviceReady(why);

    if (ready)
    {
        std::printf("aerosGPU: device ready — solves will run on the GPU\n");
    }
    else
    {
        std::printf("aerosGPU: CPU only — %s\n", why.c_str());
    }

    return ready ? 0 : 1;
}
