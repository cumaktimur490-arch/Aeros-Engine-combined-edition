/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | Device enumeration, context/kernel plumbing, and
    \\  /    A nd           | the accuracy self-check that gates the device
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef aerosGpuDevice_H
#define aerosGpuDevice_H

#include <string>
#include <vector>

namespace Foam
{
namespace aerosGpu
{

// A device as the solver cares about it: can it even run a double-precision
// sparse solver, and is it worth it next to the CPU it sits beside.
struct DeviceInfo
{
    int    index;             // global index across platforms
    int    platformIndex;
    int    deviceInPlatform;  // local index, needed to open it
    std::string name;
    std::string vendor;
    std::string driverVersion;
    bool   isGpu;
    bool   isCpu;
    bool   fp64;
    bool   unifiedMemory;
    unsigned long long memoryBytes;
    int    computeUnits;
    int    maxClockMHz;

    DeviceInfo()
    :
        index(-1), platformIndex(-1), deviceInPlatform(-1),
        isGpu(false), isCpu(false), fp64(false), unifiedMemory(false),
        memoryBytes(0), computeUnits(0), maxClockMHz(0)
    {}
};


// Enumerates every device on every platform. Safe without any OpenCL library:
// returns an empty list and fills reason().
const std::vector<DeviceInfo>& devices();

// Why enumeration produced nothing (no library, no platform, no device). Empty
// when devices() is non-empty.
const std::string& reason();

bool available();


// A compiled program bound to one device, plus the queue to run it on.
class Context
{
public:

    Context();
    ~Context();

    Context(const Context&) = delete;
    void operator=(const Context&) = delete;

    // Opens the device and builds `source`. Returns false and fills error() on any
    // failure — never throws, because a solver must be able to fall back to the
    // CPU path.
    bool open(const DeviceInfo& info, const std::string& source);
    void close();

    bool isOpen() const;
    const DeviceInfo& info() const;
    const std::string& error() const;

    // Opaque handles, cast in the .C where the OpenCL types are known.
    void* context() const;
    void* queue() const;
    void* program() const;
    void* kernel(const char* name) const;

    // Verifies that transfers round-trip before anything is trusted: a device that
    // silently corrupts a buffer would otherwise produce a converged-looking wrong
    // field.
    bool selfTest();

private:

    struct Impl;
    Impl* impl_;
};


// Whether the accelerator should use this device at all: fp64 present, not a CPU
// or unified-memory OpenCL runtime, enough memory for the mesh.
bool suitable(const DeviceInfo& info, int nCells, std::string& why);

// One line per device, with the rejection reason where applicable. Used by
// aerosGpuInfo and by gpuVerbose.
std::string describe();

} // namespace aerosGpu
} // namespace Foam

#endif
