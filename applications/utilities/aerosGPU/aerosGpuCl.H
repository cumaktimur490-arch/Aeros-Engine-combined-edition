/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | OpenCL loaded at run time, so the library builds
    \\  /    A nd           | and links on machines with no OpenCL SDK.
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

// Minimal OpenCL surface, resolved with dlopen/dlsym.
//
// Why not <CL/cl.h> and -lOpenCL: that makes the library refuse to load on any
// machine without a driver — which is exactly the machine where the CPU fallback
// matters. Resolving the entry points here means libaerosGpuSolvers.so loads
// everywhere, and reports why it is not accelerating instead of not starting.

#ifndef aerosGpuCl_H
#define aerosGpuCl_H

#include <cstddef>

namespace Foam
{
namespace aerosGpu
{
namespace cl
{

typedef int                 cl_int;
typedef unsigned int        cl_uint;
typedef unsigned long long  cl_ulong;
typedef unsigned int        cl_bool;
typedef unsigned long       cl_bitfield;

struct _cl_platform_id;
struct _cl_device_id;
struct _cl_context;
struct _cl_command_queue;
struct _cl_mem;
struct _cl_program;
struct _cl_kernel;
struct _cl_event;

typedef _cl_platform_id*    cl_platform_id;
typedef _cl_device_id*      cl_device_id;
typedef _cl_context*        cl_context;
typedef _cl_command_queue*  cl_command_queue;
typedef _cl_mem*            cl_mem;
typedef _cl_program*        cl_program;
typedef _cl_kernel*         cl_kernel;
typedef _cl_event*          cl_event;

const cl_int CL_SUCCESS_                  = 0;

const cl_uint CL_DEVICE_TYPE_DEFAULT_     = (1 << 0);
const cl_uint CL_DEVICE_TYPE_CPU_         = (1 << 1);
const cl_uint CL_DEVICE_TYPE_GPU_         = (1 << 2);

const cl_uint CL_MEM_READ_WRITE_          = (1 << 0);
const cl_uint CL_MEM_COPY_HOST_PTR_       = (1 << 5);
const cl_uint CL_MEM_READ_ONLY_           = (1 << 2);

const cl_bool CL_TRUE_                    = 1;

const cl_uint CL_DEVICE_NAME_             = 0x102B;
const cl_uint CL_DEVICE_VENDOR_           = 0x102C;
const cl_uint CL_DEVICE_TYPE_             = 0x1000;
const cl_uint CL_DEVICE_GLOBAL_MEM_SIZE_  = 0x101F;
const cl_uint CL_DEVICE_MAX_COMPUTE_UNITS_= 0x1002;
const cl_uint CL_DEVICE_MAX_CLOCK_FREQ_   = 0x100C;
const cl_uint CL_DEVICE_DRIVER_VERSION_   = 0x102D;
const cl_uint CL_DEVICE_EXTENSIONS_       = 0x1030;
const cl_uint CL_DEVICE_HOST_UNIFIED_MEMORY_ = 0x1035;

const cl_uint CL_PROGRAM_BUILD_LOG_       = 0x1183;


// Resolved entry points. Null when load() failed or the installed runtime does not
// export them; nothing here is called before load() returns true.
extern cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
extern cl_int (*clGetDeviceIDs)(cl_platform_id, cl_uint, cl_uint, cl_device_id*, cl_uint*);
extern cl_int (*clGetDeviceInfo)(cl_device_id, cl_uint, std::size_t, void*, std::size_t*);
extern cl_context (*clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
extern cl_int (*clReleaseContext)(cl_context);
extern cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*);
extern cl_int (*clReleaseCommandQueue)(cl_command_queue);
extern cl_int (*clFinish)(cl_command_queue);
extern cl_mem (*clCreateBuffer)(cl_context, cl_bitfield, std::size_t, void*, cl_int*);
extern cl_int (*clReleaseMemObject)(cl_mem);
extern cl_int (*clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t, const void*, cl_uint, const cl_event*, cl_event*);
extern cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t, void*, cl_uint, const cl_event*, cl_event*);
extern cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**, const std::size_t*, cl_int*);
extern cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
extern cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, std::size_t, void*, std::size_t*);
extern cl_int (*clReleaseProgram)(cl_program);
extern cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*);
extern cl_int (*clReleaseKernel)(cl_kernel);
extern cl_int (*clSetKernelArg)(cl_kernel, cl_uint, std::size_t, const void*);
extern cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const std::size_t*, const std::size_t*, const std::size_t*, cl_uint, const cl_event*, cl_event*);
extern cl_int (*clGetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_uint, std::size_t, void*, std::size_t*);


// Loads the library once. Idempotent; false when no usable OpenCL is present.
bool load();
bool loaded();

// Human-readable outcome of the last load() attempt, e.g.
// "loaded libOpenCL.so.1" or "no OpenCL library (tried ...)".
const char* libraryStatus();

} // namespace cl
} // namespace aerosGpu
} // namespace Foam

#endif
