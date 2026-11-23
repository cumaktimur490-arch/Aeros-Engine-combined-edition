/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | OpenCL entry points, resolved with dlopen at
    \\  /    A nd           | run time (no link-time dependency on libOpenCL)
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "aerosGpuCl.H"

#include <dlfcn.h>

#include <string>

namespace Foam
{
namespace aerosGpu
{
namespace cl
{

// Definitions (no `extern` here — that would declare them again and leave the
// symbols undefined at link time, which is a confusing way to find this out).
cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
cl_int (*clGetDeviceIDs)(cl_platform_id, cl_uint, cl_uint, cl_device_id*, cl_uint*) = nullptr;
cl_int (*clGetDeviceInfo)(cl_device_id, cl_uint, std::size_t, void*, std::size_t*) = nullptr;
cl_context (*clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*) = nullptr;
cl_int (*clReleaseContext)(cl_context) = nullptr;
cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*) = nullptr;
cl_int (*clReleaseCommandQueue)(cl_command_queue) = nullptr;
cl_int (*clFinish)(cl_command_queue) = nullptr;
cl_mem (*clCreateBuffer)(cl_context, cl_bitfield, std::size_t, void*, cl_int*) = nullptr;
cl_int (*clReleaseMemObject)(cl_mem) = nullptr;
cl_int (*clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t, const void*, cl_uint, const cl_event*, cl_event*) = nullptr;
cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t, void*, cl_uint, const cl_event*, cl_event*) = nullptr;
cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**, const std::size_t*, cl_int*) = nullptr;
cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*) = nullptr;
cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, std::size_t, void*, std::size_t*) = nullptr;
cl_int (*clReleaseProgram)(cl_program) = nullptr;
cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*) = nullptr;
cl_int (*clReleaseKernel)(cl_kernel) = nullptr;
cl_int (*clSetKernelArg)(cl_kernel, cl_uint, std::size_t, const void*) = nullptr;
cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const std::size_t*, const std::size_t*, const std::size_t*, cl_uint, const cl_event*, cl_event*) = nullptr;
cl_int (*clGetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_uint, std::size_t, void*, std::size_t*) = nullptr;


namespace
{
    void* handle = nullptr;
    std::string statusText = "not loaded";
}


const char* libraryStatus()
{
    return statusText.c_str();
}


bool loaded()
{
    return handle != nullptr;
}


bool load()
{
    if (handle) return true;

    // The ICD loader first (what a normal Linux install provides), then vendor
    // runtimes for machines where only the driver is installed.
    static const char* candidates[] =
    {
        "libOpenCL.so.1", "libOpenCL.so", "libOpenCL.so.1.0.0",
        "libcl.so.3", "libcl.so",
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
        "/usr/local/cuda/lib64/libOpenCL.so",
        nullptr
    };

    std::string tried;
    for (int i = 0; candidates[i]; ++i)
    {
        handle = ::dlopen(candidates[i], RTLD_NOW | RTLD_GLOBAL);
        if (handle)
        {
            statusText = std::string("loaded ") + candidates[i];
            break;
        }
        tried += std::string(tried.empty() ? "" : ", ") + candidates[i];
    }

    if (!handle)
    {
        statusText = "no OpenCL library (tried " + tried + ")";
        return false;
    }

    bool ok = true;

#define AEROS_CL_LOAD(fn)                                                     \
    fn = reinterpret_cast<decltype(fn)>(::dlsym(handle, #fn));                 \
    if (!fn) ok = false;

    AEROS_CL_LOAD(clGetPlatformIDs);
    AEROS_CL_LOAD(clGetDeviceIDs);
    AEROS_CL_LOAD(clGetDeviceInfo);
    AEROS_CL_LOAD(clCreateContext);
    AEROS_CL_LOAD(clReleaseContext);
    AEROS_CL_LOAD(clCreateCommandQueue);
    AEROS_CL_LOAD(clReleaseCommandQueue);
    AEROS_CL_LOAD(clFinish);
    AEROS_CL_LOAD(clCreateBuffer);
    AEROS_CL_LOAD(clReleaseMemObject);
    AEROS_CL_LOAD(clEnqueueWriteBuffer);
    AEROS_CL_LOAD(clEnqueueReadBuffer);
    AEROS_CL_LOAD(clCreateProgramWithSource);
    AEROS_CL_LOAD(clBuildProgram);
    AEROS_CL_LOAD(clGetProgramBuildInfo);
    AEROS_CL_LOAD(clReleaseProgram);
    AEROS_CL_LOAD(clCreateKernel);
    AEROS_CL_LOAD(clReleaseKernel);
    AEROS_CL_LOAD(clSetKernelArg);
    AEROS_CL_LOAD(clEnqueueNDRangeKernel);
    AEROS_CL_LOAD(clGetKernelWorkGroupInfo);

#undef AEROS_CL_LOAD

    if (!ok)
    {
        ::dlclose(handle);
        handle = nullptr;
        statusText = "libOpenCL found but incomplete (missing entry points)";
        return false;
    }

    return true;
}

} // namespace cl
} // namespace aerosGpu
} // namespace Foam
