/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | aerosGPU
   \\    /   O peration     | Device enumeration, context/kernel plumbing, and
    \\  /    A nd           | the accuracy self-check that gates the device
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "aerosGpuDevice.H"
#include "aerosGpuCl.H"

#include <cstring>
#include <sstream>

namespace Foam
{
namespace aerosGpu
{

namespace
{
    // Enumeration touches driver code; doing it per solver construction shows up
    // as a startup stall on machines with several platforms.
    std::vector<DeviceInfo> deviceList;
    std::string deviceReason;
    bool enumerated = false;

    std::string getStringParam(cl::cl_device_id device, cl::cl_uint param)
    {
        if (!cl::clGetDeviceInfo) return std::string();

        char buf[512];
        std::memset(buf, 0, sizeof(buf));
        if
        (
            cl::clGetDeviceInfo(device, param, sizeof(buf) - 1, buf, nullptr)
            != cl::CL_SUCCESS_
        )
        {
            return std::string();
        }
        return std::string(buf);
    }

    template<class T>
    T getScalarParam(cl::cl_device_id device, cl::cl_uint param)
    {
        T value = T();
        if
        (
            cl::clGetDeviceInfo(device, param, sizeof(T), &value, nullptr)
            != cl::CL_SUCCESS_
        )
        {
            return T();
        }
        return value;
    }

    // Devices of one platform, by the type that platform actually has.
    cl::cl_uint deviceCount(cl::cl_platform_id platform, cl::cl_uint type)
    {
        cl::cl_uint n = 0;
        if
        (
            cl::clGetDeviceIDs(platform, type, 0, nullptr, &n) != cl::CL_SUCCESS_
        )
        {
            return 0;
        }
        return n;
    }
}


const std::vector<DeviceInfo>& devices()
{
    if (enumerated)
    {
        return deviceList;
    }
    enumerated = true;

    if (!cl::load())
    {
        // Not an error: the solvers fall back to the CPU. The reason is surfaced so
        // a user who expected acceleration learns why they did not get it.
        deviceReason = std::string("OpenCL unavailable: ") + cl::libraryStatus();
        return deviceList;
    }

    cl::cl_uint nPlatforms = 0;
    if
    (
        cl::clGetPlatformIDs(0, nullptr, &nPlatforms) != cl::CL_SUCCESS_
     || nPlatforms == 0
    )
    {
        deviceReason =
            "OpenCL library loaded but no platform is installed "
            "(no GPU driver / ICD registered)";
        return deviceList;
    }

    std::vector<cl::cl_platform_id> platforms(nPlatforms);
    cl::clGetPlatformIDs(nPlatforms, platforms.data(), nullptr);

    for (cl::cl_uint p = 0; p < nPlatforms; ++p)
    {
        // GPUs first, then whatever the platform calls its default device (that is
        // where a CPU runtime shows up, and it is enumerated so that it can be
        // reported as rejected instead of silently missing).
        cl::cl_uint n = deviceCount(platforms[p], cl::CL_DEVICE_TYPE_GPU_);
        cl::cl_uint type = cl::CL_DEVICE_TYPE_GPU_;
        if (n == 0)
        {
            n = deviceCount(platforms[p], cl::CL_DEVICE_TYPE_DEFAULT_);
            type = cl::CL_DEVICE_TYPE_DEFAULT_;
        }
        if (n == 0)
        {
            continue;
        }

        std::vector<cl::cl_device_id> ids(n);
        if
        (
            cl::clGetDeviceIDs(platforms[p], type, n, ids.data(), nullptr)
            != cl::CL_SUCCESS_
        )
        {
            continue;
        }

        for (cl::cl_uint d = 0; d < n; ++d)
        {
            DeviceInfo info;
            info.index = int(deviceList.size());
            info.platformIndex = int(p);
            info.deviceInPlatform = int(d);
            info.name = getStringParam(ids[d], cl::CL_DEVICE_NAME_);
            info.vendor = getStringParam(ids[d], cl::CL_DEVICE_VENDOR_);
            info.driverVersion = getStringParam(ids[d], cl::CL_DEVICE_DRIVER_VERSION_);

            const cl::cl_uint t =
                getScalarParam<cl::cl_uint>(ids[d], cl::CL_DEVICE_TYPE_);
            info.isGpu = (t & cl::CL_DEVICE_TYPE_GPU_) != 0;
            info.isCpu = (t & cl::CL_DEVICE_TYPE_CPU_) != 0;
            info.unifiedMemory =
                getScalarParam<cl::cl_uint>
                (
                    ids[d], cl::CL_DEVICE_HOST_UNIFIED_MEMORY_
                ) != 0;
            info.memoryBytes =
                getScalarParam<cl::cl_ulong>(ids[d], cl::CL_DEVICE_GLOBAL_MEM_SIZE_);
            info.computeUnits =
                getScalarParam<cl::cl_uint>(ids[d], cl::CL_DEVICE_MAX_COMPUTE_UNITS_);
            info.maxClockMHz =
                getScalarParam<cl::cl_uint>(ids[d], cl::CL_DEVICE_MAX_CLOCK_FREQ_);

            // FP64 is what the linear algebra needs: a PCG residual in FP32 is not
            // a residual, so a device without it cannot hold a 1e-9 tolerance.
            const std::string ext = getStringParam(ids[d], cl::CL_DEVICE_EXTENSIONS_);
            info.fp64 =
                ext.find("cl_khr_fp64") != std::string::npos
             || ext.find("cl_amd_fp64") != std::string::npos;

            deviceList.push_back(info);
        }
    }

    if (deviceList.empty())
    {
        deviceReason = "OpenCL platforms found but no compute device";
    }

    return deviceList;
}


const std::string& reason()
{
    devices();
    return deviceReason;
}


bool available()
{
    return !devices().empty();
}


bool suitable(const DeviceInfo& info, int nCells, std::string& why)
{
    why.clear();

    if (!info.fp64)
    {
        why = "no double precision (cl_khr_fp64) — an FP32 solver cannot hold "
              "the residual tolerance";
        return false;
    }

    if (info.isCpu || info.unifiedMemory)
    {
        // A CPU OpenCL runtime adds a queue and a copy on top of the same silicon.
        // OpenFOAM's own solver already uses those cores and is faster.
        why = "CPU/unified-memory OpenCL runtime — OpenFOAM's own solver is faster";
        return false;
    }

    if (info.memoryBytes < 512ull*1024*1024)
    {
        why = "less than 512 MB of device memory";
        return false;
    }

    // The accelerator keeps ~15 double vectors of nCells plus the row structure,
    // so the requirement scales with the mesh; 200 bytes/cell is a safe envelope.
    const unsigned long long need = 200ull*1024*1024 + 200ull*uint64_t(nCells > 0 ? nCells : 0);
    if (info.memoryBytes < need)
    {
        std::ostringstream os;
        os << "needs about " << (need/(1024*1024))
           << " MB for this mesh, device has " << (info.memoryBytes/(1024*1024))
           << " MB";
        why = os.str();
        return false;
    }

    if (nCells > 0 && nCells < 20000)
    {
        why = "mesh too small (below 20000 cells) for the transfer cost";
        return false;
    }

    return true;
}


std::string describe()
{
    const std::vector<DeviceInfo>& list = devices();
    if (list.empty())
    {
        // No trailing newline on purpose: callers stream this into Info << and
        // append their own endl. OpenFOAM quotes a string that contains a newline and
        // escapes it, so a stray "\n" here used to come out as
        //   "aerosGPU: ... )\n"
        // in every log line instead of a plain message.
        return std::string("aerosGPU: ") + reason();
    }

    std::ostringstream os;
    for (const DeviceInfo& d : list)
    {
        os << "aerosGPU device " << d.index << ": " << d.name
           << " [" << d.vendor << "] "
           << d.computeUnits << " CU @ " << d.maxClockMHz << " MHz, "
           << (d.memoryBytes/(1024*1024)) << " MB, fp64 "
           << (d.fp64 ? "yes" : "NO") << ", "
           << (d.isGpu ? "GPU" : (d.isCpu ? "CPU" : "other"));

        std::string why;
        if (!suitable(d, 0, why))
        {
            os << " — unusable: " << why;
        }
        os << '\n';
    }

    // One trailing newline is removed for the same reason as above — the caller adds
    // its own. describe() is also printed by aerosGpuInfo, which does not care.
    std::string out = os.str();
    if (!out.empty() && out.back() == '\n') out.pop_back();
    return out;
}


// * * * * * * * * * * * * * Context * * * * * * * * * * * * * * * * * * * * //

struct Context::Impl
{
    DeviceInfo info;
    std::string error;
    cl::cl_device_id device;
    cl::cl_context context;
    cl::cl_command_queue queue;
    cl::cl_program program;

    Impl() : device(nullptr), context(nullptr), queue(nullptr), program(nullptr) {}
};


Context::Context()
:
    impl_(new Impl)
{}


Context::~Context()
{
    close();
    delete impl_;
}


bool Context::open(const DeviceInfo& info, const std::string& source)
{
    close();
    impl_->info = info;
    impl_->error.clear();

    // Enumeration must have run so the indices are meaningful.
    devices();

    if (!cl::load())
    {
        impl_->error = cl::libraryStatus();
        return false;
    }

    cl::cl_uint nPlatforms = 0;
    if
    (
        cl::clGetPlatformIDs(0, nullptr, &nPlatforms) != cl::CL_SUCCESS_
     || nPlatforms == 0
    )
    {
        impl_->error = "no OpenCL platform";
        return false;
    }
    std::vector<cl::cl_platform_id> platforms(nPlatforms);
    cl::clGetPlatformIDs(nPlatforms, platforms.data(), nullptr);

    if (info.platformIndex < 0 || size_t(info.platformIndex) >= platforms.size())
    {
        impl_->error = "platform index out of range";
        return false;
    }

    const cl::cl_platform_id platform = platforms[size_t(info.platformIndex)];
    cl::cl_uint nDevices = deviceCount(platform, cl::CL_DEVICE_TYPE_GPU_);
    cl::cl_uint type = cl::CL_DEVICE_TYPE_GPU_;
    if (nDevices == 0)
    {
        nDevices = deviceCount(platform, cl::CL_DEVICE_TYPE_DEFAULT_);
        type = cl::CL_DEVICE_TYPE_DEFAULT_;
    }
    if (nDevices == 0 || info.deviceInPlatform < 0
        || info.deviceInPlatform >= int(nDevices))
    {
        impl_->error = "device index out of range (device disappeared?)";
        return false;
    }

    std::vector<cl::cl_device_id> ids(nDevices);
    if
    (
        cl::clGetDeviceIDs(platform, type, nDevices, ids.data(), nullptr)
        != cl::CL_SUCCESS_
    )
    {
        impl_->error = "clGetDeviceIDs failed";
        return false;
    }
    impl_->device = ids[size_t(info.deviceInPlatform)];

    cl::cl_int err = cl::CL_SUCCESS_;
    impl_->context = cl::clCreateContext(nullptr, 1, &impl_->device, nullptr, nullptr, &err);
    if (err != cl::CL_SUCCESS_ || !impl_->context)
    {
        impl_->error = "clCreateContext failed (" + std::to_string(err) + ")";
        impl_->context = nullptr;
        return false;
    }

    impl_->queue = cl::clCreateCommandQueue(impl_->context, impl_->device, 0, &err);
    if (err != cl::CL_SUCCESS_ || !impl_->queue)
    {
        impl_->error = "clCreateCommandQueue failed (" + std::to_string(err) + ")";
        impl_->queue = nullptr;
        close();
        return false;
    }

    const char* src = source.c_str();
    impl_->program = cl::clCreateProgramWithSource(impl_->context, 1, &src, nullptr, &err);
    if (err != cl::CL_SUCCESS_ || !impl_->program)
    {
        impl_->error = "clCreateProgramWithSource failed";
        impl_->program = nullptr;
        close();
        return false;
    }

    // -cl-fast-relaxed-math is deliberately NOT passed: it allows reassociation that
    // breaks the reduction order the residual depends on, and the whole point of
    // this library is that the device answer matches the CPU one.
    err = cl::clBuildProgram(impl_->program, 1, &impl_->device, "-cl-mad-enable", nullptr, nullptr);
    if (err != cl::CL_SUCCESS_)
    {
        std::size_t logSize = 0;
        cl::clGetProgramBuildInfo
        (
            impl_->program, impl_->device, cl::CL_PROGRAM_BUILD_LOG_, 0, nullptr, &logSize
        );
        std::string log(logSize ? logSize : 1, '\0');
        if (logSize)
        {
            cl::clGetProgramBuildInfo
            (
                impl_->program, impl_->device, cl::CL_PROGRAM_BUILD_LOG_,
                logSize, &log[0], nullptr
            );
        }
        impl_->error = "kernel build failed: " + log;
        close();
        return false;
    }

    return true;
}


void Context::close()
{
    if (impl_->program) { cl::clReleaseProgram(impl_->program); impl_->program = nullptr; }
    if (impl_->queue) { cl::clReleaseCommandQueue(impl_->queue); impl_->queue = nullptr; }
    if (impl_->context) { cl::clReleaseContext(impl_->context); impl_->context = nullptr; }
    impl_->device = nullptr;
}


bool Context::isOpen() const
{
    return impl_->context != nullptr && impl_->program != nullptr;
}


const DeviceInfo& Context::info() const { return impl_->info; }
const std::string& Context::error() const { return impl_->error; }
void* Context::context() const { return impl_->context; }
void* Context::queue() const { return impl_->queue; }
void* Context::program() const { return impl_->program; }


void* Context::kernel(const char* name) const
{
    if (!impl_->program) return nullptr;

    cl::cl_int err = cl::CL_SUCCESS_;
    cl::cl_kernel k = cl::clCreateKernel(impl_->program, name, &err);
    if (err != cl::CL_SUCCESS_)
    {
        return nullptr;
    }
    return k;
}


bool Context::selfTest()
{
    if (!isOpen())
    {
        impl_->error = "context is not open";
        return false;
    }

    // A round trip catches a device that silently drops transfers. The solvers do
    // a stronger check of their own (comparing a sparse product with the CPU), but
    // this one costs nothing and runs before any of that.
    unsigned int pattern[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    cl::cl_int err = cl::CL_SUCCESS_;
    cl::cl_mem buf =
        cl::clCreateBuffer
        (
            impl_->context,
            cl::CL_MEM_READ_WRITE_ | cl::CL_MEM_COPY_HOST_PTR_,
            sizeof(pattern), pattern, &err
        );
    if (err != cl::CL_SUCCESS_ || !buf)
    {
        impl_->error = "self-test: clCreateBuffer failed";
        return false;
    }

    unsigned int back[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    err = cl::clEnqueueReadBuffer
    (
        impl_->queue, buf, cl::CL_TRUE_, 0, sizeof(back), back, 0, nullptr, nullptr
    );
    cl::clReleaseMemObject(buf);

    if (err != cl::CL_SUCCESS_)
    {
        impl_->error = "self-test: clEnqueueReadBuffer failed";
        return false;
    }

    for (int i = 0; i < 8; ++i)
    {
        if (back[i] != pattern[i])
        {
            impl_->error = "self-test: the device returned corrupted data";
            return false;
        }
    }

    return true;
}

} // namespace aerosGpu
} // namespace Foam
