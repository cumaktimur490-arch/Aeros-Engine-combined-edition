// ============================================================================
// МОСТ Aeros <-> FluidX3D 3.9 — реализация.
//
// Правило этого файла: всё, что можно проверить без OpenCL-устройства, лежит
// ВНЕ #ifdef AEROS_WITH_FX3D. Отображение индексов и геометрии — это логика,
// которая ошибается тихо (перепутанный порядок индексов даёт правдоподобное,
// но чужое поле), поэтому она обязана собираться и тестироваться на машине без
// единого устройства. Вызовы самого решателя — под флагом.
// ============================================================================

#include "fx3d_backend.h"

#include <algorithm>
#include <cstdio>
#include <dlfcn.h>
#include <cstring>
#include <limits>

#ifdef AEROS_WITH_FX3D
#include "lbm.hpp"   // FluidX3D: LBM, Memory_Container, TYPE_S/TYPE_E/TYPE_F, Info, Device_Info
#endif

namespace aerosFx3d
{

// Сверка с исходниками FluidX3D: если он когда-нибудь переименует типы границ,
// это должна быть ошибка сборки, а не тихо другой флаг.
#ifdef AEROS_WITH_FX3D
static_assert(TYPE_S == (unsigned char)TYPE_S, "TYPE_S changed in FluidX3D");
static_assert(TYPE_E == (unsigned char)TYPE_E, "TYPE_E changed in FluidX3D");
static_assert(TYPE_F == (unsigned char)TYPE_F, "TYPE_F changed in FluidX3D");
#endif


// * * * * * * * * * * * * * отображение геометрии * * * * * * * * * * * * * //

std::size_t mapGeometry
(
    int nx, int ny, int nz,
    const std::vector<int>& voxels,
    std::vector<unsigned char>& flags,
    int flowAxis,
    int flowSign,
    bool periodicX
)
{
    if (nx <= 0 || ny <= 0 || nz <= 0) return 0;

    const std::size_t N = std::size_t(nx)*std::size_t(ny)*std::size_t(nz);
    flags.assign(N, 0);

    // Сколько вокселей реально можно прочитать: массив Aeros может быть меньше
    // сетки (тогда остальное — свободный поток), но не должен читаться за
    // своими границами.
    const std::size_t nVox = std::min(N, voxels.size());

    std::size_t solid = 0;
    const int ox = std::max(0, std::min(nx - 1, flowSign > 0 ? 0 : nx - 1));

    for (int z = 0; z < nz; ++z)
    {
        for (int y = 0; y < ny; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                const std::size_t a = aerosIndex(nx, ny, x, y, z);

                unsigned char f = 0;

                if (a < nVox && voxels[a] != 0)
                {
                    f = TYPE_S;
                    ++solid;
                }
                else if (flowAxis == 0 && !periodicX && x == ox)
                {
                    // Вход только на наветренной грани и только если по этой оси
                    // нет периодичности: иначе вход и периодика боролись бы за
                    // одну и ту же плоскость.
                    f = TYPE_E;
                }

                flags[fx3dIndex(nx, ny, x, y, z)] = f;
            }
        }
    }

    return solid;
}


void mapInletVelocity
(
    int nx, int ny, int nz,
    int flowAxis,
    int flowSign,
    float uLB,
    std::vector<float>& u
)
{
    const std::size_t N = std::size_t(std::max(nx, 0))*std::size_t(std::max(ny, 0))
                        * std::size_t(std::max(nz, 0));
    u.assign(3*N, 0.0f);

    const float v = (flowSign >= 0 ? uLB : -uLB);

    // Вся жидкость стартует с однородным потоком: это то же начальное условие,
    // что в initLBM, и оно снимает вопрос «чей старт лучше».
    for (std::size_t n = 0; n < N; ++n)
    {
        if (flowAxis == 0)      u[0*N + n] = v;
        else if (flowAxis == 1) u[1*N + n] = v;
        else                    u[2*N + n] = v;
    }
}


std::size_t estimateDeviceBytes(std::size_t cells, bool unifiedMemory)
{
    // 93 байта/ячейку на дискретном GPU, 55 — на CPU/iGPU: у FluidX3D v3.9
    // под D3Q19+FP16S так (README, раздел про память).
    return cells * (unifiedMemory ? 55u : 93u);
}


// * * * * * * * * * * * * * устройство * * * * * * * * * * * * * * * * * * * //

#ifdef AEROS_WITH_FX3D

struct Backend::Impl
{
    LBM* lbm = nullptr;
    int nx = 0, ny = 0, nz = 0;
    unsigned long long t = 0;
    std::vector<float> scratch;
};

Backend::Backend() : impl_(new Impl) {}

Backend::~Backend()
{
    end();
    delete impl_;
}

bool available(std::string* why)
{
    // ВАЖНО, и это выяснилось по исходникам, а не по документации:
    // get_devices() в opencl.hpp заканчивается вызовом exit(1), если устройств
    // нет. Это не исключение — процесс просто завершается. Поэтому try/catch
    // здесь бесполезен, и до FluidX3D нельзя даже дотянуться, не проверив
    // OpenCL снаружи. Проверяем сами: dlopen + clGetPlatformIDs — ровно то,
    // что делает его ICD-загрузчик, но без права убить приложение.
    void* lib = nullptr;
    static const char* candidates[] =
    {
        "libOpenCL.so.1", "libOpenCL.so", "libOpenCL.so.1.0.0",
        "libcl.so.3", "libcl.so",
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
        "/usr/local/cuda/lib64/libOpenCL.so",
        nullptr
    };

    std::string tried;
    for (int i = 0; candidates[i] && !lib; ++i)
    {
        lib = ::dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (!lib) tried += std::string(tried.empty() ? "" : ", ") + candidates[i];
    }

    if (!lib)
    {
        if (why) *why = "OpenCL unavailable: no library (tried " + tried + ")";
        return false;
    }

    typedef int (*GetPlatformIDs)(unsigned, void**, unsigned*);
    GetPlatformIDs getPlatforms =
        reinterpret_cast<GetPlatformIDs>(::dlsym(lib, "clGetPlatformIDs"));

    if (!getPlatforms)
    {
        ::dlclose(lib);
        if (why) *why = "OpenCL library loaded but exports no clGetPlatformIDs";
        return false;
    }

    unsigned n = 0;
    const int rc = getPlatforms(0, nullptr, &n);
    ::dlclose(lib);

    if (rc != 0 || n == 0)
    {
        if (why) *why = "OpenCL library loaded but no platform is installed "
                        "(no driver / ICD registered)";
        return false;
    }
    return true;
}


bool describeDevice(DeviceSummary* out, std::string* why)
{
    // Та же ловушка, что и в begin(): get_devices() внутри вызывает exit(1),
    // если устройств нет. Без этой проверки «просто посмотреть, что за
    // устройство» завершало бы приложение — проверено запуском, не по
    // документации.
    {
        std::string probeWhy;
        if (!available(&probeWhy))
        {
            if (why) *why = probeWhy;
            return false;
        }
    }

    try
    {
        const std::vector<Device_Info> infos = get_devices(true);
        if (infos.empty())
        {
            if (why) *why = "OpenCL unavailable: no device (no driver or no ICD)";
            return false;
        }

        const Device_Info& d = infos[0];
        if (out)
        {
            out->name = d.name;
            // В Device_Info платформы нет — есть вендор; для журнала этого
            // достаточно, а выдумывать поле хуже, чем показать мало.
            out->platform = d.vendor;
            // Info::memory у FluidX3D в МЕГАБАЙТАХ, а не в байтах.
            out->memoryBytes = std::size_t(d.memory)*1024u*1024u;
            const std::size_t perCell = d.uses_ram ? 55u : 93u;
            // Половина памяти: вторая нужна приложению и драйверу.
            out->cells = out->memoryBytes / (2*perCell);
        }
        return true;
    }
    catch (const std::exception& e)
    {
        if (why) *why = std::string("OpenCL probe failed: ") + e.what();
        return false;
    }
}


bool Backend::begin(int nx, int ny, int nz, const Params& p, std::string* why)
{
    end();

    if (nx <= 0 || ny <= 0 || nz <= 0)
    {
        if (why) *why = "fx3d: empty domain";
        return false;
    }

    // Без этой проверки следующий new LBM(...) увёл бы процесс в exit(1) из
    // get_devices(). Приложение получило бы не «нет ускорения», а завершение.
    {
        std::string probeWhy;
        if (!available(&probeWhy))
        {
            if (why) *why = std::string("fx3d: ") + probeWhy;
            return false;
        }
    }

    try
    {
        const uint N = (uint)(nx*ny*nz);
        impl_->lbm = new LBM
        (
            (uint)nx, (uint)ny, (uint)nz,
            p.nuLB,
            p.bodyForceX, p.bodyForceY, p.bodyForceZ,
            0u            // частиц нет: они не нужны мосту, а память занимают
        );
        impl_->lbm->flags.reset(0u);
        impl_->nx = nx; impl_->ny = ny; impl_->nz = nz;
        impl_->t = 0;

        (void)N;
        return true;
    }
    catch (const std::exception& e)
    {
        if (impl_->lbm) { delete impl_->lbm; impl_->lbm = nullptr; }
        if (why) *why = std::string("fx3d: ") + e.what();
        return false;
    }
}


bool Backend::running() const
{
    return impl_ && impl_->lbm != nullptr;
}


bool Backend::step(int steps, std::string* why)
{
    if (!running())
    {
        if (why) *why = "fx3d: not initialised";
        return false;
    }
    if (steps <= 0) return true;

    try
    {
        impl_->lbm->run((ulong)steps);
        impl_->t += (unsigned long long)steps;
        return true;
    }
    catch (const std::exception& e)
    {
        if (why) *why = std::string("fx3d step failed: ") + e.what();
        return false;
    }
}


bool Backend::readVelocity(std::vector<float>& ux, std::vector<float>& uy,
                           std::vector<float>& uz, std::string* why)
{
    if (!running())
    {
        if (why) *why = "fx3d: not initialised";
        return false;
    }

    const std::size_t N = std::size_t(impl_->nx)*std::size_t(impl_->ny)
                        * std::size_t(impl_->nz);
    ux.assign(N, 0.0f); uy.assign(N, 0.0f); uz.assign(N, 0.0f);

    try
    {
        LBM& lbm = *impl_->lbm;
        lbm.u.read_from_device();   // ровно то, что видит сам FluidX3D

        for (std::size_t n = 0; n < N; ++n)
        {
            ux[n] = (float)lbm.u.x[n];
            uy[n] = (float)lbm.u.y[n];
            uz[n] = (float)lbm.u.z[n];
        }
        return true;
    }
    catch (const std::exception& e)
    {
        if (why) *why = std::string("fx3d read failed: ") + e.what();
        return false;
    }
}


bool Backend::readDensity(std::vector<float>& rho, std::string* why)
{
    if (!running())
    {
        if (why) *why = "fx3d: not initialised";
        return false;
    }

    const std::size_t N = std::size_t(impl_->nx)*std::size_t(impl_->ny)
                        * std::size_t(impl_->nz);
    rho.assign(N, 1.0f);

    try
    {
        impl_->lbm->rho.read_from_device();
        for (std::size_t n = 0; n < N; ++n) rho[n] = (float)impl_->lbm->rho[n];
        return true;
    }
    catch (const std::exception& e)
    {
        if (why) *why = std::string("fx3d read failed: ") + e.what();
        return false;
    }
}


unsigned long long Backend::steps() const
{
    return impl_ ? impl_->t : 0ull;
}


void Backend::end()
{
    if (!impl_) return;
    if (impl_->lbm)
    {
        delete impl_->lbm;
        impl_->lbm = nullptr;
    }
    impl_->t = 0;
}

#else // AEROS_WITH_FX3D

// Сборка без FluidX3D: модуль обязан оставаться рабочим, чтобы тест отображения
// собирался на любой машине, а приложение получало внятную причину отказа, а не
// ошибку линковки.

struct Backend::Impl { int unused = 0; };

Backend::Backend() : impl_(new Impl) {}
Backend::~Backend() { delete impl_; }

static const char* kNoFx3d =
    "fx3d: this binary was built without FluidX3D (AEROS_WITH_FX3D); "
    "the Aeros LBM solver is used instead";

bool available(std::string* why)
{
    if (why) *why = kNoFx3d;
    return false;
}

bool describeDevice(DeviceSummary*, std::string* why)
{
    if (why) *why = kNoFx3d;
    return false;
}

bool Backend::begin(int, int, int, const Params&, std::string* why)
{
    if (why) *why = kNoFx3d;
    return false;
}

bool Backend::running() const { return false; }

bool Backend::step(int, std::string* why)
{
    if (why) *why = kNoFx3d;
    return false;
}

bool Backend::readVelocity
(
    std::vector<float>&, std::vector<float>&, std::vector<float>&, std::string* why
)
{
    if (why) *why = kNoFx3d;
    return false;
}

bool Backend::readDensity(std::vector<float>&, std::string* why)
{
    if (why) *why = kNoFx3d;
    return false;
}

unsigned long long Backend::steps() const { return 0ull; }

void Backend::end() {}

#endif // AEROS_WITH_FX3D

} // namespace aerosFx3d
