#pragma once
// ============================================================================
// МОСТ Aeros <-> FluidX3D 3.9
//
// Зачем отдельный модуль: FluidX3D нельзя просто «подключить». У него своя
// раскладка памяти, свои флаги границ и свой контракт инициализации, а его
// решатель скомпилирован под OpenCL и стартует только при наличии устройства.
// Поэтому всё, что можно проверить БЕЗ устройства — отображение индексов и
// перевод геометрии во флаги — вынесено в чистые функции, а сами вызовы
// FluidX3D спрятаны за AEROS_WITH_FX3D. При выключенном флаге модуль
// собирается и тестируется где угодно.
//
// Что здесь сопоставляется (по исходникам FluidX3D v3.9):
//
//   индекс ячейки: Aeros  n = (z*Ny + y)*Nx + x
//                  FX3D   n = x + Nx*(y + Ny*z)      (lbm.hpp, Memory_Container)
//   твёрдое тело:  Aeros  g_voxelData[n] == 1
//                  FX3D   flags[n] = TYPE_S          (0b00000001, defines.hpp)
//   вход потока:   FX3D   flags[n] = TYPE_E          (0b00000010, равновесная
//                  граница: плотность/скорость задаются на ячейке и не
//                  отражают волны), скорость кладётся в u.x/u.y/u.z
//   жидкость:      FX3D   flags[n] = TYPE_F          (0b00001000), выставляется
//                  самим решателем при инициализации, вручную не пишется
//
// ГРАНИЦА ОТВЕТСТВЕННОСТИ. Мост заполняет flags и начальное поле скорости и
// читает rho/u обратно; он НЕ трогает решатель и НЕ подменяет физику Aeros.
// Включение бэкенда — решение вызывающего кода, а не этого модуля.
// ============================================================================

#include <cstddef>
#include <string>
#include <vector>

namespace aerosFx3d {

// --- Значения флагов FluidX3D v3.9 (defines.hpp). Дублируются здесь, чтобы
// этот заголовок не тянул за собой OpenCL и весь FluidX3D: тест отображения
// должен собираться на машине без единого устройства. Значения сверяются
// статически при сборке с AEROS_WITH_FX3D (см. fx3d_backend.cpp).
constexpr unsigned char TYPE_S = 0x01;   // твёрдая граница (bounce-back)
constexpr unsigned char TYPE_E = 0x02;   // равновесная граница (вход/выход)
constexpr unsigned char TYPE_F = 0x08;   // жидкость (ставит решатель)

// --- Параметры включения бэкенда. Значения по умолчанию согласованы с
// lbm.cpp: скорость входа 0.08 в решёточных единицах и nu = 1/6 подобраны там
// же, чтобы сравнение двух решателей было на одном режиме.
struct Params {
    bool  enabled     = false;
    float nuLB        = 1.0f/6.0f;   // кинематическая вязкость, решёточные ед.
    float inletU      = 0.08f;       // скорость входа, решёточные ед.
    float bodyForceX  = 0.0f;        // объёмная сила, решёточные ед.
    float bodyForceY  = 0.0f;
    float bodyForceZ  = 0.0f;
    bool  periodicX   = false;       // как lbmParams.periodicX
    bool  periodicZ   = false;       // как lbmParams.periodicZ
};

// --- Чистое отображение индексов -------------------------------------------

inline std::size_t aerosIndex(int nx, int ny, int x, int y, int z)
{
    return (std::size_t(z)*std::size_t(ny) + std::size_t(y))*std::size_t(nx)
         + std::size_t(x);
}

inline std::size_t fx3dIndex(int nx, int ny, int x, int y, int z)
{
    return std::size_t(x)
         + std::size_t(nx)*(std::size_t(y) + std::size_t(ny)*std::size_t(z));
}

// Перекладывает геометрию Aeros (voxels[n] != 0 — твёрдое тело) во флаги
// FluidX3D. Жидкость остаётся 0: её досчитает решатель, а запись TYPE_F руками
// до инициализации — это ровно та ошибка, которую он ловит своими sanity
// checks. Возвращает число твёрдых ячеек.
//
// flowAxis/flowSign — та же ось потока, что считает computeFlowAxis() в
// lbm.cpp: вход ставится на наветренной грани, если поток не периодичен по
// этой оси. При periodicX вход не ставится вообще (иначе на одной границе
// оказались бы два разных условия).
std::size_t mapGeometry
(
    int nx, int ny, int nz,
    const std::vector<int>& voxels,
    std::vector<unsigned char>& flags,
    int flowAxis,
    int flowSign,
    bool periodicX
);

// Начальная скорость: равномерный поток вдоль оси, со стенками на остальных
// гранях (тот же режим, что ставит initLBM для внешнего обтекания).
void mapInletVelocity
(
    int nx, int ny, int nz,
    int flowAxis,
    int flowSign,
    float uLB,
    std::vector<float>& u   // длина 3*N, раскладка FX3D: [компонента*N + n]
);

// Приблизительная оценка памяти устройства: FluidX3D v3.9 под D3Q19+FP16S
// держит 55 байт на ячейку на CPU/iGPU и 93 на дискретном GPU (README).
std::size_t estimateDeviceBytes(std::size_t cells, bool unifiedMemory);

// --- Работа с устройством (только при AEROS_WITH_FX3D) --------------------
//
// Без устройства эти функции возвращают false и заполняют why — тот же принцип,
// что у aerosGPU: библиотека загружается везде и объясняет, почему не ускоряет.

bool available(std::string* why);

struct DeviceSummary {
    std::string name;
    std::string platform;
    std::size_t memoryBytes = 0;
    std::size_t cells = 0;       // сколько ячеек влезает в память с запасом
};

bool describeDevice(DeviceSummary* out, std::string* why);

class Backend
{
public:

    Backend();
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // Создаёт решатель FluidX3D на nx*ny*nz, заливает флаги и начальную
    // скорость. false — с причиной в why (нет устройства, не хватает памяти,
    // конфигурация не поддержана).
    bool begin(int nx, int ny, int nz, const Params& p, std::string* why);

    bool running() const;

    // Шаги решётки. false — если не инициализировано.
    bool step(int steps, std::string* why);

    // Читает поле обратно в раскладку Aeros (cell*1, компоненты рядом):
    // ux/uy/uz длиной N. Читается через read_from_device(), то есть ровно то,
    // что видит сам FluidX3D, а не промежуточный кэш.
    bool readVelocity(std::vector<float>& ux, std::vector<float>& uy,
                      std::vector<float>& uz, std::string* why);

    bool readDensity(std::vector<float>& rho, std::string* why);

    // Сколько шагов решётки сделано — для журнала и панели.
    unsigned long long steps() const;

    void end();

private:

    struct Impl;
    Impl* impl_;
};

} // namespace aerosFx3d
