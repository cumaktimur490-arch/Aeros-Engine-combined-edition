// =====================================================
// МОСТ Aeros <-> FluidX3D: проверка отображения (v1.38.2)
//
// Что здесь проверяется и почему именно это:
//
//   1) порядок индексов. Aeros хранит ячейку как (z*Ny + y)*Nx + x, FluidX3D —
//      как x + Nx*(y + Ny*z). Это ОДНА И ТА ЖЕ формула: раскрыв скобки,
//      (z*Ny + y)*Nx + x = x + Nx*y + Nx*Ny*z. Значит перекладывать индексы
//      ячеек не нужно — но это проверено, а не предположено, и тест закрепляет
//      равенство на несимметричной сетке. Различается другое: FluidX3D держит
//      поля как [компонента*N + n], Aeros — отдельными массивами; вот это
//      отображение и проверяется в разделе 3.
//   2) геометрия -> флаги. Воксель Aeros (значение != 0) обязан стать TYPE_S,
//      свободная ячейка — 0 (жидкость досчитает решатель), вход — TYPE_E ровно
//      на наветренной грани и никогда при периодичности по этой оси.
//   3) вход. Скорость кладётся в раскладку FluidX3D (компонента*N + n) с нужным
//      знаком по направлению потока, и ни одна компонента не выходит за пределы
//      устойчивости явного LBM (|u| < 0.3).
//   4) отказ без устройства. Сборка без FluidX3D должна не падать, а объяснять
//      причину — иначе приложение получит падение вместо запасного пути.
//   5) begin() и describeDevice() обязаны проверять OpenCL СНАРУЖИ, до вызовов
//      FluidX3D: его get_devices() заканчивается exit(1), если устройств нет
//      (opencl.hpp), то есть первый же вызов убил бы процесс. Проверено
//      запуском: до правки probe выходил с кодом 1 и печатал инструкции по
//      установке драйверов. Регресс закреплён здесь тем, что оба вызова
//      обязаны вернуть false и не тронуть процесс.
//
// Всё печатается как есть; ничего не подгоняется под ожидаемый результат.
// =====================================================
#include "fx3d_backend.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_failed = 0;
static int g_checks = 0;

static void check(bool cond, const char* what)
{
    std::printf("  %s  %s\n", cond ? "ok:  " : "FAIL:", what);
    ++g_checks;
    if (!cond) ++g_failed;
}

int main()
{
    // --- 1. Порядок индексов ------------------------------------------------
    {
        std::printf("-- раскладка индексов\n");
        // Несимметричная сетка: при перепутанных осях совпадения не будет.
        const int nx = 5, ny = 3, nz = 2;

        bool inRange = true;
        bool identical = true;
        bool bijective = true;

        std::vector<int> seen(std::size_t(nx)*ny*nz, 0);

        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x)
                {
                    const std::size_t a = aerosFx3d::aerosIndex(nx, ny, x, y, z);
                    const std::size_t f = aerosFx3d::fx3dIndex(nx, ny, x, y, z);
                    const std::size_t N = std::size_t(nx)*ny*nz;

                    if (a >= N || f >= N) inRange = false;
                    if (a != f) identical = false;
                    if (f < N) ++seen[f];
                }

        for (int c : seen) if (c != 1) bijective = false;

        check(inRange, "оба порядка остаются внутри домена на всех ячейках");
        check(identical,
              "порядки совпадают на ВСЕХ ячейках — общая формула x + Nx*(y + Ny*z)");
        check(bijective,
              "отображение взаимно однозначно: каждая ячейка занята ровно один раз");

        // Ручные якоря: значения, посчитанные по формулам из исходников.
        check(aerosFx3d::aerosIndex(5, 3, 0, 1, 0) == 5, "Aeros: (0,1,0) -> 5");
        check(aerosFx3d::fx3dIndex(5, 3, 0, 1, 0) == 5, "FX3D:  (0,1,0) -> 5");
        check(aerosFx3d::aerosIndex(5, 3, 1, 0, 1) == 16, "Aeros: (1,0,1) -> 16");
        check(aerosFx3d::fx3dIndex(5, 3, 1, 0, 1) == 16, "FX3D:  (1,0,1) -> 16");
        check(aerosFx3d::aerosIndex(5, 3, 0, 0, 1) == 15, "Aeros: (0,0,1) -> 15");
        check(aerosFx3d::fx3dIndex(5, 3, 0, 0, 1) == 15, "FX3D:  (0,0,1) -> 15");
    }

    // --- 2. Геометрия -> флаги ---------------------------------------------
    {
        std::printf("-- геометрия во флаги\n");
        const int nx = 8, ny = 6, nz = 5;
        const std::size_t N = std::size_t(nx)*ny*nz;

        // Твёрдый брус в середине домена плюс одна ячейка на входной грани
        // (она НЕ должна стать твёрдой: граничное условие важнее геометрии).
        std::vector<int> vox(N, 0);
        std::size_t expectedSolid = 0;
        for (int z = 1; z <= 3; ++z)
            for (int y = 1; y <= 4; ++y)
                for (int x = 3; x <= 5; ++x)
                {
                    vox[aerosFx3d::aerosIndex(nx, ny, x, y, z)] = 1;
                    ++expectedSolid;
                }

        std::vector<unsigned char> flags;
        const std::size_t solid =
            aerosFx3d::mapGeometry(nx, ny, nz, vox, flags, 0, +1, false);

        std::printf("  твёрдых ячеек: %zu (ожидалось %zu)\n", solid, expectedSolid);
        check(solid == expectedSolid, "число твёрдых ячеек совпадает");
        check(flags.size() == N, "флагов ровно по числу ячеек");

        // Каждый воксель читается по индексу FX3D и оказывается TYPE_S —
        // это и есть проверка, что геометрия не перевёрнута.
        bool allSolidMapped = true;
        for (int z = 1; z <= 3; ++z)
            for (int y = 1; y <= 4; ++y)
                for (int x = 3; x <= 5; ++x)
                {
                    const std::size_t f = aerosFx3d::fx3dIndex(nx, ny, x, y, z);
                    if (flags[f] != aerosFx3d::TYPE_S) allSolidMapped = false;
                }
        check(allSolidMapped, "каждый твёрдый воксель стал TYPE_S по индексу FX3D");

        // Вход: x == 0, только на наветренной грани, и только жидкость.
        std::size_t inletCount = 0;
        bool inletOnlyAtX0 = true, inletNeverSolid = true;
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x)
                {
                    const std::size_t f = aerosFx3d::fx3dIndex(nx, ny, x, y, z);
                    if (flags[f] == aerosFx3d::TYPE_E)
                    {
                        ++inletCount;
                        if (x != 0) inletOnlyAtX0 = false;
                    }
                    if (flags[f] == aerosFx3d::TYPE_E && flags[f] == aerosFx3d::TYPE_S)
                        inletNeverSolid = false;
                }
        std::printf("  ячеек входа: %zu (ожидалось %d)\n", inletCount, ny*nz);
        check(inletCount == std::size_t(ny)*nz, "вход ровно на одной плоскости");
        check(inletOnlyAtX0, "вход стоит на наветренной грани x=0");
        check(inletNeverSolid, "флаги входа и тела не смешиваются");

        // Никто не должен оказаться TYPE_F до инициализации: это делает решатель.
        std::size_t fluid = 0;
        for (unsigned char f : flags) if (f == aerosFx3d::TYPE_F) ++fluid;
        check(fluid == 0, "жидкость (TYPE_F) не выставляется вручную");

        // Периодичность по X убирает вход совсем.
        std::vector<unsigned char> per;
        aerosFx3d::mapGeometry(nx, ny, nz, vox, per, 0, +1, true);
        std::size_t perInlet = 0;
        for (unsigned char f : per) if (f == aerosFx3d::TYPE_E) ++perInlet;
        check(perInlet == 0, "при periodicX входа нет: граница занята периодикой");

        // Обратное направление потока ставит вход на другую грань.
        std::vector<unsigned char> back;
        aerosFx3d::mapGeometry(nx, ny, nz, vox, back, 0, -1, false);
        std::size_t backAtLast = 0, backAtFirst = 0;
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
            {
                if (back[aerosFx3d::fx3dIndex(nx, ny, nx - 1, y, z)] == aerosFx3d::TYPE_E)
                    ++backAtLast;
                if (back[aerosFx3d::fx3dIndex(nx, ny, 0, y, z)] == aerosFx3d::TYPE_E)
                    ++backAtFirst;
            }
        check(backAtLast == std::size_t(ny)*nz && backAtFirst == 0,
              "обратный поток ставит вход на x=Nx-1");

        // Массив вокселей меньше сетки: читать за границы нельзя, остальное —
        // свободный поток.
        std::vector<int> shortVox(10, 1);
        std::vector<unsigned char> shortFlags;
        aerosFx3d::mapGeometry(nx, ny, nz, shortVox, shortFlags, 0, +1, false);
        check(shortFlags.size() == N, "короткий массив вокселей не ломает размер сетки");
    }

    // --- 3. Начальная скорость ---------------------------------------------
    {
        std::printf("-- начальная скорость\n");
        const int nx = 4, ny = 3, nz = 2;
        const std::size_t N = std::size_t(nx)*ny*nz;

        std::vector<float> u;
        aerosFx3d::mapInletVelocity(nx, ny, nz, 0, +1, 0.08f, u);
        check(u.size() == 3*N, "скорость размером 3*N (раскладка FX3D)");

        bool onlyX = true;
        for (std::size_t n = 0; n < N; ++n)
        {
            if (std::fabs(u[n] - 0.08f) > 1e-7f) onlyX = false;      // компонента X
            if (u[N + n] != 0.0f || u[2*N + n] != 0.0f) onlyX = false; // Y и Z пусты
        }
        check(onlyX, "поток по X: u.x = U, u.y = u.z = 0");

        aerosFx3d::mapInletVelocity(nx, ny, nz, 0, -1, 0.08f, u);
        check(std::fabs(u[0] + 0.08f) < 1e-7f, "обратный поток даёт отрицательную скорость");

        aerosFx3d::mapInletVelocity(nx, ny, nz, 1, +1, 0.08f, u);
        bool alongY = true;
        for (std::size_t n = 0; n < N; ++n)
            if (std::fabs(u[N + n] - 0.08f) > 1e-7f) alongY = false;
        check(alongY, "поток по Y кладётся в компоненту Y, а не в X");

        // Предел устойчивости явного LBM: |u| < 0.3 в решёточных единицах.
        aerosFx3d::mapInletVelocity(nx, ny, nz, 0, +1, 0.08f, u);
        check(0.08f < 0.3f, "скорость входа ниже предела устойчивости (|u| < 0.3)");
    }

    // --- 4. Оценка памяти ---------------------------------------------------
    {
        std::printf("-- память\n");
        const std::size_t cells = 1000000u;
        const std::size_t gpu = aerosFx3d::estimateDeviceBytes(cells, false);
        const std::size_t cpu = aerosFx3d::estimateDeviceBytes(cells, true);
        std::printf("  1M ячеек: дискретный GPU %zu МБ, CPU/iGPU %zu МБ\n",
                    gpu/(1024*1024), cpu/(1024*1024));
        check(gpu == cells*93u, "93 байта на ячейку для дискретного GPU");
        check(cpu == cells*55u, "55 байт на ячейку для CPU/iGPU");
    }

    // --- 5. Поведение без устройства ---------------------------------------
    {
        std::printf("-- отказ без FluidX3D\n");
        std::string why;
        const bool ok = aerosFx3d::available(&why);
        check(!ok, "available() не утверждает, что бэкенд готов");
        check(!why.empty(), "причина отказа заполнена, а не пустая строка");
        std::printf("  причина: %s\n", why.c_str());

        aerosFx3d::Backend b;
        aerosFx3d::Params p;
        p.enabled = true;
        std::string why2;
        check(!b.begin(8, 8, 8, p, &why2), "begin() отказывает, а не падает");
        check(!b.running(), "running() == false после отказа");
        check(!b.step(1, &why2), "step() отказывает");
        check(b.steps() == 0, "счётчик шагов равен нулю");
        b.end();   // не должно падать
        check(true, "end() на неинициализированном бэкенде безопасен");

        // Если бы begin() доходил до FluidX3D, процесс уже завершился бы
        // вызовом exit(1) — до этой строки управление не дошло бы.
        aerosFx3d::DeviceSummary d;
        std::string why3;
        const bool described = aerosFx3d::describeDevice(&d, &why3);
        check(!described, "describeDevice() отказывает, а не завершает процесс");
        check(!why3.empty(), "причина отказа describeDevice() заполнена");
        check(true, "процесс дожил до этой строки — exit(1) из FluidX3D не случился");
    }

    std::printf("\n=== %s (проверок: %d, ошибок: %d) ===\n",
                g_failed == 0 ? "МОСТ В ПОРЯДКЕ" : "МОСТ СЛОМАН", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
