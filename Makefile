# Tests and build helpers for aerosGPU.
#
# The library itself is built with OpenFOAM's wmake (see
# applications/utilities/aerosGPU/README.md). Everything here is buildable with a plain
# g++, which matters because the tests have to run where OpenFOAM is not installed:
#   * the core (matrix, preconditioners, Krylov solvers, device layer) needs no
#     OpenFOAM at all — it works on plain arrays and indices;
#   * that is what makes it testable at all, and the tests are the only reason to
#     believe the numbers.
#
# make            build and run the core tests
# make asan       the same under AddressSanitizer + UndefinedBehaviorSanitizer
# make clean

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -fopenmp -Wall -Wextra -Wold-style-cast
LDFLAGS ?= -fopenmp -ldl

GPU := applications/utilities/aerosGPU

GPU_CORE := \
    $(GPU)/aerosGpuCl.C \
    $(GPU)/aerosGpuDevice.C \
    $(GPU)/aerosGpuKernels.C \
    $(GPU)/aerosGpuMatrix.C \
    $(GPU)/aerosGpuSolver.C \
    $(GPU)/aerosGpuSolverDevice.C \
    $(GPU)/aerosGpuFoamCore.C

BIN := build/aerosgpu_core

.PHONY: all test asan clean

all: test

test: $(BIN)
	./$(BIN)

$(BIN): tests/aerosgpu_core.cpp $(GPU_CORE)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -I$(GPU) tests/aerosgpu_core.cpp $(GPU_CORE) -o $@ $(LDFLAGS)

# The same suite under sanitizers. Worth running after touching the row structure: an
# off-by-one in a rowStart is read out of bounds long before it gives a wrong number.
asan:
	@mkdir -p build
	$(CXX) -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined \
	    -I$(GPU) tests/aerosgpu_core.cpp $(GPU_CORE) \
	    -o build/aerosgpu_core_asan -fopenmp -ldl
	./build/aerosgpu_core_asan

clean:
	rm -rf build
