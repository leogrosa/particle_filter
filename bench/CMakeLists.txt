cmake_minimum_required(VERSION 3.5)
project(mcl_bench)

# Standalone, non-catkin build: no ROS dependency, no Cython/Python wrapper.
# Exercises RangeLib.h's real MCL hot path directly (see mcl_bench.cpp).
option(WITH_CUDA "Compile the GPU rerun targets (mcl_bench_rmgpu, mcl_bench_lutgpu)" OFF)
option(LUT_CLOCKS "Build mcl_bench_lut with discretize_theta-vs-memory-access clock-split instrumentation" OFF)

set(RANGE_LIBC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/range_libc)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
	message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fno-math-errno")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include_directories(${RANGE_LIBC_DIR})
add_definitions(-DMAP_PATH=${CMAKE_CURRENT_SOURCE_DIR}/../maps/basement_fixed_unmapped120.png)

add_executable(mcl_bench mcl_bench.cpp
	${RANGE_LIBC_DIR}/vendor/lodepng/lodepng.cpp)
target_link_libraries(mcl_bench pthread)

# LUT-only rerun of the same sweep (see mcl_bench_lut.cpp) so adding GiantLUTCast's numbers
# doesn't require redoing the bl/rm/cddt/pcddt sweeps already collected.
add_executable(mcl_bench_lut mcl_bench_lut.cpp
    ${RANGE_LIBC_DIR}/vendor/lodepng/lodepng.cpp)
target_link_libraries(mcl_bench_lut pthread)

# Per-iteration convergence probe: does particle-filter convergence shrink the memory working set
# touched by GiantLUTCast's position-indexed giant_lut lookups? Logs every iteration individually
# (see mcl_convergence.cpp), unlike mcl_bench_lut's averaged WARMUP+TIMED sweep. Local-only /
# exploratory tool, not gated by WITH_CUDA.
add_executable(mcl_convergence mcl_convergence.cpp
    ${RANGE_LIBC_DIR}/vendor/lodepng/lodepng.cpp)
target_link_libraries(mcl_convergence pthread)

# opt-in only: -g -O0 for mcl_convergence specifically, invoked from a separate build dir
# (build_convergence_debug.sh) so the normal Release build above is untouched. Clang/GCC honor the
# last -O flag on the command line, so appending -O0 after CMAKE_CXX_FLAGS' global -O3 wins.
option(DEBUG_CONVERGENCE "Build mcl_convergence with -g -O0 instead of the global -O3, for lldb" OFF)
if(DEBUG_CONVERGENCE)
    target_compile_options(mcl_convergence PRIVATE -g -O0)
endif()

# opt-in only: splits calc_range's time into discretize_theta (arithmetic) vs giant_lut access
# (memory) via a standalone replica function inside mcl_bench_lut.cpp -- RangeLib.h itself is
# never touched. Same pattern as k_hops' COUNT_RM_HOPS below: flag off is a strict no-op.
if(LUT_CLOCKS)
	target_compile_definitions(mcl_bench_lut PRIVATE LUT_CLOCK_SPLIT=1)
endif()

# separate diagnostic tool: measures k (RayMarching dependent-hop count per ray), not timing.
# its own target so the COUNT_RM_HOPS instrumentation never touches mcl_bench's hot path.
add_executable(k_hops k_hops.cpp
	${RANGE_LIBC_DIR}/vendor/lodepng/lodepng.cpp)
target_compile_definitions(k_hops PRIVATE COUNT_RM_HOPS=1)
target_link_libraries(k_hops pthread)

# RMGPU-only rerun (see mcl_bench_rmgpu.cpp), same rationale as mcl_bench_lut: adds RMGPU's
# numbers to the same sweep without needing to rebuild/rerun the CPU-only methods with CUDA.
if(WITH_CUDA)
	find_package(CUDA REQUIRED)
	# sm_53 = Jetson Nano / TX1 (Maxwell). Not CMake's default target arch for this CUDA
	# toolkit version -- without it, kernels may fail to run on the actual board.
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_53")
	cuda_add_executable(mcl_bench_rmgpu mcl_bench_rmgpu.cpp
		${RANGE_LIBC_DIR}/includes/kernels.cu
		${RANGE_LIBC_DIR}/vendor/lodepng/lodepng.cpp)
	target_compile_definitions(mcl_bench_rmgpu PRIVATE USE_CUDA=1 CHUNK_SIZE=262144 NUM_THREADS=256)
	target_link_libraries(mcl_bench_rmgpu pthread)

	# GPU LUT rerun (see mcl_bench_lutgpu.cpp): the missing experiment -- a GPU version of the giant
	# LUT (GiantLUTCastGPU). Same structure/flags as mcl_bench_rmgpu above (same kernels.cu, same
	# -arch=sm_53, same CHUNK_SIZE/NUM_THREADS), adds this method's numbers to the same sweep.
	cuda_add_executable(mcl_bench_lutgpu mcl_bench_lutgpu.cpp
		${RANGE_LIBC_DIR}/includes/kernels.cu
		${RANGE_LIBC_DIR}/vendor/lodepng/lodepng.cpp)
	target_compile_definitions(mcl_bench_lutgpu PRIVATE USE_CUDA=1 CHUNK_SIZE=262144 NUM_THREADS=256)
	target_link_libraries(mcl_bench_lutgpu pthread)
endif()
