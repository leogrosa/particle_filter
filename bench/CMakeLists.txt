cmake_minimum_required(VERSION 3.5)
project(mcl_bench)

# Standalone, non-catkin build: no ROS dependency, no Cython/Python wrapper.
# Exercises RangeLib.h's real MCL hot path directly (see mcl_bench.cpp).
option(WITH_CUDA "Compile the RMGPU rerun target (mcl_bench_rmgpu)" OFF)

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
add_definitions(-DMAP_PATH=${CMAKE_CURRENT_SOURCE_DIR}/../maps/basement_fixed.png)

add_executable(mcl_bench mcl_bench.cpp
	${RANGE_LIBC_DIR}/vendor/lodepng/lodepng.cpp)
target_link_libraries(mcl_bench pthread)

# LUT-only rerun of the same sweep (see mcl_bench_lut.cpp) so adding GiantLUTCast's numbers
# doesn't require redoing the bl/rm/cddt/pcddt sweeps already collected.
add_executable(mcl_bench_lut mcl_bench_lut.cpp
    ${RANGE_LIBC_DIR}/vendor/lodepng/lodepng.cpp)
target_link_libraries(mcl_bench_lut pthread)

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
endif()
