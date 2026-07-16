cmake_minimum_required(VERSION 3.5)
project(mcl_bench)

# Standalone, non-catkin build: no ROS dependency, no Cython/Python wrapper.
# Exercises RangeLib.h's real MCL hot path directly (see mcl_bench.cpp).
option(WITH_CUDA "Compile GPU ray casting variant (RayMarchingGPU)" OFF)

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

if(WITH_CUDA)
	add_definitions(-DUSE_CUDA=1 -DCHUNK_SIZE=262144 -DNUM_THREADS=256)
	find_package(CUDA REQUIRED)
	cuda_add_executable(mcl_bench mcl_bench.cpp
		${RANGE_LIBC_DIR}/includes/kernels.cu
		${RANGE_LIBC_DIR}/vendor/lodepng/lodepng.cpp)
else()
	add_definitions(-DUSE_CUDA=0)
	add_executable(mcl_bench mcl_bench.cpp
		${RANGE_LIBC_DIR}/vendor/lodepng/lodepng.cpp)
endif()

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
