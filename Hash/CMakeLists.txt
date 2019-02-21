#Minimum Cmake version requirement
cmake_minimum_required(VERSION 3.0)
#Information about project
project(HashTable)

#Set C++ compiling standard
set(CMAKE_CXX_STANDARD 11)
#Set C++ compiling flags
set(CMAKE_CXX_FLAGS "-O0 -Wall -g")

find_package("CUDA" REQUIRED)	
    macro(cuda_add_compile_options options)
        list(APPEND CUDA_NVCC_FLAGS "${options}")
    endmacro()
    cuda_add_compile_options("--gpu-architecture=sm_30 -Xptxas -dlcm=cg -Xcompiler '-fPIC'")
    cuda_add_compile_options("-std=c++11 -g -G -lpthread")
    cuda_add_compile_options("-Xcompiler ${OpenMP_CXX_FLAGS}")

include_directories(/usr/local/cudpp/include)

link_directories(/usr/local/cuda-7.5/lib64)
link_directories(/usr/local/cuda-9.2/lib64)
link_directories(/usr/local/cudpp/lib)

cuda_add_library(test_driver SHARED cudpp_hash_test.cu)
target_link_libraries(test_driver cudart)
target_link_libraries(test_driver cudpp_hash)
target_link_libraries(test_driver cudpp)

set(SOURCE_FILES main.cpp)
add_executable(hashtest ${SOURCE_FILES})

target_link_libraries(hashtest test_driver)