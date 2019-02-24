#Minimum Cmake version requirement
cmake_minimum_required(VERSION 3.0)
#Information about project
project(SampleCode)

#Set C++ compiling standard
set(CMAKE_CXX_STANDARD 11)
#Set C++ compiling flags
set(CMAKE_CXX_FLAGS "-O0 -Wall -g")

find_package("CUDA" REQUIRED)
    macro(cuda_add_compile_options options)
        list(APPEND CUDA_NVCC_FLAGS "${options}")
    endmacro()

include_directories(/usr/local/cudpp/include)
link_directories(/usr/local/cuda-7.5/lib64)
link_directories(/usr/local/cuda-9.2/lib64)
link_directories(/usr/local/cudpp/lib)

add_subdirectory(Hash)
set(SOURCE_FILES Hash/main.cpp)
add_executable(hashtest ${SOURCE_FILES})
link_directories(./Hash/)
target_link_libraries(hashtest test_driver)

set(MY_HASH_SOURCE Hash/test.cpp)
add_executable(myhash ${MY_HASH_SOURCE})
link_directories(./Hash/)
target_link_libraries(myhash warphash)

add_subdirectory(demo)