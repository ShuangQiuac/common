#Minimum Cmake version requirement
cmake_minimum_required(VERSION 2.8)
#Information about project
project(SampleCode)

#Set C++ compiling standard
set(CMAKE_CXX_STANDARD 11)
#Set C++ compiling flags
set(CMAKE_CXX_FLAGS "-O0 -Wall -g")

#Set source code files to ${VECTOR_TEST_SOURCE}
set(VECTOR_TEST_SOURCE
        User.cpp
        Vector.cpp)
#Set source code files to ${MEMORY_TEST_SOURCE}
set(MEMORY_TEST_SOURCE
        Test.cpp
        Vector.cpp)

#Compile and generate executable file VectorTest from ${VECTOR_TEST_SOURCE}
add_executable(VectorTest ${VECTOR_TEST_SOURCE})
#Compile and generate executable file MemoryTest from ${MEMORY_TEST_SOURCE}
add_executable(MemoryTest ${MEMORY_TEST_SOURCE})