cmake_minimum_required(VERSION 3.6)
project(fox_multiplication)

#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")

add_executable(fox_multiplication main.cpp)

include_directories("/usr/include/mpi/")