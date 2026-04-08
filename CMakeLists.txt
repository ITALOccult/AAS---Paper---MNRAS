cmake_minimum_required(VERSION 3.20)

project(AASPaperArtifacts LANGUAGES CXX)

option(AAS_BUILD_CPP_BENCHMARKS "Build the C++ benchmark artifacts" ON)

if(AAS_BUILD_CPP_BENCHMARKS)
    add_subdirectory(benchmarks/cpp)
endif()