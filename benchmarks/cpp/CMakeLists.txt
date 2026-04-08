cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(AASPaperCppBenchmarks LANGUAGES CXX)
endif()

set(ASTDYN_SOURCE_DIR "" CACHE PATH "Path to a local AstDyn checkout")

set(astdyn_target "")

if(ASTDYN_SOURCE_DIR)
    if(NOT EXISTS "${ASTDYN_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "ASTDYN_SOURCE_DIR does not point to a valid AstDyn source tree: ${ASTDYN_SOURCE_DIR}")
    endif()

    set(ASTDYN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ASTDYN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ASTDYN_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(ASTDYN_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(ASTDYN_BUILD_CATALOG OFF CACHE BOOL "" FORCE)

    add_subdirectory("${ASTDYN_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/astdyn-build")
    set(astdyn_target astdyn)
else()
    find_package(AstDyn CONFIG REQUIRED)
    set(astdyn_target AstDyn::astdyn)
endif()

add_executable(benchmark_long_term_v2 benchmark_long_term_v2.cpp)
target_compile_features(benchmark_long_term_v2 PRIVATE cxx_std_23)

get_filename_component(AAS_PAPER_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
target_compile_definitions(benchmark_long_term_v2 PRIVATE AAS_PAPER_REPO_ROOT="${AAS_PAPER_REPO_ROOT}")

if(ASTDYN_SOURCE_DIR)
    target_include_directories(benchmark_long_term_v2 PRIVATE
        "${ASTDYN_SOURCE_DIR}/include"
        "${ASTDYN_SOURCE_DIR}"
        "${CMAKE_BINARY_DIR}/astdyn-build/include"
    )
endif()

target_link_libraries(benchmark_long_term_v2 PRIVATE ${astdyn_target})

set_target_properties(benchmark_long_term_v2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)