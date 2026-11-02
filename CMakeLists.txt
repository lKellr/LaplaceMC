cmake_minimum_required(VERSION 3.23)

project(LaplaceMC)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)
set(CMAKE_CXX_FLAGS "-O3 -march=native")

find_package(OpenMP REQUIRED)

add_executable(LaplaceMClvs)

target_compile_features(LaplaceMClvs PUBLIC cxx_std_23)

target_link_libraries(LaplaceMClvs PRIVATE OpenMP::OpenMP_CXX)

target_sources(LaplaceMClvs
  PRIVATE
    LaplaceMClvs.cpp
)
