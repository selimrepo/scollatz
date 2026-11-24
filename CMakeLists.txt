cmake_minimum_required(VERSION 3.20)
project(scollatz VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SCOLLATZ_ENABLE_OPENCL "Build the OpenCL GPU backend when available" ON)
option(SCOLLATZ_BUILD_TESTS "Build tests" ON)

find_package(Threads REQUIRED)
if(SCOLLATZ_ENABLE_OPENCL)
  find_package(OpenCL QUIET)
endif()

add_executable(scollatz src/main.cpp)\nadd_executable(scollatz-ga src/genetic.cpp)\ntarget_link_libraries(scollatz-ga PRIVATE Threads::Threads)\ntarget_compile_options(scollatz-ga PRIVATE\n  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>\n)
target_link_libraries(scollatz PRIVATE Threads::Threads)
target_compile_options(scollatz PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>
)
target_compile_definitions(scollatz PRIVATE
  SCOLLATZ_KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/collatz.cl"
)
if(OpenCL_FOUND)
  target_link_libraries(scollatz PRIVATE OpenCL::OpenCL)
  target_compile_definitions(scollatz PRIVATE SCOLLATZ_HAS_OPENCL=1)
  message(STATUS "OpenCL backend enabled")
else()
  message(STATUS "OpenCL not found; building CPU-only backend")
endif()

if(SCOLLATZ_BUILD_TESTS)
  enable_testing()
  add_test(NAME cpu_smoke COMMAND scollatz --start 1 --end 10000 --backend cpu --threads 2)
  add_test(NAME argument_validation COMMAND scollatz --start 0 --end 10)
  set_tests_properties(argument_validation PROPERTIES WILL_FAIL TRUE)
endif()

install(TARGETS scollatz RUNTIME DESTINATION bin)
