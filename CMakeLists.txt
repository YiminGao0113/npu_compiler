cmake_minimum_required(VERSION 3.20)
project(npu_compiler LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found MLIR: ${MLIR_DIR}")
message(STATUS "Found LLVM: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(npu-opt
  tool/npu-opt.cpp
  pass/LowerMatmulToNPU.cpp
)
target_link_libraries(npu-opt PRIVATE
  MLIRMlirOptMain
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRDialect
  MLIRRewrite
  MLIRAnalysis
)

target_include_directories(npu-opt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

llvm_update_compile_flags(npu-opt)
mlir_check_all_link_libraries(npu-opt)
