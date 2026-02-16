# npu_compiler/compiler/plugins/preprocessing/NPU/CMakeLists.txt
cmake_minimum_required(VERSION 3.21)

add_library(npu-preprocessing-passes STATIC
  ${CMAKE_CURRENT_LIST_DIR}/../../../../pass/LowerMatmulToNPU.cpp
  ${CMAKE_CURRENT_LIST_DIR}/registration.cpp
)

target_include_directories(npu-preprocessing-passes PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Match LLVM/MLIR build flags (RTTI/exceptions typically OFF).
target_compile_options(npu-preprocessing-passes PRIVATE
  -fno-rtti
  -fno-exceptions
)

target_link_libraries(npu-preprocessing-passes PRIVATE
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)