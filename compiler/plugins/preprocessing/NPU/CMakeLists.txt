# npu_compiler/compiler/plugins/preprocessing/NPU/CMakeLists.txt
cmake_minimum_required(VERSION 3.21)

# Phase-0 preprocessing passes for IREE.
add_library(npu-preprocessing-passes STATIC
../../../../pass/LowerMatmulToNPU.cpp
registration.cpp
)

# This library is internal to the plugin, so keep includes private.
target_include_directories(npu-preprocessing-passes
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against MLIR components used by the pass.
# Keep PRIVATE so we don't leak MLIR deps to anything that links this library.
target_link_libraries(npu-preprocessing-passes
  PRIVATE
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRIR
    MLIRPass
    MLIRSupport
)
