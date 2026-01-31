# npu_compiler/iree_compiler_plugin.cmake
# This file is picked up by IREE when you set IREE_CMAKE_PLUGIN_PATHS.

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/compiler/plugins/preprocessing/NPU
                 npu-preprocessing/NPU)
