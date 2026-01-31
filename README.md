```bash
export ROOT=/localsdd/yg9bq
export NPU_COMPILER=$ROOT/npu_compiler
export IREE_SRC=$ROOT/iree
export IREE_BUILD=$ROOT/iree-build

cd $ROOT
git clone https://github.com/iree-org/iree.git
cd $IREE_SRC
git submodule update --init --recursive

cmake -B $IREE_BUILD -S $IREE_SRC \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_TOOLS=ON \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_INPUT_TOSA=OFF \
  -DIREE_INPUT_TORCH=OFF \
  -DIREE_INPUT_STABLEHLO=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
  -DIREE_CMAKE_PLUGIN_PATHS=$NPU_COMPILER

cmake --build $IREE_BUILD -j2 --target iree-compile

$IREE_BUILD/tools/iree-compile \
  $NPU_COMPILER/test/matmul_8xKx8.mlir \
  --iree-preprocessing-pass-pipeline="builtin.module(func.func(lower-matmul-to-npu))" \
  --compile-to=vm

```
