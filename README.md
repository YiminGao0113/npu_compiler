# NPU IREE Plugin – Complete Exploration Workflow

This is the exact reproducible workflow used to validate the
`lower-matmul-to-npu` preprocessing pass.

Assumed directory layout:

workspace/
  iree/
  npu_compiler/

---

## 0️⃣ Setup Environment (relative paths)

From inside `workspace/`:

```bash
export WORKSPACE=$(pwd)
export IREE_SRC=$WORKSPACE/iree
export NPU_COMPILER=$WORKSPACE/npu_compiler
export IREE_BUILD=$IREE_SRC/build
```

---

## 1️⃣ Clone IREE (if not already cloned)

```bash
git clone https://github.com/iree-org/iree.git
```

---

## 2️⃣ Configure IREE with plugin (relative path)

```bash
cmake -G Ninja \
  -B $IREE_BUILD \
  -S $IREE_SRC \
  -DIREE_CMAKE_PLUGIN_PATHS=$NPU_COMPILER
```

---

## 3️⃣ Build iree-compile

```bash
ninja -C $IREE_BUILD tools/iree-compile
```

---

## 4️⃣ Confirm Pass Is Linked

```bash
nm -C $IREE_BUILD/tools/iree-compile | grep LowerMatmulToNPUPass
```

You should see symbols for the pass.

---

## 5️⃣ Create 8x8x8 Test Matmul

```bash
cat > /tmp/test.mlir <<'EOF'
module {
  func.func @run(%a: tensor<8x8xi8>,
                 %b: tensor<8x8xi8>,
                 %c: tensor<8x8xi16>)
      -> tensor<8x8xi16> {
    %0 = linalg.matmul
        ins(%a, %b : tensor<8x8xi8>, tensor<8x8xi8>)
        outs(%c : tensor<8x8xi16>)
        -> tensor<8x8xi16>
    return %0 : tensor<8x8xi16>
  }
}
EOF
```

---

## 6️⃣ Run Preprocessing Only

```bash
$IREE_BUILD/tools/iree-compile \
  /tmp/test.mlir \
  --iree-preprocessing-pass-pipeline="builtin.module(lower-matmul-to-npu)" \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=vmvx \
  --compile-to=preprocessing \
  -o /tmp/after_preprocess.mlir
```

Verify rewrite:

```bash
grep -n "npu.matmul_8x8x8" /tmp/after_preprocess.mlir
```

Expected:
- `linalg.matmul` replaced by `func.call @npu.matmul_8x8x8`
- Private declaration emitted

---

## 7️⃣ Run Full Pipeline to VM

```bash
$IREE_BUILD/tools/iree-compile \
  /tmp/test.mlir \
  --iree-preprocessing-pass-pipeline="builtin.module(lower-matmul-to-npu)" \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=vmvx \
  --compile-to=vm \
  -o /tmp/after_vm.mlir
```

Verify VM import:

```bash
grep -n "vm.import" /tmp/after_vm.mlir
```

Expected:
- `vm.call @npu.matmul_8x8x8`
- `vm.import private @npu.matmul_8x8x8`

---

---

## 8️⃣ Generate Deployable VM Bytecode (.vmfb) + Runtime Flow

The real deployable artifact for runtime is the VM bytecode file.

```bash
$IREE_BUILD/tools/iree-compile \
  /tmp/test.mlir \
  --iree-preprocessing-pass-pipeline="builtin.module(lower-matmul-to-npu)" \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=vmvx \
  -o /tmp/model.vmfb
```

This produces:

```
/tmp/model.vmfb
```

The `.vmfb` contains:
- VM bytecode
- `vm.call @npu.matmul_8x8x8`
- `vm.import @npu.matmul_8x8x8`

---

### Runtime Integration Model

At runtime:

1. IREE loads `model.vmfb`
2. VM executes `vm.call @npu.matmul_8x8x8`
3. VM resolves `vm.import @npu.matmul_8x8x8`
4. Runtime must provide a native C function with the same symbol name
5. That C function:
   - Extracts raw buffer pointers from `!hal.buffer_view`
   - Converts to physical addresses (if needed)
   - Calls ARM + FPGA GEMM kernel
   - Wraps output into a buffer view
   - Returns to VM

---

### End-to-End Architecture

```
linalg.matmul
      ↓
lower-matmul-to-npu (MLIR preprocessing pass)
      ↓
func.call @npu.matmul_8x8x8
      ↓
vm.import @npu.matmul_8x8x8
      ↓
model.vmfb
      ↓
IREE Runtime
      ↓
C binding function
      ↓
ARM driver
      ↓
FPGA 8×8 systolic NPU
```

---

### What This Demonstrates

• Custom preprocessing pass registered  
• `linalg.matmul` intercepted  
• Rewritten to hardware-specific ABI  
• Rewrite survives full IREE lowering  
• VM import automatically generated  
• `.vmfb` is deployable artifact  
• Ready for backend C binding  

This completes the compiler → VM → runtime → hardware pipeline.
