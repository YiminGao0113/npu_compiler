#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct LowerMatmulToNPUPass
    : public PassWrapper<LowerMatmulToNPUPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMatmulToNPUPass)

  // Phase-0 “device” assumptions (only used for printing today).
  static constexpr int kTileM = 8;
  static constexpr int kTileN = 8;
  static constexpr int kKGranularity = 8;  // K multiple of 8
  static constexpr int kBramWidthBits = 64;

  StringRef getArgument() const final { return "lower-matmul-to-npu"; }
  StringRef getDescription() const final {
    return "Phase-0: recognize 8xKx8 linalg.matmul and print NPU dispatch record";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Walk all linalg.matmul ops in the function.
    func.walk([&](linalg::MatmulOp op) {
      // linalg.matmul has 2 inputs (A, B) and 1 output (C).
      // We’ll require ranked tensor types for Phase 0.
      auto aTy = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
      auto bTy = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
      auto cTy = dyn_cast<RankedTensorType>(op.getOutputs()[0].getType());

      if (!aTy || !bTy || !cTy) {
        op.emitError("Phase-0 NPU expects ranked tensor types for matmul operands");
        signalPassFailure();
        return;
      }
      if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2) {
        op.emitError("Phase-0 NPU expects 2D tensors for matmul");
        signalPassFailure();
        return;
      }

      // Shapes: A[M,K], B[K,N], C[M,N]
      int64_t M = aTy.getDimSize(0);
      int64_t K_a = aTy.getDimSize(1);
      int64_t K_b = bTy.getDimSize(0);
      int64_t N = bTy.getDimSize(1);
      int64_t M_c = cTy.getDimSize(0);
      int64_t N_c = cTy.getDimSize(1);

      auto fail = [&](StringRef msg) {
        op.emitError() << msg
                       << " (got A=[" << M << "x" << K_a
                       << "], B=[" << K_b << "x" << N
                       << "], C=[" << M_c << "x" << N_c << "])";
        signalPassFailure();
      };

      // Must be statically known for Phase 0.
      if (M < 0 || N < 0 || K_a < 0 || K_b < 0 || M_c < 0 || N_c < 0) {
        fail("Phase-0 NPU requires static shapes");
        return;
      }

      if (K_a != K_b) {
        fail("K dimension mismatch");
        return;
      }
      if (M != M_c || N != N_c) {
        fail("Output shape mismatch");
        return;
      }

      // Check hardware constraints: 8xKx8.
      if (M != kTileM || N != kTileN) {
        fail("Unsupported matmul tile: only 8xKx8 supported in Phase-0");
        return;
      }
      if ((K_a % kKGranularity) != 0) {
        fail("Unsupported K: must be multiple of 8 in Phase-0");
        return;
      }

      // Check element types: i8 x i8 -> i16 (your description).
      Type aElem = aTy.getElementType();
      Type bElem = bTy.getElementType();
      Type cElem = cTy.getElementType();

      auto aInt = dyn_cast<IntegerType>(aElem);
      auto bInt = dyn_cast<IntegerType>(bElem);
      auto cInt = dyn_cast<IntegerType>(cElem);

      if (!aInt || !bInt || !cInt) {
        op.emitError("Phase-0 NPU expects integer element types");
        signalPassFailure();
        return;
      }
      if (aInt.getWidth() != 8 || bInt.getWidth() != 8 || cInt.getWidth() != 16) {
        op.emitError()
            << "Phase-0 NPU expects i8 x i8 -> i16, got "
            << aElem << " x " << bElem << " -> " << cElem;
        signalPassFailure();
        return;
      }

      // (Optional) signedness in MLIR IntegerType is signless; you can enforce
      // your ABI later. For Phase 0, we treat i8 as "your int8".

      // “Dispatch record” – we only print for now.
      // This is the key: you now have a compiler-recognized operation.
      llvm::outs()
          << "NPU_DISPATCH: op=matmul_8xKx8"
          << " K=" << K_a
          << " bram_width_bits=" << kBramWidthBits
          << " act_row_bytes=" << (kTileM * 1)   // 8 * int8
          << " wgt_row_bytes=" << (kTileN * 1)   // 8 * int8 (per col, but same bytes)
          << "\n";
    });
  }
};

}  // namespace

// Public registration hook used by npu-opt.cpp
void registerLowerMatmulToNPUPass() {
  PassRegistration<LowerMatmulToNPUPass>();
}
