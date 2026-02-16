// LowerMatmulToNPU.cpp
// Phase-1: match linalg.matmul for EXACT 8x8x8 (i8*i8->i16) and REWRITE it to:
//   %y = call @npu.matmul_8x8x8(%a, %b) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi16>
//
// Also auto-inserts a private declaration for @npu.matmul_8x8x8 if missing.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct LowerMatmulToNPUPass
    : public PassWrapper<LowerMatmulToNPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMatmulToNPUPass)

  // Phase-1 target: ONLY 8x8x8 for sanity.
  static constexpr int64_t kM = 8;
  static constexpr int64_t kN = 8;
  static constexpr int64_t kK = 8;
  static constexpr int kBramWidthBits = 64;

  StringRef getArgument() const final { return "lower-matmul-to-npu"; }
  StringRef getDescription() const final {
    return "Phase-1: rewrite 8x8x8 linalg.matmul (i8xi8->i16) to call @npu.matmul_8x8x8";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Important: rewriting while walking can invalidate iterators.
    // Collect ops first, then rewrite.
    SmallVector<linalg::MatmulOp, 8> matmuls;
    module.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });

    for (linalg::MatmulOp op : matmuls) {
      // Require ranked tensors.
      auto aTy = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
      auto bTy = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
      auto cTy = dyn_cast<RankedTensorType>(op.getOutputs()[0].getType());

      if (!aTy || !bTy || !cTy) {
        op.emitError("NPU expects ranked tensor types for matmul operands");
        signalPassFailure();
        return;
      }
      if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2) {
        op.emitError("NPU expects 2D tensors for matmul");
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
        op.emitError()
            << msg
            << " (got A=[" << M << "x" << K_a
            << "], B=[" << K_b << "x" << N
            << "], C=[" << M_c << "x" << N_c << "])";
        signalPassFailure();
      };

      // Must be static for Phase-1.
      if (M < 0 || N < 0 || K_a < 0 || K_b < 0 || M_c < 0 || N_c < 0) {
        fail("NPU requires static shapes");
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

      // Hardware constraint: ONLY 8x8x8.
      if (M != kM || N != kN || K_a != kK) {
        // Not an error: just skip non-matching matmuls (so the rest of the model still compiles).
        // If you want strict behavior, replace with fail(...).
        continue;
      }

      // Types: i8 x i8 -> i16 (accum tensor element type).
      auto aInt = dyn_cast<IntegerType>(aTy.getElementType());
      auto bInt = dyn_cast<IntegerType>(bTy.getElementType());
      auto cInt = dyn_cast<IntegerType>(cTy.getElementType());
      if (!aInt || !bInt || !cInt) {
        op.emitError("NPU expects integer element types");
        signalPassFailure();
        return;
      }
      if (aInt.getWidth() != 8 || bInt.getWidth() != 8 || cInt.getWidth() != 16) {
        op.emitError()
            << "NPU expects i8 x i8 -> i16, got "
            << aTy.getElementType() << " x " << bTy.getElementType()
            << " -> " << cTy.getElementType();
        signalPassFailure();
        return;
      }

      // ---- Rewrite: linalg.matmul -> func.call @npu.matmul_8x8x8 ----
      OpBuilder b(op);
      auto calleeName = StringRef("npu.matmul_8x8x8");

      // Declare the callee if it doesn't exist.
      if (!module.lookupSymbol<func::FuncOp>(calleeName)) {
        OpBuilder mb(module.getBodyRegion());
        mb.setInsertionPointToEnd(module.getBody());
        auto fnType = mb.getFunctionType({aTy, bTy}, {cTy});
        auto decl = mb.create<func::FuncOp>(op.getLoc(), calleeName, fnType);
        decl.setPrivate();
      }

      auto call = b.create<func::CallOp>(
          op.getLoc(), calleeName, TypeRange{cTy},
          ValueRange{op.getInputs()[0], op.getInputs()[1]});

      // linalg.matmul returns a tensor result. Replace it with call result.
      op.getResult(0).replaceAllUsesWith(call.getResult(0));

      // Remove the linalg.matmul op.
      op.erase();

      // Optional: log once per rewrite.
      llvm::errs()
          << "NPU_REWRITE: linalg.matmul -> call @" << calleeName
          << " (8x8x8, bram_width_bits=" << kBramWidthBits << ")\n";
      llvm::errs().flush();
    }
  }
};

}  // namespace

void registerLowerMatmulToNPUPass() {
  PassRegistration<LowerMatmulToNPUPass>();
}