#include "mlir/Pass/Pass.h"

// Your pass registration hook (defined in LowerMatmulToNPU.cpp).
void registerLowerMatmulToNPUPass();

namespace {
struct NPUPreprocessingPassesRegistration {
  NPUPreprocessingPassesRegistration() { registerLowerMatmulToNPUPass(); }
};
}  // namespace

// Static initializer so that linking this library registers the pass.
static NPUPreprocessingPassesRegistration kReg;
