#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"


/// Declared in our pass file:
void registerLowerMatmulToNPUPass();

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Register standard MLIR passes/dialects so npu-opt behaves like mlir-opt.
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  // Register our custom pass.
  registerLowerMatmulToNPUPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "npu-opt\n", registry));
}
