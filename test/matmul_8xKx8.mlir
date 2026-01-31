module {
  func.func @main(%A: tensor<8x64xi8>, %B: tensor<64x8xi8>, %init: tensor<8x8xi16>)
      -> tensor<8x8xi16> {
    %0 = linalg.matmul
        ins(%A, %B : tensor<8x64xi8>, tensor<64x8xi8>)
        outs(%init : tensor<8x8xi16>)
        -> tensor<8x8xi16>
    return %0 : tensor<8x8xi16>
  }
}
