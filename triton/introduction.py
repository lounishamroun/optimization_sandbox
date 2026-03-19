#import torch

import triton
import triton.language as tl

@triton.jit
def kernel(
    x_ptr, #first input vecor
    y_ptr, #second input vecor
    output_ptr,
    n_elements,
    BLOCK_SIZE=tl.constexpr,
)