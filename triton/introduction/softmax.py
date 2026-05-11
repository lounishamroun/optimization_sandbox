import torch
import triton
import triton.language as tl
from triton.runtime import driver
import numpy as np
import statistics
import pandas as pd 
import matplotlib.pyplot as plt 
from boilerplates.reverse_engineering import reverse_engineer_debug_ as red 


''' KEY TRITON CONCEPTS

A kernel can be simplified as being a mathematical operation.

Let X be a vector of size S=3072. The kernel being => 'subtract the number 1 to each element E of X'.

Instead of applying the kernel (substraction operation) to every E in one pass.

We'll divide X by a value called the 'block_size' which must be a power of 2,
which is basically the number of element E we want each of our kernel instance to process.

Let set our block_size to 1024.

Hence we'll find ourselve with 3 blocks of E (3072/1024 = 3) each of size=1024.

We can also say that we're launching a grid of size 3.

To which we'll assign a program id, being clones of our original Kernel,
hence we'll be able to apply 3 kernels in parallel on 3 different portion of our original vector X.
   
'''



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


''' M×N Row-wise Softmax

Even though softmax is defined on a single logit vector of dimension N, in practice we often compute many independent softmaxes at once.

So the input is represented as an M×N matrix, where each row is one logit vector of length N.

The kernel applies softmax independently to each row.
'''
def naive_softmax(logits): # N*D
    x_max = logits.max(dim=1)[0] 
    print(f'From x tensor of shape : {logits.shape}, \n we substract the max values (shape:{x_max[:,None].shape}) of each row from it')
    z=logits-x_max.unsqueeze(1)
    #N*D - N*1
    numerator=torch.exp(logits)
    denominator=numerator.sum(dim=1) #sum logits for each neuron
    softmax=numerator/denominator.unsqueeze(1) #Add a dimension, equivalent to [:,None]
    return softmax
    

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        
        '''
        The stride represents how much we need to increase the pointer to advance by one row.

        Intuitively, we could ask: why not just use the number of columns?

        Because we should not mix up the logical shape of the tensor with its physical memory layout.

        For a contiguous tensor with 3 columns, the next row starts 3 elements later, so row_stride = 3.

        But the tensor may be stored with extra physical padding or may be a non-contiguous view. 
        In that case, even if the tensor logically has 3 columns, the next row may start 4, 5, ...

        Example:

        logical tensor:
        row 0: a b c
        row 1: d e f

        physical memory:
        a b c PAD d e f PAD

        Here:
        n_cols = 3
        row_stride = 4

        '''

        row_start_ptr = input_ptr + row_idx * input_row_stride
        

        col_offsets = tl.arange(0, BLOCK_SIZE) 
        
        ''' /!\ Physical vs BLOCK_SIZE padding /!\ 

        Be careful to not mix those concepts.

        We have the physical padding which are physical memory padded blocks (reason for using stride != number of columns).
            - Since computers do not always store memory contiguously.    
        But we also have logical padding used in order to have the right block size (more on that later).
        Pointers lies in the physical world so we'll use the row stride in order to count physical memory padding (not seen by triton).
        Block size lie in the logical world so we'll need a mask to tell Triton to ignore those columns.
        
        
        '''
        input_ptrs = row_start_ptr + col_offsets #Shift the input pointer by offsets.
        mask = col_offsets < n_cols

        ''' Since we want our BLOCK_SIZE to be the next power of 2 wrt the number of columns of the original tensor, 
            let's take the following case:
        
                E.G: BLOCK_SIZE=4 | Tensor Col number = 3 
                
                col_offsets=tl.arange(0, BLOCK_SIZE)=> [0,1,2,3] This will generate a vector containing 4 indexes.

                However since our true tensor has only 3 columns we'll mask the last column.
                
                mask=col_offsets<n_cols => The last column will be masked as such : [0,1,2,-infinity], we apply a sort of "padding".

        '''




        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

''' Explanation (in my own words)


Why aren't we starting with columns instead of rows?


'''



