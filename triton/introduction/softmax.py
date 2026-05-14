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

''' TERMINOLOGY 
VGPRs = vector general purpose registers

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
    

'''_ I KERNEL _'''
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

        Be careful not to mix these concepts.

        We have physical padding, which means extra physical memory slots may exist between rows.
        This is one reason why row_stride can be different from the number of columns.
            - Computers do not always store the logical tensor compactly row after row (contiguously).

        But we also have computational padding, used to get the right Triton BLOCK_SIZE.

        Pointers live in the physical memory world, so we use row_stride to move correctly from one row to the next, including any physical padding or layout gaps.

        BLOCK_SIZE lives in the computational/kernel world, so we need a mask to tell Triton to ignore lanes that do not correspond to real columns.
                
        
        '''
        input_ptrs = row_start_ptr + col_offsets #Shift the input pointer by offsets.
        mask = col_offsets < n_cols
        
        # If you're familiar with CUDA, this is analogous to computing per-thread
        # addresses in a block:
        #
        # int idx = blockIdx.x * blockDim.x + threadIdx.x;
        #
        # Here, `row_idx * input_row_stride` gives the start of the row,
        # and `tl.arange(0, BLOCK_SIZE)` gives the per-element offsets inside
        # that row, similar to `threadIdx.x`.

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

        '''Why using a different row stride ? 
        - Even though Softmax doesn't change the output logical shape, like I said physical memory layout can change.
            - We differentiate between input_row_stride and output_row_stride.
                - because the input and output tensors may have the same logical shape but different physical memory layouts.
                  So we differenciate between input/output in-memory tensor row stride.
        
        '''
        tl.store(output_ptrs, softmax_output, mask=mask)


'''_ II - HELPER FUNCTION _'''


properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 8
    
    """
    Conceptually, block i can handle row i of a matrix, so multiple rows
    can be processed in parallel by multiple blocks.

    Inside each block, the row's work is further split among the threads
    of that block. For example, different threads can process different
    columns/components of the row.

    The GPU actually executes those threads in groups called warps
    (usually 32 threads). So block-level parallelism is made of smaller
    warp-level execution underneath.
    
    """
    

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(
        y, 
        x, 
        x.stride(0), 
        y.stride(0), 
        n_rows, n_cols, 
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages, 
        num_warps=num_warps, 
        grid=(1, ))
    
    """
    Retrieve kernel resource usage information in order to later
    compare it against the GPU resource limits
    (e.g. size_smem vs. SIZE_SMEM).
    """
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    
    
    if is_hip(): # True when running on HIP/ROCm (AMD GPUs)
        '''
        Simplified explanation:
            AMD NUM_REGS already represents the full register count,
            while CUDA NUM_REGS represents only half of the available registers (we will not get into details of why).
        '''
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2 

        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE # Number of warps/wavefronts that can run concurrently on one SM/CU
        
        
        """
        Terminology mapping between AMD and NVIDIA/CUDA architectures:
        
        - AMD "compute unit" (CU) ~= NVIDIA "streaming multiprocessor" (SM)
        - AMD "wave" or "wavefront" ~= NVIDIA "warp"
        """

        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps) #register level.
    occupancy = min(occupancy, SIZE_SMEM // size_smem) #register vs shared memory occupancy
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)
    
    ''' OCCUPANCY 
    
    
    
    
    '''
    

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1)) #Torch native Softmax function
        #'do_bench' needs a calable object that's why use 'lambda' instead of a function call to 'softmax()'.
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x)) #Our Kernel
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    #2 * (n elements of the matrix * size of each elements in bytes)* 1e-9 / (duration in ms * 1e-3) 
    '''
    We put a '2*' simply because there's 2 memory operations:
        1/ Read input tensor x
        2/ Write output tensor y
    '''
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)


