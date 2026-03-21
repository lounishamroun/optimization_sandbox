#import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,
               y_ptr,  
               output_ptr,  
               n_elements,  
               BLOCK_SIZE: tl.constexpr, 
               # NOTE: 
               ):
    pid=tl.program_id(axis=0)
    block_start=pid*BLOCK_SIZE
    offsets=block_start+tl.arange(0,BLOCK_SIZE)
    mask=offsets<n_elements #The pointer step size should never exceed the end of a fixed block size
    
    '''
    IN DRAM WE READ 2 VECTORS STARTING FROM THEIR RESPECTIVE POINTER + STEP SIZE
    '''
    x=tl.load(x_ptr+offsets,mask=mask)
    y=tl.load(y_ptr+offsets,mask=mask)
    
    '''
    THEN WE WRITE THE RESULT BACK IN DRAM
    '''
    output=x+y
    tl.store(output_ptr+offsets,output,mask=mask)
    
    
if __name__=='__main__':
    pass