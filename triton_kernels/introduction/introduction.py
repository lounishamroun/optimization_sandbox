import torch
import triton
import triton.language as tl
import numpy as np
import statistics
import pandas as pd 
import matplotlib.pyplot as plt 




DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@triton.jit
def _add_kernel(x_ptr, #private function
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
    
def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    BLOCK_SIZE=1024
    
    '''<ERROR HANDLING'''
    if BLOCK_SIZE <= 0 or BLOCK_SIZE & (BLOCK_SIZE - 1) != 0:
        raise ValueError("BLOCK_SIZE must be a positive power of 2")
    
    #assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE,f"Device aren't the same, X Device is :{x.device}, Y Device is :{y.device}, Output Device is :{y.device}"
    
    '''ERROR HANDLING>'''
    
    n_elements = output.numel() #n_elements inside the output tensor
    
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
        
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    
    _add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE) # => An helper function acts as an intermediate before calling the target function
    #:/!\ Block sizes must always be power of 2 /!\
    
    return output

def torch_style_bench(triton_fn, *args, n_calls=1000):
    
    x,y=args
    
    torch.testing.assert_close(add(x,y),x+y) #Check value correctness
    
    start_torch=torch.cuda.Event(enable_timing=True)
    stop_torch=torch.cuda.Event(enable_timing=True)
    start_triton=torch.cuda.Event(enable_timing=True)
    stop_triton=torch.cuda.Event(enable_timing=True) 
   
    '''BENCHING NATIVE TORCH FUNCTION'''
    torch.cuda.synchronize()
    start_torch.record()
    for _ in range(n_calls):
        x+y
    stop_torch.record()
    torch.cuda.synchronize()
    
    '''BENCHING TRITON KERNEL'''
    torch.cuda.synchronize()
    start_triton.record()
    for _ in range(n_calls):
        triton_fn(x,y)
    stop_triton.record()
    torch.cuda.synchronize()
    
    
    torch_fn_duration_ms=start_torch.elapsed_time(stop_torch)/1000
    triton_fn_duration_ms=start_triton.elapsed_time(stop_triton)/1000
    return torch_fn_duration_ms,triton_fn_duration_ms


def benchmark_viz(bench_value,tensor_value_range):
    
    '''
    Parameters
    bench_value : np.array(n,n)
    shape(bench_value) => (10,8) => Benchmark repeated 10 times for 8 different tensor size 
    
    Return
    pd.DataFrame
    '''
    
    col_means=[statistics.fmean(bench_value[:,col_idx]) for col_idx in range(np.shape(bench_value)[1])]
    
    stringified_tensor_size=[str(x) for x in tensor_value_range]    

    
    benchmark_df=pd.DataFrame(
        data=np.array(col_means).reshape(1, -1),
        columns=stringified_tensor_size
        )
    
    return benchmark_df

def matplotlib_bench():
    print("Comparing Torch VS Triton")
    
    torch.manual_seed(0)
    warmp_x=torch.rand(2**4,device=DEVICE)
    warmp_y=torch.rand(2**4,device=DEVICE)
    
    '''TORCH WARMUP'''
    for _ in range(30):
        warmp_x+warmp_y
        
    '''TRITON WARMUP'''
    for _ in range(30):
        add(warmp_x,warmp_y)
        
        
    #TO DO : Include memory guardrail for tensor size
    
    # /!\ Make sure your GPU has enough memory to run tensors of such size /!\
    tensor_value_range=[2**10, 2**14, 2**18, 2**22, 2**24]
    torch_fn_avg_duration_ms=[]
    triton_fn_avg_duration_ms=[]


    for tensor_size in tensor_value_range:
        x=torch.rand(tensor_size,device=DEVICE)
        y=torch.rand(tensor_size,device=DEVICE)
        torch_fn_duration_ms,triton_fn_duration_ms=torch_style_bench(add,x,y)
        torch_fn_avg_duration_ms.append(torch_fn_duration_ms)
        triton_fn_avg_duration_ms.append(triton_fn_duration_ms)
    
    benchmark_viz_torch=benchmark_viz(np.array(torch_fn_avg_duration_ms).reshape(1,-1),tensor_value_range)
    benchmark_viz_triton=benchmark_viz(np.array(triton_fn_avg_duration_ms).reshape(1,-1),tensor_value_range)
    print(f"Triton Benchmark {benchmark_viz_triton} VS \n Torch Benchmark {benchmark_viz_torch}  ")
    
    # TO DO : Matplotlib plot bench torch vs triton
    ''' DATAVIZ '''
    
    torch_ms = benchmark_viz_torch.to_numpy().squeeze()
    triton_ms = benchmark_viz_triton.to_numpy().squeeze()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tensor_value_range, torch_ms, c='r', linewidth=2, label='Torch')
    ax.plot(tensor_value_range, triton_ms, c='g', linewidth=2, label='Triton')
    ax.set_ylabel('Duration (ms)')
    ax.set_xlabel('Tensor size')
    ax.legend()
    plt.show()
    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  
        x_vals=[2**i for i in range(12, 28, 1)],  
        x_log=True, 
        line_arg='provider',  
        line_vals=['triton', 'torch'],  
        line_names=['Triton', 'Torch'], 
        styles=[('blue', '-'), ('green', '-')],  
        ylabel='GB/s',  
        plot_name='vector-add-performance',  
        args={},  
    ))

def benchmark(size,provider):
    x=torch.rand(size,device=DEVICE,dtype=torch.float32)
    y=torch.rand(size,device=DEVICE,dtype=torch.float32)
    quantiles=[0.2,0.5,0.8]
    if provider=='triton':
        ms,min_ms,max_ms=triton.testing.do_bench(lambda:add(x,y),quantiles=quantiles)
        
    elif provider=='torch':
        ms,min_ms,max_ms=triton.testing.do_bench(lambda:x+y,quantiles=quantiles)
        
    gbps = lambda ms:3*x.numel()*x.element_size()*1e-9 /ms*1e-3 #=> N values of X * How much bytes per value * Conversion in GB
    return gbps(ms),gbps(min_ms),gbps(max_ms)

if __name__=="__main__":
    benchmark.run(print_data=True, show_plots=True)
    
    