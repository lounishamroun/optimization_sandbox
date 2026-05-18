import torch

"""
This function enables its user to compare the execution timing of:
    - `fn_input` V.S `target_fn`.

Features:
    - Performance measurement with warmup option:
        -> e.g: warmup=20, will run
"""
def a_b_timing_bench(fn_a,fn_b,warmup=None|int):
 
    torch.cuda.synchronize()
    start_fn_a=torch.cuda.Event(enable_timing=True)
    start_fn_b=torch.cuda.Event(enable_timing=True)
    end_fn_a=torch.cuda.Event(enable_timing=True)
    end_fn_b=torch.cuda.Event(enable_timing=True)
    
    if warmup is not None:
    #warmup bench
        for _ in range(warmup):
            fn_a()
            fn_b()
    
    #real bench
    torch.cuda.synchronize() #SYNC fn_a
    start_fn_a.record()
    fn_a()
    end_fn_a.record()
    torch.cuda.synchronize() #SYNC fn_a
    
    torch.cuda.synchronize() #SYNC fn_b
    start_fn_b.record()
    fn_b()
    end_fn_b.record()
    torch.cuda.synchronize() #SYNC fn_b
    
    fn_a_timing= start_fn_a.elapsed_time(end_fn_a) #elapsed time fn_a
    fn_b_timing=start_fn_b.elapsed_time(start_fn_b) #elapsed time fn_b
    
    return fn_a_timing,fn_b_timing