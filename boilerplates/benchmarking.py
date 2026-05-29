import torch

"""
This function enables its user to compare the execution timing of:
    - `fn_input` V.S `target_fn`.

Args :
    - fn_a : Any callable.
    - args_a : Arguments that this callabl will take in order for it to execute.

Features:
    - Performance measurement with warmup option:
        -> e.g: warmup=20, will run
"""
def a_b_timing_bench(fn_a,
                     args_a,
                     fn_b,
                     args_b,
                     warmup=None|int):
    
    if not torch.cuda.is_available():
        warnings.warn("No CUDA instance detected.")
 

    torch.cuda.synchronize()
    start_fn_a=torch.cuda.Event(enable_timing=True)
    start_fn_b=torch.cuda.Event(enable_timing=True)
    end_fn_a=torch.cuda.Event(enable_timing=True)
    end_fn_b=torch.cuda.Event(enable_timing=True)
    
    if warmup is not None:
    #warmup bench
        for _ in range(warmup):
            fn_a(*args_a)
            fn_b(*args_b)
    
    #real bench
    torch.cuda.synchronize() #SYNC fn_a
    start_fn_a.record()
    fn_a(*args_a)
    end_fn_a.record()
    torch.cuda.synchronize() #SYNC fn_a
    
    torch.cuda.synchronize() #SYNC fn_b
    start_fn_b.record()
    fn_b(*args_b)
    end_fn_b.record()
    torch.cuda.synchronize() #SYNC fn_b
    
    fn_a_timing=start_fn_a.elapsed_time(end_fn_a) #elapsed time fn_a
    fn_b_timing=start_fn_b.elapsed_time(start_fn_b) #elapsed time fn_b
    
    print(f'Function A duration = {fn_a_timing} ms | Function B duration = {fn_b_timing} ms ')
    print(f'Function B is {fn_a_timing/fn_b_timing} .X faster than Function A ')

    return fn_a_timing,fn_b_timing