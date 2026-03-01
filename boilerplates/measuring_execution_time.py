import torch

def bench(fn_input,target_fn,warmup=20):

    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    
    #warmup bench
    for _ in range(warmup):
        target_fn(fn_input)
    
    #real bench
    start.record()
    target_fn(fn_input)
    end.record()
    torch.cuda.synchronize()
    
    time_ms=start.elapsed_time(end) #elapsed time
    return time_ms
