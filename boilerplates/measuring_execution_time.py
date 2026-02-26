import torch

def bench(target_fn,warmup=20):

    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    
    #warmup bench
    for _ in warmup:
        target_fn()
    
    #real bench
    start.record()
    target_fn()
    end.record()
    torch.cuda.synchronize()
    
    time_ms=start.elapsed_time(end) #elapsed time
    return time_ms
