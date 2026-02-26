import torch

def micro_bench(fn):
    start=torch.cuda.Event(enable_timing=True)
    stop=torch.cuda.Event(enable_timing=True)
    start.record()
    function_return=fn()
    stop.record()
    torch.cuda.synchronize()
    elapsed_time_=start.elapsed_time(stop)
    return (elapsed_time_/1000)


def random_fn():
    tensor_1=torch.randn(size=(2,2))
    tensor_2=torch.randn(size=(2,2))
    torch.matmul(tensor_1,tensor_2)

@torch.compile
def compiled_random_fn():
    tensor_1=torch.randn(size=(2,2))
    tensor_2=torch.randn(size=(2,2))
    torch.matmul(tensor_1,tensor_2)
    
if __name__=="__main__":
    print(micro_bench(random_fn))
    print(micro_bench(compiled_random_fn))