



# Torch Errors:

## torch.OutOfMemoryError : CUDA out of memory

### Trigger : When Increasing the tensor size.

Benchmarking a vector addition Triton Kernel was working with those values => tensor_value_range=[2\*\*10, 2\*\*14, 2\*\*18, 2\*\*22, 2\*\*24,2\*\*24].

But adding 2**40 raised an error =>  *Tried to allocate 4096.00 GiB. GPU 0 has a total capacity of 31.84 GiB of which 29.98 GiB is free*.

Obviously the GPU I used has a max capacity of 32607MiB.