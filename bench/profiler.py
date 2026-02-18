import torch
import torch.nn
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.utils.benchmark as benchmark
import timeit

device="cpu"

if torch.cuda.is_available() == True:
    device="cuda"


print(torch.cuda.is_available())

print(f'Device set to {device}')

tensor_a=torch.randn(size=(1000,1000),device=device)
tensor_b=torch.randn(size=(1000,1000),device=device)

def matmul_ops(a,b): 
    torch.matmul(a,b)

matmul_ops(tensor_a,tensor_b)

t1=timeit.timeit(
    stmt="matmul_ops(tensor_a,tensor_b)",
    setup='from __main__ import matmul_ops',
    globals={'tensor_a':tensor_a,'tensor_b':tensor_b}
    )


print(f'matmul tensor a and b = {t1}')