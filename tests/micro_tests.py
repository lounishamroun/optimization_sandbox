'''
ONLY USED IN ORDER TO ISOLATE FUNCTION SNIPPETS
'''

import torch

start=torch.cuda.Event(enable_timing=True)
stop=torch.cuda.Event(enable_timing=True)

start.record()
torch.matmul(torch.randn(size=(10,5)),torch.randn(size=(5,10)))
stop.record()

torch.cuda.synchronize()

print(start.elapsed_time(stop))

