'''
ONLY USED IN ORDER TO ISOLATE FUNCTION SNIPPETS
'''

import torch

#in inference optimization, we need to launch a first warmup to avoid measuring useless stuff

start=torch.cuda.Event(enable_timing=True)
end=torch.cuda.Event(enable_timing=True)

start.record()
torch.matmul(torch.randn(size=(20,20)),torch.randn(size=(20,20)))
end.record()

print(start.elapsed_time(end))

print(torch.cuda.get_device_name())