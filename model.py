# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torchvision.transforms import v2
from PIL import Image
import inspect
import re
import matplotlib.pyplot as plt
import numpy as np


torch.backends.cudnn.benchmark=True
if torch.cuda.is_available() == True:
    device="cuda:0"
else:
    device="cpu"
    
print(f"Current device is {device}")
    
USE_FP16=True
BATCH=30

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50").to(device).eval()
 
if USE_FP16==True:
    model=model.half()

assert torch.device(model.device)==torch.device(device), f"Current device = {device} | model device = {model.device} "
    
img = Image.open("data/test_img.png").convert("RGB")


'''Benchmarking Function'''

def bench(batch,iterations=200,warmup=30):
    inputs=processor(images=[img]*batch,return_tensors="pt")
    inputs={k: v.to(dtype=torch.float16 if USE_FP16 else torch.float32, device=device, non_blocking=True) for k,v in inputs.items()}
    pv = inputs["pixel_values"]
    assert pv.device == torch.device(device), f"Wrong device: {pv.device} vs {device}"
    assert pv.dtype == torch.float16, f"Wrong dtype: {pv.dtype} vs fp16"

    #warmup
    with torch.inference_mode():
        for _ in range(30):
            _=model(**inputs)
    torch.cuda.synchronize()
    
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    
    with torch.inference_mode():       
        start.record()
        for _ in range(iterations):
            _=model(**inputs)
        end.record()
        
    torch.cuda.synchronize()
    time_ms=start.elapsed_time(end) #duration for 200 iterations
    avg_ms=time_ms / iterations #duration on avg for 1 iteration
    thr=(batch * 1000) / avg_ms 
    return avg_ms, thr


#TO DO : Run the inference 5 times for each batch number to see the average performance progression
'''
def avg_perf():
benchmark_table=numpy.array()
 for _ in range(5):
    for B in [1,8,32]:
    batch,throughput=bench(B)
    print(f"n_batch={B} | throughput={throughput}")
'''

