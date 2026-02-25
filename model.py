# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torchvision.transforms import v2
from PIL import Image
import inspect
import re
import matplotlib.pyplot as plt

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
    inputs=processor(images=[img]*batch,return_tensors="pt",device=device)
    inputs={k: v.to(dtype=torch.float16, device=device) for k,v in inputs.items()}
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
        if USE_FP16==True:
            inputs["pixel_values"]=inputs["pixel_values"].half() #reducing precision of the actual image tensor
        
        start.record()
        for _ in range(iterations):
            _=model(**inputs)
        end.record()
    torch.cuda.synchronize()
    
    time_ms=start.elapsed_time(end) #duration for 200 iterations
    avg_ms_per_iteration=time_ms / iterations #duration on avg for 1 iteration
    throughput=(batch * 1000) / avg_ms_per_iteration 
    return throughput


test_processor=processor(images=img,return_tensors="pt",device=device)

elapsed_time=[]

for B in [1,5,30]:
    elapsed_time.append(bench(B))

plt.plot(elapsed_time, [1,5,30], color="red")
plt.xlabel("Elapsed Time (ms)")
plt.ylabel("Number of Batches")
plt.title("Batches vs Time")
plt.show()
