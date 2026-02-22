# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torchvision.transforms import v2
from PIL import Image
import inspect
import re

if torch.cuda.is_available() == True:
    device="cuda:0"
else:
    device="cpu"
    
print(f"Current device is {device}")
    
USE_FP16=True
BATCH=30

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50",device_map=device)
 
if USE_FP16==True:
    model=model.half()

assert torch.device(model.device)==torch.device(device), f"Current device = {device} | model device = {model.device} "
    
img = Image.open("data/test_img.png").convert("RGB")


'''Benchmarking Function'''

def bench(batch,inputs,iterations):
    inputs=processor(images=[img]*batch,return_tensors="pt")
    inputs.to(device=device)
    inputs.to(torch.float16)

    #warmup
    with torch.inference_mode():
        for _ in range(30):
            _=model(**inputs)
    torch.cuda.synchronize()
    
    iters=200
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    
    with torch.inference_mode():
        if USE_FP16==True:
            inputs["pixel_values"]=inputs["pixel_values"].half() #reducing precision of the actual image tensor
        
        start.record()
        for _ in range(iters):
            _=model(**inputs)
        end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end))/iterations


processed_inputs=processor(images=img,return_tensors="pt",device=device)



elapsed_time=bench(BATCH,iterations=200,inputs=processed_inputs)

print(f'Elapsed time is the following : {elapsed_time:>3}ms')