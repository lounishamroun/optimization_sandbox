# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torchvision.transforms import v2
from PIL import Image
import inspect
import re

if torch.cuda.is_available() == True:
    device="cuda"
else:
    device="cpu"
    
USE_FP16=True

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50",device_map=device)
 
if USE_FP16==True:
    model=model.half()

    
img = Image.open("data/test_img.png").convert("RGB")


'''Benchmarking Function'''

def bench(batch):
    inputs=processor(images=[img]*batch,return_tensors="pt")
    inputs={k:v.to_device(device=device,non_blocking=True) for k,v in inputs.items()}

        
    with torch.inference_mode():
        for _ in range(30):
            _=model(**inputs)
    torch.cuda.synchronize()
    
    iters=200
    start=torch.cuda.Event(elapsed_time=True)
    end=torch.cuda.Event(elapsed_time=True)
    
    with torch.inference_mode():
        if USE_FP16==True:
            inputs["pixel_values"]=inputs["pixel_values"].half() #reducing precision of the actual image tensor
        start.record()
        for _ in range(iters):
            _=model(**inputs)
        end.record()
    torch.cuda.synchronize()   



input=processor(images=img,return_tensors="pt",device=device) #torch.cuda.FloatTensor
