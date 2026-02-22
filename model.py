# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torchvision.transforms import v2
from PIL import Image
import inspect
import re

USE_FP16=True


if torch.cuda.is_available() == True:
    device="cuda"
else:
    device="cpu"
    
img = Image.open("data/test_img.png").convert("RGB")

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50",device_map=device)

input=processor(images=img,return_tensors="pt",device=device) #torch.cuda.FloatTensor

model.eval()
with torch.no_grad():
    output=model(**input).logits

print(int(torch.cuda.memory_reserved())/10000000)