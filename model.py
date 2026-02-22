# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torchvision.transforms import v2
from PIL import Image
import inspect


if torch.cuda.is_available() == True:
    device="cuda"
else:
    device="cpu"
    
    
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50",device_map=device)
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")


raw_image=Image.open("data/test_img.png",mode='r').convert("RGB") #convert to PIL

transforms=v2.Compose([
    v2.ToImage(),
    v2.Resize(size=(224,224)),
    v2.ConvertImageDtype(torch.float64)
    ]
)

processed_image=transforms(raw_image)

print(type(processed_image))

input=processor(images=processed_image,return_tensors="pt",device=device) #torch.cuda.FloatTensor

with torch.no_grad():
    output=model(**input)

print(output.__dict__)