# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import statistics as stats

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
    
compiled_model=torch.compile(model,mode="max-autotune")



assert torch.device(model.device)==torch.device(device), f"Current device = {device} | model device = {model.device} "
    
img = Image.open("data/test_img.png").convert("RGB")


'''Benchmarking Function'''


def format_data(img_data,batch_size):
    '''
    GOAL: Used to generate data once instead of doing it each time we run the 'bench()' function.
    '''
    inputs=processor(images=[img_data]*batch_size,return_tensors="pt")
    inputs={k: v.to(dtype=torch.float16 if USE_FP16 else torch.float32, device=device, non_blocking=True) for k,v in inputs.items()}
    pv = inputs["pixel_values"]
    assert pv.device == torch.device(device), f"Wrong device: {pv.device} vs {device}"
    assert pv.dtype == torch.float16, f"Wrong dtype: {pv.dtype} vs fp16"
    return inputs

def bench(batch_size,inputs,model,iterations=200,warmup=30):
        
    with torch.inference_mode():
        for _ in range(warmup):
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
    thr=(batch_size * 1000) / avg_ms 
    return avg_ms, thr


def run_repeats(inputs,batch_size,model=model, reps=5):
    avgs = []
    thrs = []
    for _ in range(reps):
        avg_ms, thr = bench(batch_size,inputs=inputs,model=model)
        avgs.append(avg_ms)
        thrs.append(thr)

    print(
        f"B={batch_size:>3}\t"
        f"avg_ms={stats.mean(avgs):.3f} ± {stats.pstdev(avgs):.3f}\t"
        f"thr={stats.mean(thrs):.1f} ± {stats.pstdev(thrs):.1f}\t"
    )

if __name__=="__main__":
    
    print("EAGER")
    for b in [1, 8, 32]:
        inputs=format_data(img,batch_size=b)
        run_repeats(inputs,b, reps=5)
    
    print("COMPILED")
    for b in [1, 8, 32]:
        inputs=format_data(img,batch_size=b)
        run_repeats(inputs,b,model=compiled_model,reps=5)
    
    