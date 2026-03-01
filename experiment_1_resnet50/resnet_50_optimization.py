# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import statistics as stats

torch._logging.set_logs(graph_breaks=True)
torch.backends.cudnn.benchmark=True
if torch.cuda.is_available() == True:
    device=torch.device(0)
else:
    device="cpu"
    
print(f"Current device is {device}")
    
USE_FP16=True
CL_MEMORY_FORMAT=True #channels last memory format
COMPILE_MODEL=True
BATCH=30

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50").to(device).eval()
 
if USE_FP16==True:
    model=model.half()

if COMPILE_MODEL==True:
    with torch.no_grad():
        compiled_model=torch.compile(model,mode="max-autotune",fullgraph=True) #will raise an error in case of graph break

if CL_MEMORY_FORMAT==True:
    if COMPILE_MODEL==True:
        compiled_model=compiled_model.to(memory_format=torch.channels_last)
    model=model.to(memory_format=torch.channels_last)
    
    
    

assert torch.device(model.device)==torch.device(device), f"Current device = {device} | model device = {model.device} "
    
img = Image.open("data/test_img.png").convert("RGB")


'''Benchmarking Function'''


def format_data(img_data,batch_size):
    '''
    GOAL: Used to generate data once instead of doing it each time we run the 'bench()' function.
    '''
    inputs=processor(images=[img_data]*batch_size,return_tensors="pt")

    pv = inputs["pixel_values"]
    
    if CL_MEMORY_FORMAT==True:
        pv=pv.to(memory_format=torch.channels_last)
        assert pv.is_contiguous(memory_format=torch.channels_last),f"Inputs aren't channels last" 
        
    if USE_FP16==True:
        pv=pv.to(dtype=torch.float16)
        assert pv.dtype == torch.float16, f"Inputs wrong dtype: {pv.dtype} vs fp16"
    
    pv=pv.to(device)
    
    assert pv.device == torch.device(device), f"Inputs on wrong device: {pv.device} vs {device}"
    
    return pv

def bench(batch_size,inputs,iterations=200,warmup=30):
        
    with torch.inference_mode():
        for _ in range(warmup):
            if COMPILE_MODEL==False:
                _=model(inputs)
            else:
                _=compiled_model(inputs)
    torch.cuda.synchronize()
    
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    
    with torch.inference_mode():       
        start.record()
        for _ in range(iterations):
            if COMPILE_MODEL==False:
                _=model(inputs)
            else:
                _=compiled_model(inputs)
        end.record()
        
    torch.cuda.synchronize()
    time_ms=start.elapsed_time(end) #duration for 200 iterations
    avg_ms=time_ms / iterations #duration on avg for 1 iteration
    thr=(batch_size * 1000) / avg_ms 
    return avg_ms, thr


def run_repeats(inputs,batch_size, reps=5):
    avgs = []
    thrs = []
    for _ in range(reps):
        avg_ms, thr = bench(batch_size,inputs=inputs)
        avgs.append(avg_ms)
        thrs.append(thr)

    print(
        f"B={batch_size:>3}\t"
        f"avg_ms={stats.mean(avgs):.3f} ± {stats.pstdev(avgs):.3f}\t"
        f"thr={stats.mean(thrs):.1f} ± {stats.pstdev(thrs):.1f}\t"
    )

if __name__=="__main__":
    
    precision_status="Precision [X]"
    memory_format_status="Channels Last Memory Format [X]"
    model_compilation_status="Model Compiled [X]"
    if USE_FP16==True:
        precision_status="Precision [✓]"
    if CL_MEMORY_FORMAT==True:
        memory_format_status="Channels Last Memory [✓]"   
    if COMPILE_MODEL==True:
        model_compilation_status="Model Compiled [✓]"
    
    print(f'State :{precision_status} | {memory_format_status} | {model_compilation_status}')    
    for b in [1, 8, 32]:
        inputs=format_data(img,batch_size=b)
        run_repeats(inputs,b, reps=5)


# END OF PROGRAM -------------------------------------------------------------------------------------------------- #

''' Bibliography :

https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
https://pytorch.org/blog/accelerating-pytorch-vision-models-with-channels-last-on-cpu/
https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#graph-breaks


'''