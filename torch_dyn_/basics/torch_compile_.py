import torch
from torch import _dynamo as torchdynamo

""" 

Let 'toy_function' be a random function taking a and b as inputs.

"""

def personalized_compiler(gm: torch.fx.graph,example_inputs:int): 
    print("compiling...")
    gm.graph.print_tabular()
    return gm.forward #callable object (function) which will interpret the graph

@torchdynamo.optimize(personalized_compiler)
def toy_function(a,b):
    c=torch.exp(a) + torch.exp(b)
    return c*a

for _ in range(100):
    toy_function(a=torch.randint(low=10,high=20,size=(1,1)),b=torch.randint(low=10,high=20,size=(1,1)))
    
    
""" Steps

If we set the following logging env var :  export TORCH_LOGS="+dynamo,guards,bytecode"

We can analyze the output and find the following steps:

Step 1: torchdynamo start tracing toy_function 
Step 2: calling compiler function 'personalized_compiler' get a modified bytecode

Use guards in order to check specific function behaviour.


If we zoom out we get the following stack: 

Python source
   ↓ CPython compiler
CPython bytecode / code object
   ↓ CPython frame execution
TorchDynamo intercepts the frame
   ↓ bytecode analysis + runtime guards
FX graph of PyTorch tensor ops
   ↓ backend, (e.g: TorchInductor)
optimized CPU/GPU kernels

"""