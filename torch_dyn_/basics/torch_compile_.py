import torch
from torch import _dynamo as torchdynamo
import torch.nn as nn
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from boilerplates import benchmarking as bm 


""" 
TorchDynamo is part of PyTorch’s *just-in-time compiler system. 
It looks at Python bytecode while the program is running, 
finds the PyTorch operations, and turns them into a graph. 
This graph can then be optimized and compiled to run faster.

*just-in-time compiler : 
   - A JIT compiler uses bytecode or another *intermediate representation 
   that was generated before execution, then compiles it into machine code at runtime when needed

*intermediate representation (IR):
   A format that source code is converted into so it can be more easily analyzed, optimized, or compiled.

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
   rec_=toy_function(a=torch.randint(low=10,high=20,size=(1,1)),b=torch.randint(low=10,high=20,size=(1,1)))


    
    
""" 

If we set the following logging env var :  export TORCH_LOGS="+dynamo,guards,bytecode"

We can analyze the output and find the following steps:

Step 1: torchdynamo start tracing toy_function 
Step 2: calling compiler function 'personalized_compiler' get a modified bytecode

## Guards

TorchDynamo caches the optimized FX graph of a function. 
When we call the optimized function again, it checks the guards to see if the cached graph is still valid. 
If it is valid, TorchDynamo reuses it instead of compiling again.

## Steps
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
class toy_model(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear=nn.linear(in_features=1024,out_features=64)
      self.relu=nn.relu()
      self.conv_1d=nn.conv1d(in_channels=64,out_channels=32,kernel=1)
      self.conv_1d_bis=nn.conv1d(in_channels=32,out_channels=16,kernel=1)
      
   def forward(self,x):
      n=nn.sequential(
         self.linear(),
         self.relu(),
         self.conv_1d()
      )
      return n(x)


""" 
*custom_eval_frame -> check already compiled code in the cache.

context manager -> c code

"""
