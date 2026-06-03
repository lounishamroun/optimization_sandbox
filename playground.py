import torch
from torch import _dynamo as torchdynamo
import torch.nn as nn
import sys
from pathlib import Path
from depyf import decompile
from torch._dynamo.eval_frame import _debug_get_cache_entry_list, innermost_fn
import dis
from boilerplates import benchmarking as bm
import os


def toy_fn(a,b):
    return a+b

if __name__=='__main__':

    print(next(dis.get_instructions(toy_fn)))
    
#46:35