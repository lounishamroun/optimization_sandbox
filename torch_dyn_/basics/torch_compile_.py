import torch
from torch import _dynamo as torchdynamo

""" 

Let 'toy_function' be a random function taking a and b as inputs.

"""

def personalized_compiler(graph_: torch.fx.graph,inputs: int): 
    print("compiling...")
    graph_.graph.print_tabular()
    return graph_.forward() #callable object (function) which will interpret the graph

@torchdynamo.optimize(personalized_compiler)
def toy_function(a,b):
    c=a+b
    return c*a

for _ in range(100):
    toy_function(torch.randn(10), torch.randn(10))