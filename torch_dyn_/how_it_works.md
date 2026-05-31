
## Torch Dynamo

TorchDynamo intercepts Python frame execution, analyzes the function’s bytecode, and traces the PyTorch operations into an FX graph, while recording guards for assumptions like shapes, dtypes, devices, and control-flow choices.

Torch Inductor (or another backend) takes this intermediate representation (FX graph) and compiles it into optimized code.
