# Compilation in PyTorch

## Torch Dynamo

TorchDynamo intercepts Python frame execution, analyzes the function’s bytecode, and traces the PyTorch operations into an FX graph. This FX graph captures the operations performed during the function’s execution.

TorchDynamo uses guards to ensure that assumptions such as shapes, dtypes, and devices are still valid, so the compiled function can execute correctly and be safely reused.

TorchDynamo captures the PyTorch operations from the user code into an FX graph. **AOTAutograd** (also called *AOTDispatcher*) is then used *during training* to also **capture the backward pass**, turning the gradient computation into FX graphs that can be compiled as well.

## Graph Lowering

Before compiling the FX graph into optimized GPU code, PyTorch refines it by decomposing complex tensor operations into simpler primitive operations.

These primitive operations are represented as ATen operators. In the PyTorch compiler stack, many operations are lowered to a smaller, stable set of operators called Core ATen operators. You can think of them as the basic building blocks used by the compiler.

Once the graph is expressed in terms of these Core ATen operators, PyTorch does not simplify it further at the graph level. From there, a backend like TorchInductor lowers the graph into optimized CPU or GPU code.

Example: *aten.cos* applies the cosine operation to a tensor.

## Torch Inductor

Torch Inductor (or another backend) takes this intermediate representation (FX graph) and compiles it into optimized code.

In Python, the default eval-frame function executes bytecode for each frame. **TorchDynamo installs its own frame handler**, which intercepts execution and compiles suitable regions of code into optimized versions.

A cache miss happens when Dynamo cannot reuse a previously compiled version of a region because no cached version matches the current inputs and guards.