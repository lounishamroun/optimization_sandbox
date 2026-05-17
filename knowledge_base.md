# INDEX



# Concepts


## Torch Dynamo

### Python compilation basics

Python source code is compiled by CPython into bytecode, which is then executed by the Python virtual machine/interpreter.

### Built in 'compile()' function

```python
string_of_code="""
a=2
b=7
print(a+b)
"""
code_object=compile(string_of_code,"<string>",mode="exec")
exec(code_object)
```

#### Explanation

We take a string representing python code, compile it into bytecode using 'CPython' and executing it on Python virtual machine.

CPython is both a compiler and an interpreter. It first compiles Python source code into bytecode, then the Python Virtual Machine interprets and executes that bytecode.

Note: CodeObject are generated at compile time (static) while FrameObject are generated at runtime (dynamic) with info about what is currently happening with the code.







# LESSONS
- Using add_kernel[grid](...) does not work with regular Python function because Kernel overloads the '[]' operator.


# BREAKING POINTS
- Don't set a block sizes which aren't powers of 2.


# CREDITS
- TRITON OFFICAL DOC : https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
- Torch compile doc : https://docs.pytorch.org/docs/2.12/generated/torch.compile.html

