
## First Results:

Eager model run = 409.3149869986519 ms 
Optimized model run = 471.0841950000031 ms 

Optimized model run 15.1% slower than Optimized model run.

With logits : logits=torch.randn(1000,1024,100).

### Hypothesis 1:

We have to feed bigger logits, since eager mode PyTorch already has a built in optimization.

## Results

By increasing the first dimension of our tensor from 1000 to 6000 *i.e:torch.randn(1000->6000,1024,100)* we get a small improvemement. 

Optimized model run is 10.06% slower than model running in eager mode.


### Hypothesis 2:

The repeated invocation used inside the *optimize* decorator could have an effect on performance.

Let's try to increase the number of repeated calls to the TorchDynamo-optimized function using randomly generated input tensors. Allowing us to evaluate whether repeated invocations with fresh inputs affect TorchDynamo’s graph capture, compilation behavior, and runtime optimization performance.


*for _ in range(10->50):* **Changing invocation number**
   optimized_model(model=model_instance,x=(
         torch.randn([1000,torch.randint(1024,1025,[1]),100] #1000 samples | 1024 features | sequence length = 100 frames
....

### Observation

Our optimized function is now only **3.49% slower** than our eager function.
Optimized function (model) exec duration = 2386.756332999994 ms | Eager function exec duration = 2473.1667500000185 ms 