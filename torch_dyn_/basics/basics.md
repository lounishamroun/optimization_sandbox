
## First Results:

Eager model run = 409.3149869986519 ms 
Optimized model run = 471.0841950000031 ms 

Optimized model run 15.1% slower than Optimized model run.

With logits : logits=torch.randn(1000,1024,100).

### Hypothesis:

We have to feed bigger logits, since eager mode PyTorch already has a built in optimization.
