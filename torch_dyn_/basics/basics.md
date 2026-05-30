
## First Results:

Eager model run = 409.3149869986519 ms 
Optimized model run = 471.0841950000031 ms 
Function B is 0.8688786237004814 .X faster than Function A 

With logits : logits=torch.randn(1000,1024,100).

### Hypothesis:

We have to feed bigger logits, since eager mode PyTorch already has a built in optimization.
