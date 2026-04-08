import triton

'''MULTIPLY A=(M,K) | B=(K,M) => A@B'''

'''Attempt 1'''
@triton_jit
for m in range(M,SIZE_M):#program 1
    for k in range(K,SIZE_K):#program 2
        acc=zero([m;m:BLOCKSIZE_M],[k;k:BLOCKSIZE_K]) #parallel tile computation, has to be reset at 0
        for l in range(l,SIZE_l):
            A=[m;m:BLOCKSIZE_M]
            B=[k;k:BLOCKSIZE_K]
            k=zero[l;m:BLOCKSIZE_l]
    acc=A@B
    k=acc

    

    