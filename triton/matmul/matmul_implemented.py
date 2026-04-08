import triton

'''MULTIPLY A=(M,K) | B=(K,M) => A@B'''

'''Attempt 1'''
#pid*start_ptr_m + tl.arange(0,BLOCKSIZE_M) % M => in case m is not a multiple of blocksize


    

    