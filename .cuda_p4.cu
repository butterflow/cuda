#include <stdio.h>

#define NUM_BLOCKS 1
#define BLOCK_WIDTH 512

__global__ void hello()
{
    printf("Hello world! I'm thread %d\n", threadIdx.x);
}


int main(int argc,char **argv)
{
    
    hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

    
    cudaDeviceSynchronize();

    printf("That's all!\n");

    return 0;
}
