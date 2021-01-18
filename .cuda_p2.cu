#include <stdio.h>


__global__ void use_local_memory_GPU(float in)
{
    float f;    // variable "f" is in local memory and private to each thread
    f = in;     // parameter "in" is in local memory and private to each thread
}


__global__ void use_global_memory_GPU(float *array)
{
    array[threadIdx.x] = 2.0f * (float) threadIdx.x;
}


__global__ void use_shared_memory_GPU(float *array)
{
    int i, index = threadIdx.x;
    float average, sum = 0.0f;

    __shared__ float sh_arr[128];

    sh_arr[index] = array[index];

    __syncthreads();    // ensure all the writes to shared memory have completed

    for (i=0; i<index; i++) { sum += sh_arr[i]; }
    average = sum / (index + 1.0f);

	printf("Thread id = %d\t Average = %f\n",index,average);
    if (array[index] > average) { array[index] = average; }

    sh_arr[index] = 3.14;
}

int main(int argc, char **argv)
{
    use_local_memory_GPU<<<1, 128>>>(2.0f);

    float h_arr[128];   // convention: h_ variables live on host
    float *d_arr;       // convention: d_ variables live on device (GPU global mem)

    cudaMalloc((void **) &d_arr, sizeof(float) * 128);
    cudaMemcpy((void *)d_arr, (void *)h_arr, sizeof(float) * 128, cudaMemcpyHostToDevice);
    use_global_memory_GPU<<<1, 128>>>(d_arr);  // modifies the contents of array at d_arr
    cudaMemcpy((void *)h_arr, (void *)d_arr, sizeof(float) * 128, cudaMemcpyDeviceToHost);


    use_shared_memory_GPU<<<1, 128>>>(d_arr); 
    cudaMemcpy((void *)h_arr, (void *)d_arr, sizeof(float) * 128, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    return 0;
}

