#include <iostream>
#include <cuda_runtime.h>

__global__ void testKernel() {
    // Simple test: each thread prints its ID if thread 0
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Kernel executed successfully on device 0.\n");
    }
}

int main() {
    // Select GPU 0 explicitly
    cudaSetDevice(0);

    // Check device
    int device;
    cudaGetDevice(&device);
    std::cout << "Using GPU device: " << device << std::endl;

    // Launch a trivial kernel
    testKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Check for async errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Test completed.\n";
    return 0;
}
