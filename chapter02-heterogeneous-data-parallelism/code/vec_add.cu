// vec_add.cu
// First CUDA program: vector addition with CUDA error checking.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr,                                                \
                         "CUDA error: %s in %s at line %d\n",                  \
                         cudaGetErrorString(err), __FILE__, __LINE__);          \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__global__ void vecAddKernel(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(const float *A_h, const float *B_h, float *C_h, int n) {
    int size = n * sizeof(float);

    float *A_d = nullptr;
    float *B_d = nullptr;
    float *C_d = nullptr;

    // 1. Allocate device global memory.
    CUDA_CHECK(cudaMalloc((void **)&A_d, size));
    CUDA_CHECK(cudaMalloc((void **)&B_d, size));
    CUDA_CHECK(cudaMalloc((void **)&C_d, size));

    // 2. Copy input vectors from host memory to device global memory.
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // 3. Launch kernel.
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vecAddKernel<<<numBlocks, blockSize>>>(A_d, B_d, C_d, n);

    // Check kernel launch configuration errors.
    CUDA_CHECK(cudaGetLastError());

    // Wait for kernel completion and catch runtime execution errors.
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Copy result vector from device global memory back to host memory.
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    // 5. Free device global memory.
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

int main() {
    const int n = 1 << 20;  // 1,048,576 elements
    const int size = n * sizeof(float);

    float *A_h = (float *)std::malloc(size);
    float *B_h = (float *)std::malloc(size);
    float *C_h = (float *)std::malloc(size);

    if (A_h == nullptr || B_h == nullptr || C_h == nullptr) {
        std::fprintf(stderr, "Host memory allocation failed.\n");
        std::free(A_h);
        std::free(B_h);
        std::free(C_h);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; ++i) {
        A_h[i] = static_cast<float>(i);
        B_h[i] = static_cast<float>(2 * i);
        C_h[i] = 0.0f;
    }

    vecAdd(A_h, B_h, C_h, n);

    bool passed = true;
    for (int i = 0; i < n; ++i) {
        float expected = A_h[i] + B_h[i];
        if (std::fabs(C_h[i] - expected) > 1e-5f) {
            std::fprintf(stderr,
                         "Mismatch at index %d: C=%f, expected=%f\n",
                         i, C_h[i], expected);
            passed = false;
            break;
        }
    }

    if (passed) {
        std::printf("PASS: vector addition result is correct.\n");
        std::printf("Example: C[123] = %f\n", C_h[123]);
    } else {
        std::printf("FAIL: vector addition result is incorrect.\n");
    }

    std::free(A_h);
    std::free(B_h);
    std::free(C_h);

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}