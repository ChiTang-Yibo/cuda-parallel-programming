// vec_add.cu
// Chapter 2 experiment: CUDA vector addition with configurable
// vector size and block size.
//
// Usage:
//   vec_add.exe [vector_size] [block_size]
//
// Examples:
//   vec_add.exe 1000 256
//   vec_add.exe 1048576 256
//   vec_add.exe 1000000 128

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr,                                               \
                         "CUDA error: %s in %s at line %d\n",                  \
                         cudaGetErrorString(err), __FILE__, __LINE__);          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


// ============================================================
// CUDA kernel
// Each thread processes one vector element.
// ============================================================

__global__ void vecAddKernel(
    const float* A,
    const float* B,
    float* C,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}


// ============================================================
// Parse a positive integer command-line argument.
// ============================================================

bool parse_positive_int(const char* text, int& value)
{
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 10);

    if (text == end || *end != '\0') {
        return false;
    }

    if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }

    value = static_cast<int>(parsed);
    return true;
}


// ============================================================
// CUDA vector addition
// ============================================================

void vecAdd(
    const float* A_h,
    const float* B_h,
    float* C_h,
    int n,
    int blockSize)
{
    const std::size_t size =
        static_cast<std::size_t>(n) * sizeof(float);

    float* A_d = nullptr;
    float* B_d = nullptr;
    float* C_d = nullptr;

    // 1. Allocate device global memory.
    CUDA_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_CHECK(cudaMalloc((void**)&C_d, size));

    // 2. Copy input vectors from host to device.
    CUDA_CHECK(cudaMemcpy(
        A_d,
        A_h,
        size,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        B_d,
        B_h,
        size,
        cudaMemcpyHostToDevice));

    // 3. Configure and launch the kernel.
    const int numBlocks =
        (n + blockSize - 1) / blockSize;

    vecAddKernel<<<numBlocks, blockSize>>>(
        A_d,
        B_d,
        C_d,
        n);

    // Check launch errors.
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU execution to finish.
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Copy the result back to the host.
    CUDA_CHECK(cudaMemcpy(
        C_h,
        C_d,
        size,
        cudaMemcpyDeviceToHost));

    // 5. Free device memory.
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}


// ============================================================
// Main
// ============================================================

int main(int argc, char** argv)
{
    // Default experiment configuration.
    int n = 1 << 20;      // 1,048,576 elements
    int blockSize = 256;

    // Optional command-line arguments:
    //   argv[1] -> vector size
    //   argv[2] -> block size
    if (argc >= 2) {
        if (!parse_positive_int(argv[1], n)) {
            std::fprintf(
                stderr,
                "Invalid vector size: %s\n",
                argv[1]);
            return EXIT_FAILURE;
        }
    }

    if (argc >= 3) {
        if (!parse_positive_int(argv[2], blockSize)) {
            std::fprintf(
                stderr,
                "Invalid block size: %s\n",
                argv[2]);
            return EXIT_FAILURE;
        }
    }

    if (argc > 3) {
        std::fprintf(
            stderr,
            "Usage: %s [vector_size] [block_size]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    // Query the current CUDA device.
    int deviceId = 0;
    CUDA_CHECK(cudaGetDevice(&deviceId));

    cudaDeviceProp deviceProp{};
    CUDA_CHECK(cudaGetDeviceProperties(
        &deviceProp,
        deviceId));

    // Validate the requested block size.
    if (blockSize > deviceProp.maxThreadsPerBlock) {
        std::fprintf(
            stderr,
            "Block size %d exceeds this GPU's maximum "
            "threads per block (%d).\n",
            blockSize,
            deviceProp.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    const std::size_t size =
        static_cast<std::size_t>(n) * sizeof(float);

    // Compute launch configuration for reporting.
    const int numBlocks =
        (n + blockSize - 1) / blockSize;

    const long long threadsLaunched =
        static_cast<long long>(numBlocks) * blockSize;

    const long long extraThreads =
        threadsLaunched - n;

    // --------------------------------------------------------
    // Allocate host memory.
    // --------------------------------------------------------

    float* A_h =
        static_cast<float*>(std::malloc(size));

    float* B_h =
        static_cast<float*>(std::malloc(size));

    float* C_h =
        static_cast<float*>(std::malloc(size));

    if (A_h == nullptr ||
        B_h == nullptr ||
        C_h == nullptr)
    {
        std::fprintf(
            stderr,
            "Host memory allocation failed.\n");

        std::free(A_h);
        std::free(B_h);
        std::free(C_h);

        return EXIT_FAILURE;
    }

    // --------------------------------------------------------
    // Initialize input vectors.
    // --------------------------------------------------------

    for (int i = 0; i < n; ++i) {
        A_h[i] = static_cast<float>(i);
        B_h[i] = static_cast<float>(2 * i);
        C_h[i] = 0.0f;
    }

    // --------------------------------------------------------
    // Print experiment configuration.
    // --------------------------------------------------------

    std::printf(
        "CUDA Vector Addition Experiment\n"
        "================================\n");

    std::printf(
        "GPU                 : %s\n",
        deviceProp.name);

    std::printf(
        "Vector size         : %d\n",
        n);

    std::printf(
        "Block size          : %d\n",
        blockSize);

    std::printf(
        "Number of blocks    : %d\n",
        numBlocks);

    std::printf(
        "Threads launched    : %lld\n",
        threadsLaunched);

    std::printf(
        "Extra threads       : %lld\n",
        extraThreads);

    std::printf(
        "Memory per vector   : %zu bytes (%.3f MiB)\n",
        size,
        static_cast<double>(size) /
            (1024.0 * 1024.0));

    std::printf(
        "Total vector memory : %zu bytes (%.3f MiB)\n\n",
        3 * size,
        static_cast<double>(3 * size) /
            (1024.0 * 1024.0));

    // --------------------------------------------------------
    // Run CUDA vector addition.
    // --------------------------------------------------------

    vecAdd(
        A_h,
        B_h,
        C_h,
        n,
        blockSize);

    // --------------------------------------------------------
    // Verify result on the CPU.
    // --------------------------------------------------------

    bool passed = true;

    for (int i = 0; i < n; ++i) {
        const float expected =
            A_h[i] + B_h[i];

        if (std::fabs(
                C_h[i] - expected) > 1e-5f)
        {
            std::fprintf(
                stderr,
                "Mismatch at index %d: "
                "C=%f, expected=%f\n",
                i,
                C_h[i],
                expected);

            passed = false;
            break;
        }
    }

    if (passed) {
        std::printf(
            "Result              : PASS\n");

        int sampleIndex =
            (n > 123) ? 123 : (n - 1);

        std::printf(
            "Sample              : "
            "C[%d] = %f\n",
            sampleIndex,
            C_h[sampleIndex]);
    }
    else {
        std::printf(
            "Result              : FAIL\n");
    }

    // --------------------------------------------------------
    // Release host memory.
    // --------------------------------------------------------

    std::free(A_h);
    std::free(B_h);
    std::free(C_h);

    return passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
