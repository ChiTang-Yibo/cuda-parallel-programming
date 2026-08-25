#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                    \
    } while (0)

// One CUDA thread computes one output vector element A[row].
__global__
void matrixVectorMulKernel(float* A,
                           const float* B,
                           const float* C,
                           int width)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width) {
        float value = 0.0f;

        for (int k = 0; k < width; ++k) {
            value += B[row * width + k] * C[k];
        }

        A[row] = value;
    }
}

// Host stub with the four parameters required by the exercise.
void matrixVectorMul(float* h_A,
                     const float* h_B,
                     const float* h_C,
                     int width)
{
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    size_t matrixBytes =
        static_cast<size_t>(width) * width * sizeof(float);
    size_t vectorBytes =
        static_cast<size_t>(width) * sizeof(float);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), vectorBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), matrixBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), vectorBytes));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B, matrixBytes, cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_C, h_C, vectorBytes, cudaMemcpyHostToDevice
    ));

    dim3 block(256, 1, 1);
    dim3 grid((width + block.x - 1) / block.x, 1, 1);

    matrixVectorMulKernel<<<grid, block>>>(d_A, d_B, d_C, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_A, d_A, vectorBytes, cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void initializeMatrix(std::vector<float>& matrix)
{
    for (size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = static_cast<float>(i % 19) / 19.0f;
    }
}

void initializeVector(std::vector<float>& vector)
{
    for (size_t i = 0; i < vector.size(); ++i) {
        vector[i] = static_cast<float>((i + 3) % 13) / 13.0f;
    }
}

bool checkSamples(const std::vector<float>& B,
                  const std::vector<float>& C,
                  const std::vector<float>& A,
                  int width)
{
    const int samples[5] = {
        0,
        width / 4,
        width / 2,
        3 * width / 4,
        width - 1
    };

    for (int row : samples) {
        float reference = 0.0f;

        for (int k = 0; k < width; ++k) {
            reference += B[row * width + k] * C[k];
        }

        float tolerance =
            1.0e-3f * std::max(1.0f, std::fabs(reference));

        if (std::fabs(reference - A[row]) > tolerance) {
            std::cerr << "Check failed at A[" << row << "]"
                      << ": reference = " << reference
                      << ", GPU = " << A[row] << std::endl;
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    int width = 1024;

    if (argc >= 2) {
        width = std::atoi(argv[1]);
    }

    if (width <= 0) {
        std::cerr << "Width must be positive." << std::endl;
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaFree(0));

    std::vector<float> h_B(
        static_cast<size_t>(width) * width
    );
    std::vector<float> h_C(width);
    std::vector<float> h_A(width, 0.0f);

    initializeMatrix(h_B);
    initializeVector(h_C);

    matrixVectorMul(
        h_A.data(),
        h_B.data(),
        h_C.data(),
        width
    );

    bool passed = checkSamples(h_B, h_C, h_A, width);

    dim3 block(256, 1, 1);
    dim3 grid((width + block.x - 1) / block.x, 1, 1);

    std::cout << "Exercise 2: matrix-vector multiplication\n";
    std::cout << "Matrix size : " << width << " x " << width << '\n';
    std::cout << "Vector size : " << width << '\n';
    std::cout << "Block       : (" << block.x << ", 1, 1)\n";
    std::cout << "Grid        : (" << grid.x << ", 1, 1)\n";
    std::cout << "Sample check: "
              << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
