#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
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

// Exercise 1(a): one CUDA thread computes one complete output row.
__global__
void matMulRowKernel(const float* M,
                     const float* N,
                     float* P,
                     int width)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width) {
        for (int col = 0; col < width; ++col) {
            float value = 0.0f;

            for (int k = 0; k < width; ++k) {
                value += M[row * width + k] * N[k * width + col];
            }

            P[row * width + col] = value;
        }
    }
}

// Exercise 1(b): one CUDA thread computes one complete output column.
__global__
void matMulColumnKernel(const float* M,
                        const float* N,
                        float* P,
                        int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width) {
        for (int row = 0; row < width; ++row) {
            float value = 0.0f;

            for (int k = 0; k < width; ++k) {
                value += M[row * width + k] * N[k * width + col];
            }

            P[row * width + col] = value;
        }
    }
}

void initializeMatrix(std::vector<float>& matrix, int offset)
{
    for (size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = static_cast<float>((static_cast<int>(i) + offset) % 17) /
                    17.0f;
    }
}

bool checkSamples(const std::vector<float>& M,
                  const std::vector<float>& N,
                  const std::vector<float>& P,
                  int width)
{
    const int samples[5][2] = {
        {0, 0},
        {0, width - 1},
        {width - 1, 0},
        {width - 1, width - 1},
        {width / 2, width / 2}
    };

    for (const auto& sample : samples) {
        int row = sample[0];
        int col = sample[1];

        float reference = 0.0f;
        for (int k = 0; k < width; ++k) {
            reference += M[row * width + k] * N[k * width + col];
        }

        float result = P[row * width + col];
        float tolerance = 1.0e-3f * std::max(1.0f, std::fabs(reference));

        if (std::fabs(reference - result) > tolerance) {
            std::cerr << "Check failed at P[" << row << "][" << col << "]"
                      << ": reference = " << reference
                      << ", GPU = " << result << std::endl;
            return false;
        }
    }

    return true;
}

float timeRowKernel(const float* d_M,
                    const float* d_N,
                    float* d_P,
                    int width,
                    dim3 grid,
                    dim3 block,
                    int repeat)
{
    matMulRowKernel<<<grid, block>>>(d_M, d_N, d_P, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        matMulRowKernel<<<grid, block>>>(d_M, d_N, d_P, width);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / repeat;
}

float timeColumnKernel(const float* d_M,
                       const float* d_N,
                       float* d_P,
                       int width,
                       dim3 grid,
                       dim3 block,
                       int repeat)
{
    matMulColumnKernel<<<grid, block>>>(d_M, d_N, d_P, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        matMulColumnKernel<<<grid, block>>>(d_M, d_N, d_P, width);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / repeat;
}

int main(int argc, char** argv)
{
    int width = 256;
    int repeat = 20;

    if (argc >= 2) {
        width = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        repeat = std::atoi(argv[2]);
    }

    if (width <= 0 || repeat <= 0) {
        std::cerr << "Width and repeat must be positive." << std::endl;
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaFree(0));

    size_t elements = static_cast<size_t>(width) * width;
    size_t bytes = elements * sizeof(float);

    std::vector<float> h_M(elements);
    std::vector<float> h_N(elements);
    std::vector<float> h_P(elements, 0.0f);

    initializeMatrix(h_M, 0);
    initializeMatrix(h_N, 5);

    float* d_M = nullptr;
    float* d_N = nullptr;
    float* d_P = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_M), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_N), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_P), bytes));

    CUDA_CHECK(cudaMemcpy(d_M, h_M.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N.data(), bytes, cudaMemcpyHostToDevice));

    // Both designs need only width threads, so a one-dimensional grid is used.
    dim3 block(256, 1, 1);
    dim3 grid((width + block.x - 1) / block.x, 1, 1);

    std::cout << "Exercise 1: matrix-matrix multiplication variants\n";
    std::cout << "Matrix size : " << width << " x " << width << '\n';
    std::cout << "Block       : (" << block.x << ", 1, 1)\n";
    std::cout << "Grid        : (" << grid.x << ", 1, 1)\n";
    std::cout << "Repeat      : " << repeat << "\n\n";

    float rowAvgMs =
        timeRowKernel(d_M, d_N, d_P, width, grid, block, repeat);

    CUDA_CHECK(cudaMemcpy(
        h_P.data(), d_P, bytes, cudaMemcpyDeviceToHost
    ));

    bool rowPassed = checkSamples(h_M, h_N, h_P, width);

    float columnAvgMs =
        timeColumnKernel(d_M, d_N, d_P, width, grid, block, repeat);

    CUDA_CHECK(cudaMemcpy(
        h_P.data(), d_P, bytes, cudaMemcpyDeviceToHost
    ));

    bool columnPassed = checkSamples(h_M, h_N, h_P, width);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "One thread per row\n";
    std::cout << "  Average kernel time : " << rowAvgMs << " ms\n";
    std::cout << "  Sample check        : "
              << (rowPassed ? "PASSED" : "FAILED") << "\n\n";

    std::cout << "One thread per column\n";
    std::cout << "  Average kernel time : " << columnAvgMs << " ms\n";
    std::cout << "  Sample check        : "
              << (columnPassed ? "PASSED" : "FAILED") << '\n';

    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));

    return (rowPassed && columnPassed) ? EXIT_SUCCESS : EXIT_FAILURE;
}
