#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

constexpr int TILE_WIDTH = 16;
constexpr int COARSE_FACTOR = 4;
constexpr int GPU_WARMUP = 2;
constexpr int GPU_REPEAT = 20;
constexpr int CPU_REPEAT = 1;

void check_cuda(cudaError_t err, const char* where)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << where << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void cpu_matmul(const std::vector<float>& M,
                const std::vector<float>& N,
                std::vector<float>& P,
                int H, int K, int W)
{
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += M[i * K + k] * N[k * W + j];
            }
            P[i * W + j] = sum;
        }
    }
}

bool verify_result(const std::vector<float>& gpu_result,
                   const std::vector<float>& cpu_result,
                   float& max_abs_error)
{
    const float atol = 1.0e-4f;
    const float rtol = 1.0e-4f;

    max_abs_error = 0.0f;
    int mismatch_count = 0;

    for (size_t i = 0; i < gpu_result.size(); ++i) {
        float diff = std::abs(gpu_result[i] - cpu_result[i]);
        float tolerance = atol + rtol * std::abs(cpu_result[i]);

        if (diff > max_abs_error) {
            max_abs_error = diff;
        }

        if (diff > tolerance) {
            if (mismatch_count < 5) {
                std::cerr << "Mismatch at index " << i
                          << ": GPU = " << gpu_result[i]
                          << ", CPU = " << cpu_result[i]
                          << ", diff = " << diff << std::endl;
            }
            ++mismatch_count;
        }
    }

    return mismatch_count == 0;
}

__global__ void coarse_matmul(const float* M,
                              const float* N,
                              float* P,
                              int H, int K, int W)
{
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    int colStart =
        blockIdx.x * COARSE_FACTOR * TILE_WIDTH + threadIdx.x;

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    float Pvalue[COARSE_FACTOR] = {0.0f};

    int numPhases = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int p = 0; p < numPhases; ++p) {
        if (row < H && p * TILE_WIDTH + threadIdx.x < K) {
            Mds[threadIdx.y][threadIdx.x] =
                M[row * K + p * TILE_WIDTH + threadIdx.x];
        } else {
            Mds[threadIdx.y][threadIdx.x] = 0.0f;
        }

        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int nRow = p * TILE_WIDTH + threadIdx.y;
            int nCol = colStart + c * TILE_WIDTH;

            if (nRow < K && nCol < W) {
                Nds[threadIdx.y][threadIdx.x] =
                    N[nRow * W + nCol];
            } else {
                Nds[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            #pragma unroll
            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] +=
                    Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
            }

            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c * TILE_WIDTH;
        if (row < H && col < W) {
            P[row * W + col] = Pvalue[c];
        }
    }
}

void benchmark_size(int N)
{
    const int H = N;
    const int K = N;
    const int W = N;

    const size_t bytes_M = static_cast<size_t>(H) * K * sizeof(float);
    const size_t bytes_N = static_cast<size_t>(K) * W * sizeof(float);
    const size_t bytes_P = static_cast<size_t>(H) * W * sizeof(float);

    std::vector<float> h_M(static_cast<size_t>(H) * K);
    std::vector<float> h_N(static_cast<size_t>(K) * W);
    std::vector<float> h_P(static_cast<size_t>(H) * W);
    std::vector<float> h_P_ref(static_cast<size_t>(H) * W);

    std::mt19937 generator(12345 + N);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (float& value : h_M) value = distribution(generator);
    for (float& value : h_N) value = distribution(generator);

    using clock_type = std::chrono::high_resolution_clock;
    double cpu_total_ms = 0.0;

    for (int r = 0; r < CPU_REPEAT; ++r) {
        auto start = clock_type::now();
        cpu_matmul(h_M, h_N, h_P_ref, H, K, W);
        auto stop = clock_type::now();

        std::chrono::duration<double, std::milli> elapsed = stop - start;
        cpu_total_ms += elapsed.count();
    }

    const double cpu_ms = cpu_total_ms / CPU_REPEAT;

    float* d_M = nullptr;
    float* d_N = nullptr;
    float* d_P = nullptr;

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_M), bytes_M), "cudaMalloc(d_M)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_N), bytes_N), "cudaMalloc(d_N)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_P), bytes_P), "cudaMalloc(d_P)");

    check_cuda(cudaMemcpy(d_M, h_M.data(), bytes_M, cudaMemcpyHostToDevice), "H2D M");
    check_cuda(cudaMemcpy(d_N, h_N.data(), bytes_N, cudaMemcpyHostToDevice), "H2D N");

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(
        (W + COARSE_FACTOR * TILE_WIDTH - 1) /
            (COARSE_FACTOR * TILE_WIDTH),
        (H + TILE_WIDTH - 1) / TILE_WIDTH);

    for (int r = 0; r < GPU_WARMUP; ++r) {
        coarse_matmul<<<gridDim, blockDim>>>(d_M, d_N, d_P, H, K, W);
    }
    check_cuda(cudaGetLastError(), "warm-up launch");
    check_cuda(cudaDeviceSynchronize(), "warm-up synchronize");

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start_event), "cudaEventRecord(start)");

    for (int r = 0; r < GPU_REPEAT; ++r) {
        coarse_matmul<<<gridDim, blockDim>>>(d_M, d_N, d_P, H, K, W);
    }

    check_cuda(cudaGetLastError(), "timed launch");
    check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");

    float gpu_total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&gpu_total_ms, start_event, stop_event),
               "cudaEventElapsedTime");

    const double gpu_ms = static_cast<double>(gpu_total_ms) / GPU_REPEAT;

    check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");

    check_cuda(cudaMemcpy(h_P.data(), d_P, bytes_P, cudaMemcpyDeviceToHost), "D2H P");

    float max_abs_error = 0.0f;
    bool passed = verify_result(h_P, h_P_ref, max_abs_error);

    // Conventional useful GEMM FLOPs: ~2*H*K*W (FMA counted as 2 FLOPs).
    const double useful_flops =
        2.0 * static_cast<double>(H) * K * W;

    // Estimated arithmetic executed by THIS kernel, including padded work
    // from extra tiles/phases/threads at matrix boundaries.
    const int numPhases = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    const double launched_threads =
        static_cast<double>(gridDim.x) *
        static_cast<double>(gridDim.y) *
        TILE_WIDTH * TILE_WIDTH;

    const double estimated_kernel_flops =
        2.0 * launched_threads * COARSE_FACTOR *
        numPhases * TILE_WIDTH;

    const double padding_efficiency =
        useful_flops / estimated_kernel_flops * 100.0;

    // FLOPs / ms / 1e6 = GFLOP/s
    const double cpu_gflops = useful_flops / cpu_ms / 1.0e6;
    const double gpu_gflops = useful_flops / gpu_ms / 1.0e6;
    const double speedup = cpu_ms / gpu_ms;

    std::cout
        << std::setw(6) << N
        << std::setw(8) << ((N % 64 == 0) ? "yes" : "no")
        << std::setw(12) << std::fixed << std::setprecision(3)
        << useful_flops / 1.0e9
        << std::setw(12) << estimated_kernel_flops / 1.0e9
        << std::setw(10) << std::setprecision(1) << padding_efficiency
        << std::setw(12) << std::setprecision(3) << cpu_ms
        << std::setw(12) << std::setprecision(2) << cpu_gflops
        << std::setw(12) << std::setprecision(4) << gpu_ms
        << std::setw(12) << std::setprecision(2) << gpu_gflops
        << std::setw(10) << std::setprecision(2) << speedup
        << std::setw(14) << std::scientific << std::setprecision(2)
        << max_abs_error
        << std::setw(10) << (passed ? "PASS" : "FAIL")
        << std::defaultfloat
        << '\n';

    check_cuda(cudaFree(d_M), "cudaFree(d_M)");
    check_cuda(cudaFree(d_N), "cudaFree(d_N)");
    check_cuda(cudaFree(d_P), "cudaFree(d_P)");
}

int main(int argc, char** argv)
{
    std::vector<int> sizes;

    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            sizes.push_back(std::stoi(argv[i]));
        }
    } else {
        // Pairs around multiples of 64.
        // 16 * COARSE_FACTOR = 64 columns per coarse block.
        sizes = {
            255, 256, 257,
            319, 320, 321,
            511, 512, 513
        };
    }

    std::cout
        << "TILE_WIDTH    = " << TILE_WIDTH << '\n'
        << "COARSE_FACTOR = " << COARSE_FACTOR << '\n'
        << "GPU warm-up   = " << GPU_WARMUP << '\n'
        << "GPU repeats   = " << GPU_REPEAT << '\n'
        << "CPU repeats   = " << CPU_REPEAT << "\n\n";

    std::cout
        << std::setw(6)  << "N"
        << std::setw(8)  << "64x?"
        << std::setw(12) << "UsefulGF"
        << std::setw(12) << "KernelGF"
        << std::setw(10) << "Eff(%)"
        << std::setw(12) << "CPU_ms"
        << std::setw(12) << "CPU_GF/s"
        << std::setw(12) << "GPU_ms"
        << std::setw(12) << "GPU_GF/s"
        << std::setw(10) << "Speedup"
        << std::setw(14) << "MaxErr"
        << std::setw(10) << "Check"
        << '\n';

    std::cout << std::string(128, '-') << '\n';

    for (int N : sizes) {
        benchmark_size(N);
    }

    return 0;
}
