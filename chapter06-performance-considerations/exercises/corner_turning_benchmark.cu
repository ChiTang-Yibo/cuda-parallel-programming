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

constexpr int GPU_WARMUP = 3;
constexpr int GPU_REPEAT = 50;
constexpr int CPU_REPEAT = 1;


// ============================================================
// CUDA error checking
// ============================================================

void check_cuda(cudaError_t err, const char* where)
{
    if (err != cudaSuccess) {
        std::cerr
            << "CUDA error at " << where << ": "
            << cudaGetErrorString(err)
            << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


// ============================================================
// CPU reference 1
// A: row-major
// B: row-major
//
// C[i,j] = sum_k A[i,k] * B[k,j]
//
// Inner-loop access to B:
//     B[k * W + j]
// which is strided by W elements as k changes.
// ============================================================

void cpu_matmul_row_major_B(
    const std::vector<float>& A,
    const std::vector<float>& B_row,
    std::vector<float>& C,
    int H,
    int K,
    int W)
{
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {

            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum +=
                    A[i * K + k]
                    * B_row[k * W + j];
            }

            C[i * W + j] = sum;
        }
    }
}


// ============================================================
// CPU reference 2
// A: row-major
// B: column-major
//
// Logical B[k,j] is stored as:
//     B_col[j * K + k]
//
// Inner-loop access to B is contiguous as k changes.
// ============================================================

void cpu_matmul_column_major_B(
    const std::vector<float>& A,
    const std::vector<float>& B_col,
    std::vector<float>& C,
    int H,
    int K,
    int W)
{
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {

            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum +=
                    A[i * K + k]
                    * B_col[j * K + k];
            }

            C[i * W + j] = sum;
        }
    }
}


// ============================================================
// Verification
// ============================================================

bool verify_result(
    const std::vector<float>& result,
    const std::vector<float>& reference,
    float& max_abs_error)
{
    const float atol = 1.0e-4f;
    const float rtol = 1.0e-4f;

    max_abs_error = 0.0f;
    int mismatch_count = 0;

    for (size_t i = 0; i < result.size(); ++i) {

        const float diff =
            std::abs(result[i] - reference[i]);

        const float tolerance =
            atol + rtol * std::abs(reference[i]);

        if (diff > max_abs_error) {
            max_abs_error = diff;
        }

        if (diff > tolerance) {

            if (mismatch_count < 5) {
                std::cerr
                    << "Mismatch at index "
                    << i
                    << ": result = "
                    << result[i]
                    << ", reference = "
                    << reference[i]
                    << ", diff = "
                    << diff
                    << std::endl;
            }

            ++mismatch_count;
        }
    }

    return mismatch_count == 0;
}


// ============================================================
// GPU Kernel 1
// A: row-major
// B: row-major
//
// Standard tiled GEMM.
// Both A and B global-memory loads are coalesced.
// ============================================================

__global__ void tiled_row_major_B(
    const float* A,
    const float* B,
    float* C,
    int H,
    int K,
    int W)
{
    const int row =
        blockIdx.y * TILE_WIDTH
        + threadIdx.y;

    const int col =
        blockIdx.x * TILE_WIDTH
        + threadIdx.x;

    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

    float value = 0.0f;

    const int numPhases =
        (K + TILE_WIDTH - 1)
        / TILE_WIDTH;

    for (int p = 0; p < numPhases; ++p) {

        const int aCol =
            p * TILE_WIDTH
            + threadIdx.x;

        const int bRow =
            p * TILE_WIDTH
            + threadIdx.y;

        if (row < H && aCol < K) {
            A_shared[threadIdx.y][threadIdx.x] =
                A[row * K + aCol];
        }
        else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (bRow < K && col < W) {
            B_shared[threadIdx.y][threadIdx.x] =
                B[bRow * W + col];
        }
        else {
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            value +=
                A_shared[threadIdx.y][k]
                * B_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < H && col < W) {
        C[row * W + col] = value;
    }
}


// ============================================================
// GPU Kernel 2
// A: row-major
// B: column-major
//
// Correct mathematically, but NO corner turning.
//
// Logical B[bRow, col] is:
//     B[col * K + bRow]
//
// For a warp, threadIdx.x changes col.
// Therefore neighboring threads access addresses separated
// by K floats -> poor global-memory coalescing.
// ============================================================

__global__ void tiled_column_major_B_uncornered(
    const float* A,
    const float* B,
    float* C,
    int H,
    int K,
    int W)
{
    const int row =
        blockIdx.y * TILE_WIDTH
        + threadIdx.y;

    const int col =
        blockIdx.x * TILE_WIDTH
        + threadIdx.x;

    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

    float value = 0.0f;

    const int numPhases =
        (K + TILE_WIDTH - 1)
        / TILE_WIDTH;

    for (int p = 0; p < numPhases; ++p) {

        const int aCol =
            p * TILE_WIDTH
            + threadIdx.x;

        const int bRow =
            p * TILE_WIDTH
            + threadIdx.y;

        if (row < H && aCol < K) {
            A_shared[threadIdx.y][threadIdx.x] =
                A[row * K + aCol];
        }
        else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (bRow < K && col < W) {
            B_shared[threadIdx.y][threadIdx.x] =
                B[col * K + bRow];
        }
        else {
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            value +=
                A_shared[threadIdx.y][k]
                * B_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < H && col < W) {
        C[row * W + col] = value;
    }
}


// ============================================================
// GPU Kernel 3
// A: row-major
// B: column-major
//
// Corner turning:
//   global load follows the contiguous row direction of
//   column-major B.
//
// Load:
//   row_B = p*T + tx
//   col_B = bx*T + ty
//
// Column-major index:
//   col_B * K + row_B
//
// The value is then written to:
//   B_shared[tx][ty]
//
// so computation can still use:
//   B_shared[k][tx]
// ============================================================

__global__ void tiled_column_major_B_corner(
    const float* A,
    const float* B,
    float* C,
    int H,
    int K,
    int W)
{
    const int row =
        blockIdx.y * TILE_WIDTH
        + threadIdx.y;

    const int col =
        blockIdx.x * TILE_WIDTH
        + threadIdx.x;

    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

    float value = 0.0f;

    const int numPhases =
        (K + TILE_WIDTH - 1)
        / TILE_WIDTH;

    for (int p = 0; p < numPhases; ++p) {

        const int aCol =
            p * TILE_WIDTH
            + threadIdx.x;

        if (row < H && aCol < K) {
            A_shared[threadIdx.y][threadIdx.x] =
                A[row * K + aCol];
        }
        else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        const int bRowLoad =
            p * TILE_WIDTH
            + threadIdx.x;

        const int bColLoad =
            blockIdx.x * TILE_WIDTH
            + threadIdx.y;

        if (bRowLoad < K && bColLoad < W) {
            B_shared[threadIdx.x][threadIdx.y] =
                B[bColLoad * K + bRowLoad];
        }
        else {
            B_shared[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            value +=
                A_shared[threadIdx.y][k]
                * B_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < H && col < W) {
        C[row * W + col] = value;
    }
}


// ============================================================
// GPU Kernel 4
// Same corner turning as Kernel 3, but pad the second shared
// dimension by one element.
//
// This keeps the global-memory load coalesced while changing
// shared-memory bank mapping for the transposed write/read.
// ============================================================

__global__ void tiled_column_major_B_corner_padded(
    const float* A,
    const float* B,
    float* C,
    int H,
    int K,
    int W)
{
    const int row =
        blockIdx.y * TILE_WIDTH
        + threadIdx.y;

    const int col =
        blockIdx.x * TILE_WIDTH
        + threadIdx.x;

    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH + 1];

    float value = 0.0f;

    const int numPhases =
        (K + TILE_WIDTH - 1)
        / TILE_WIDTH;

    for (int p = 0; p < numPhases; ++p) {

        const int aCol =
            p * TILE_WIDTH
            + threadIdx.x;

        if (row < H && aCol < K) {
            A_shared[threadIdx.y][threadIdx.x] =
                A[row * K + aCol];
        }
        else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        const int bRowLoad =
            p * TILE_WIDTH
            + threadIdx.x;

        const int bColLoad =
            blockIdx.x * TILE_WIDTH
            + threadIdx.y;

        if (bRowLoad < K && bColLoad < W) {
            B_shared[threadIdx.x][threadIdx.y] =
                B[bColLoad * K + bRowLoad];
        }
        else {
            B_shared[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            value +=
                A_shared[threadIdx.y][k]
                * B_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < H && col < W) {
        C[row * W + col] = value;
    }
}


// ============================================================
// CPU timing helpers
// ============================================================

double benchmark_cpu_row_major_B(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int H,
    int K,
    int W)
{
    using clock_type =
        std::chrono::high_resolution_clock;

    double total_ms = 0.0;

    for (int r = 0; r < CPU_REPEAT; ++r) {

        const auto start =
            clock_type::now();

        cpu_matmul_row_major_B(
            A, B, C, H, K, W);

        const auto stop =
            clock_type::now();

        const std::chrono::duration<double, std::milli>
            elapsed = stop - start;

        total_ms += elapsed.count();
    }

    return total_ms / CPU_REPEAT;
}


double benchmark_cpu_column_major_B(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int H,
    int K,
    int W)
{
    using clock_type =
        std::chrono::high_resolution_clock;

    double total_ms = 0.0;

    for (int r = 0; r < CPU_REPEAT; ++r) {

        const auto start =
            clock_type::now();

        cpu_matmul_column_major_B(
            A, B, C, H, K, W);

        const auto stop =
            clock_type::now();

        const std::chrono::duration<double, std::milli>
            elapsed = stop - start;

        total_ms += elapsed.count();
    }

    return total_ms / CPU_REPEAT;
}


// ============================================================
// GPU benchmark helper
// ============================================================

enum class KernelKind
{
    RowMajorTiled,
    ColumnMajorUncornered,
    ColumnMajorCorner,
    ColumnMajorCornerPadded
};


void launch_kernel(
    KernelKind kind,
    dim3 gridDim,
    dim3 blockDim,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int H,
    int K,
    int W)
{
    switch (kind) {

        case KernelKind::RowMajorTiled:
            tiled_row_major_B<<<gridDim, blockDim>>>(
                d_A, d_B, d_C, H, K, W);
            break;

        case KernelKind::ColumnMajorUncornered:
            tiled_column_major_B_uncornered<<<gridDim, blockDim>>>(
                d_A, d_B, d_C, H, K, W);
            break;

        case KernelKind::ColumnMajorCorner:
            tiled_column_major_B_corner<<<gridDim, blockDim>>>(
                d_A, d_B, d_C, H, K, W);
            break;

        case KernelKind::ColumnMajorCornerPadded:
            tiled_column_major_B_corner_padded<<<gridDim, blockDim>>>(
                d_A, d_B, d_C, H, K, W);
            break;
    }
}


double benchmark_gpu(
    KernelKind kind,
    dim3 gridDim,
    dim3 blockDim,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int H,
    int K,
    int W)
{
    for (int r = 0; r < GPU_WARMUP; ++r) {
        launch_kernel(
            kind,
            gridDim,
            blockDim,
            d_A,
            d_B,
            d_C,
            H,
            K,
            W);
    }

    check_cuda(
        cudaGetLastError(),
        "GPU warm-up launch");

    check_cuda(
        cudaDeviceSynchronize(),
        "GPU warm-up synchronize");

    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    check_cuda(
        cudaEventCreate(&start_event),
        "cudaEventCreate(start)");

    check_cuda(
        cudaEventCreate(&stop_event),
        "cudaEventCreate(stop)");

    check_cuda(
        cudaEventRecord(start_event),
        "cudaEventRecord(start)");

    for (int r = 0; r < GPU_REPEAT; ++r) {
        launch_kernel(
            kind,
            gridDim,
            blockDim,
            d_A,
            d_B,
            d_C,
            H,
            K,
            W);
    }

    check_cuda(
        cudaGetLastError(),
        "timed kernel launch");

    check_cuda(
        cudaEventRecord(stop_event),
        "cudaEventRecord(stop)");

    check_cuda(
        cudaEventSynchronize(stop_event),
        "cudaEventSynchronize(stop)");

    float total_ms = 0.0f;

    check_cuda(
        cudaEventElapsedTime(
            &total_ms,
            start_event,
            stop_event),
        "cudaEventElapsedTime");

    check_cuda(
        cudaEventDestroy(start_event),
        "cudaEventDestroy(start)");

    check_cuda(
        cudaEventDestroy(stop_event),
        "cudaEventDestroy(stop)");

    return
        static_cast<double>(total_ms)
        / GPU_REPEAT;
}


// ============================================================
// Run all layout/coalescing experiments for one N x N matrix
// ============================================================

void run_experiment(int N)
{
    const int H = N;
    const int K = N;
    const int W = N;

    const size_t bytes_A =
        static_cast<size_t>(H)
        * K
        * sizeof(float);

    const size_t bytes_B =
        static_cast<size_t>(K)
        * W
        * sizeof(float);

    const size_t bytes_C =
        static_cast<size_t>(H)
        * W
        * sizeof(float);

    std::vector<float> h_A(
        static_cast<size_t>(H) * K);

    // Same logical B, two different physical layouts.
    std::vector<float> h_B_row(
        static_cast<size_t>(K) * W);

    std::vector<float> h_B_col(
        static_cast<size_t>(K) * W);

    std::vector<float> h_C_cpu_row(
        static_cast<size_t>(H) * W);

    std::vector<float> h_C_cpu_col(
        static_cast<size_t>(H) * W);

    std::vector<float> h_C_gpu(
        static_cast<size_t>(H) * W);

    std::mt19937 generator(
        12345 + N);

    std::uniform_real_distribution<float>
        distribution(-1.0f, 1.0f);

    for (float& value : h_A) {
        value = distribution(generator);
    }

    // Generate one logical B and store the same values in:
    //   row-major:    B_row[k * W + j]
    //   column-major: B_col[j * K + k]
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < W; ++j) {

            const float value =
                distribution(generator);

            h_B_row[k * W + j] =
                value;

            h_B_col[j * K + k] =
                value;
        }
    }

    // --------------------------------------------------------
    // CPU experiments
    // --------------------------------------------------------
    const double cpu_row_ms =
        benchmark_cpu_row_major_B(
            h_A,
            h_B_row,
            h_C_cpu_row,
            H,
            K,
            W);

    const double cpu_col_ms =
        benchmark_cpu_column_major_B(
            h_A,
            h_B_col,
            h_C_cpu_col,
            H,
            K,
            W);

    float cpu_layout_error = 0.0f;

    const bool cpu_layout_match =
        verify_result(
            h_C_cpu_col,
            h_C_cpu_row,
            cpu_layout_error);

    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------
    float* d_A = nullptr;
    float* d_B_row = nullptr;
    float* d_B_col = nullptr;
    float* d_C = nullptr;

    check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&d_A),
            bytes_A),
        "cudaMalloc(d_A)");

    check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&d_B_row),
            bytes_B),
        "cudaMalloc(d_B_row)");

    check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&d_B_col),
            bytes_B),
        "cudaMalloc(d_B_col)");

    check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&d_C),
            bytes_C),
        "cudaMalloc(d_C)");

    check_cuda(
        cudaMemcpy(
            d_A,
            h_A.data(),
            bytes_A,
            cudaMemcpyHostToDevice),
        "H2D A");

    check_cuda(
        cudaMemcpy(
            d_B_row,
            h_B_row.data(),
            bytes_B,
            cudaMemcpyHostToDevice),
        "H2D B row-major");

    check_cuda(
        cudaMemcpy(
            d_B_col,
            h_B_col.data(),
            bytes_B,
            cudaMemcpyHostToDevice),
        "H2D B column-major");

    const dim3 blockDim(
        TILE_WIDTH,
        TILE_WIDTH);

    const dim3 gridDim(
        (W + TILE_WIDTH - 1)
            / TILE_WIDTH,
        (H + TILE_WIDTH - 1)
            / TILE_WIDTH);

    const double useful_flops =
        2.0
        * static_cast<double>(H)
        * static_cast<double>(K)
        * static_cast<double>(W);

    // --------------------------------------------------------
    // GPU experiment 1: row-major B
    // --------------------------------------------------------
    const double gpu_row_ms =
        benchmark_gpu(
            KernelKind::RowMajorTiled,
            gridDim,
            blockDim,
            d_A,
            d_B_row,
            d_C,
            H,
            K,
            W);

    check_cuda(
        cudaMemcpy(
            h_C_gpu.data(),
            d_C,
            bytes_C,
            cudaMemcpyDeviceToHost),
        "D2H row-major result");

    float err_gpu_row = 0.0f;

    const bool pass_gpu_row =
        verify_result(
            h_C_gpu,
            h_C_cpu_row,
            err_gpu_row);

    // --------------------------------------------------------
    // GPU experiment 2: column-major B, no corner turning
    // --------------------------------------------------------
    const double gpu_col_uncornered_ms =
        benchmark_gpu(
            KernelKind::ColumnMajorUncornered,
            gridDim,
            blockDim,
            d_A,
            d_B_col,
            d_C,
            H,
            K,
            W);

    check_cuda(
        cudaMemcpy(
            h_C_gpu.data(),
            d_C,
            bytes_C,
            cudaMemcpyDeviceToHost),
        "D2H column-major uncornered result");

    float err_gpu_uncornered = 0.0f;

    const bool pass_gpu_uncornered =
        verify_result(
            h_C_gpu,
            h_C_cpu_row,
            err_gpu_uncornered);

    // --------------------------------------------------------
    // GPU experiment 3: column-major B + corner turning
    // --------------------------------------------------------
    const double gpu_corner_ms =
        benchmark_gpu(
            KernelKind::ColumnMajorCorner,
            gridDim,
            blockDim,
            d_A,
            d_B_col,
            d_C,
            H,
            K,
            W);

    check_cuda(
        cudaMemcpy(
            h_C_gpu.data(),
            d_C,
            bytes_C,
            cudaMemcpyDeviceToHost),
        "D2H corner-turning result");

    float err_gpu_corner = 0.0f;

    const bool pass_gpu_corner =
        verify_result(
            h_C_gpu,
            h_C_cpu_row,
            err_gpu_corner);

    // --------------------------------------------------------
    // GPU experiment 4: column-major B + corner turning + padding
    // --------------------------------------------------------
    const double gpu_corner_padded_ms =
        benchmark_gpu(
            KernelKind::ColumnMajorCornerPadded,
            gridDim,
            blockDim,
            d_A,
            d_B_col,
            d_C,
            H,
            K,
            W);

    check_cuda(
        cudaMemcpy(
            h_C_gpu.data(),
            d_C,
            bytes_C,
            cudaMemcpyDeviceToHost),
        "D2H padded corner-turning result");

    float err_gpu_corner_padded = 0.0f;

    const bool pass_gpu_corner_padded =
        verify_result(
            h_C_gpu,
            h_C_cpu_row,
            err_gpu_corner_padded);

    // --------------------------------------------------------
    // Derived performance values
    // --------------------------------------------------------
    const double cpu_row_gflops =
        useful_flops
        / cpu_row_ms
        / 1.0e6;

    const double cpu_col_gflops =
        useful_flops
        / cpu_col_ms
        / 1.0e6;

    const double gpu_row_gflops =
        useful_flops
        / gpu_row_ms
        / 1.0e6;

    const double gpu_col_uncornered_gflops =
        useful_flops
        / gpu_col_uncornered_ms
        / 1.0e6;

    const double gpu_corner_gflops =
        useful_flops
        / gpu_corner_ms
        / 1.0e6;

    const double gpu_corner_padded_gflops =
        useful_flops
        / gpu_corner_padded_ms
        / 1.0e6;

    const double cpu_col_speedup =
        cpu_row_ms
        / cpu_col_ms;

    const double corner_speedup =
        gpu_col_uncornered_ms
        / gpu_corner_ms;

    const double padded_speedup =
        gpu_col_uncornered_ms
        / gpu_corner_padded_ms;

    const bool all_pass =
        cpu_layout_match
        && pass_gpu_row
        && pass_gpu_uncornered
        && pass_gpu_corner
        && pass_gpu_corner_padded;

    // --------------------------------------------------------
    // Report
    // --------------------------------------------------------
    std::cout
        << "\n============================================================\n"
        << "N = " << N << '\n'
        << "TILE_WIDTH = " << TILE_WIDTH << '\n'
        << "============================================================\n";

    std::cout
        << std::fixed
        << std::setprecision(4);

    std::cout
        << "\nCPU layout experiment\n"
        << "  Row-major B    : "
        << cpu_row_ms << " ms, "
        << std::setprecision(2)
        << cpu_row_gflops << " GFLOP/s\n"
        << std::setprecision(4)
        << "  Column-major B : "
        << cpu_col_ms << " ms, "
        << std::setprecision(2)
        << cpu_col_gflops << " GFLOP/s\n"
        << std::setprecision(3)
        << "  Column/Row speedup : "
        << cpu_col_speedup << "x\n";

    std::cout
        << "\nGPU layout/coalescing experiment\n"
        << std::setprecision(4)
        << "  1. Row-major B, standard tiled      : "
        << gpu_row_ms << " ms, "
        << std::setprecision(2)
        << gpu_row_gflops << " GFLOP/s\n"
        << std::setprecision(4)
        << "  2. Column-major B, no corner turn   : "
        << gpu_col_uncornered_ms << " ms, "
        << std::setprecision(2)
        << gpu_col_uncornered_gflops << " GFLOP/s\n"
        << std::setprecision(4)
        << "  3. Column-major B, corner turning   : "
        << gpu_corner_ms << " ms, "
        << std::setprecision(2)
        << gpu_corner_gflops << " GFLOP/s\n"
        << std::setprecision(4)
        << "  4. Corner turning + shared padding  : "
        << gpu_corner_padded_ms << " ms, "
        << std::setprecision(2)
        << gpu_corner_padded_gflops << " GFLOP/s\n";

    std::cout
        << "\nGPU speedups relative to uncornered column-major B\n"
        << std::setprecision(3)
        << "  Corner turning          : "
        << corner_speedup << "x\n"
        << "  Corner turning + padding: "
        << padded_speedup << "x\n";

    std::cout
        << "\nCorrectness\n"
        << "  CPU row vs. column layout : "
        << (cpu_layout_match ? "PASS" : "FAIL")
        << ", max error = "
        << std::scientific
        << cpu_layout_error
        << '\n'
        << "  GPU row-major tiled       : "
        << (pass_gpu_row ? "PASS" : "FAIL")
        << ", max error = "
        << err_gpu_row
        << '\n'
        << "  GPU col-major uncornered  : "
        << (pass_gpu_uncornered ? "PASS" : "FAIL")
        << ", max error = "
        << err_gpu_uncornered
        << '\n'
        << "  GPU corner turning        : "
        << (pass_gpu_corner ? "PASS" : "FAIL")
        << ", max error = "
        << err_gpu_corner
        << '\n'
        << "  GPU corner + padding      : "
        << (pass_gpu_corner_padded ? "PASS" : "FAIL")
        << ", max error = "
        << err_gpu_corner_padded
        << '\n'
        << std::defaultfloat
        << "\nOverall: "
        << (all_pass ? "PASS" : "FAIL")
        << '\n';

    check_cuda(
        cudaFree(d_A),
        "cudaFree(d_A)");

    check_cuda(
        cudaFree(d_B_row),
        "cudaFree(d_B_row)");

    check_cuda(
        cudaFree(d_B_col),
        "cudaFree(d_B_col)");

    check_cuda(
        cudaFree(d_C),
        "cudaFree(d_C)");
}


// ============================================================
// Main
//
// Default experiments cover several matrix sizes.
// You can also pass custom sizes:
//
//   corner_turning_benchmark.exe 256 512 1024
// ============================================================

int main(int argc, char** argv)
{
    std::vector<int> sizes;

    if (argc > 1) {

        for (int i = 1; i < argc; ++i) {
            sizes.push_back(
                std::stoi(argv[i]));
        }

    } else {

        sizes = {
            128,
            256,
            512,
            768,
            1024
        };
    }

    std::cout
        << "Column-major / corner-turning benchmark\n"
        << "GPU warm-up = "
        << GPU_WARMUP
        << ", GPU repeats = "
        << GPU_REPEAT
        << ", CPU repeats = "
        << CPU_REPEAT
        << '\n';

    std::cout
        << "\nExperiment meanings:\n"
        << "  CPU row-major B:"
        << " inner-k access to B is strided.\n"
        << "  CPU column-major B:"
        << " inner-k access to B is contiguous.\n"
        << "  GPU row-major B:"
        << " standard tiled B load is coalesced.\n"
        << "  GPU column-major B without corner turning:"
        << " warp B load is strided by K floats.\n"
        << "  GPU corner turning:"
        << " global B load becomes coalesced.\n"
        << "  GPU corner + padding:"
        << " also changes shared-memory bank mapping.\n";

    for (int N : sizes) {
        run_experiment(N);
    }

    return 0;
}
