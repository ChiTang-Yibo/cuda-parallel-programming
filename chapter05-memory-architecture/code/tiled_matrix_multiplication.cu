#include <cuda_runtime.h>

#include <string>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
                                                                           \
        if (error != cudaSuccess) {                                        \
            std::cerr                                                      \
                << "CUDA error at "                                        \
                << __FILE__ << ":" << __LINE__                             \
                << std::endl                                               \
                << "Error code    : "                                      \
                << static_cast<int>(error)                                 \
                << std::endl                                               \
                << "Error message : "                                      \
                << cudaGetErrorString(error)                               \
                << std::endl;                                              \
                                                                           \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

constexpr int TILE_WIDTH = 16;

void matrix_mul_cpu_float(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N)
{
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {

            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k]
                     * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}

void cpu_matrix_mul_acc(const float* A,
                        const float* B,
                        float* C,
                        int M,
                        int N, 
                        int width)
{
    std::fill(C, C + M * N, 0.0f);
    for(int row = 0; row < M; ++row){
        for(int k = 0; k < width; ++k){
            for(int col = 0; col < N; ++col){
                C[row * N + col] += A[row * width + k] * B[k * N + col];
            }
        }
    }
}

__global__
void matrix_mul_naive_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row >= M || col >= N){
        return;
    }

    float sum = 0.0f;

    for(int k = 0; k < K; ++k){
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

__global__
void matrix_mul_tiled_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int L,
    int N
)
{
    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = blockIdx.y * TILE_WIDTH + ty;
    const int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;

    const int num_of_tiles = (L + TILE_WIDTH -1) / TILE_WIDTH;

    for(int tile = 0; tile < num_of_tiles; ++tile){
        const int A_col = tile * TILE_WIDTH + tx;
        const int B_row = tile * TILE_WIDTH + ty;

        if(row < M && A_col < L){
            A_shared[ty][tx] = A[row * L + A_col];
        } 
        else{
            A_shared[ty][tx] = 0.0f;
        }

        if (B_row < L && col < N) {
            B_shared[ty][tx] = B[B_row * N + col];
        }
        else {
            B_shared[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < TILE_WIDTH; ++k){
            sum += A_shared[ty][k] * B_shared[k][tx];
        }

        __syncthreads();
    }

    if(row < M && col < N){
        C[row * N + col] = sum;
    }
}

void matrix_mul_cpu(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int M,
                    int L,
                    int N)
{
    for(int row = 0; row < M; ++row){
        for(int col = 0; col < N; ++ col){
            double sum = 0.0;

            for(int k = 0; k < L; ++k){
                sum += static_cast<double>(A[row * L + k]) * static_cast<double>(B[k * N + col]);
            }
            C[row * N + col] = static_cast<float>(sum);
        }
        
    }
}

bool check_result(
    const std::vector<float>& reference,
    const std::vector<float>& result,
    float absolute_tolerance = 1.0e-3f,
    float relative_tolerance = 1.0e-3f
)
{
    if (reference.size() != result.size()){
        return false;
    }

    for(std::size_t i = 0; i < reference.size(); ++i){
        float reference_value = reference[i];

        float result_value = result[i];

        float error = std::fabs(reference_value - result_value);

        float tolerance = absolute_tolerance + relative_tolerance * std::fabs(reference_value);

        if (error > tolerance) {

            std::cerr
                << "Mismatch at linear index "
                << i
                << std::endl
                << "Reference : "
                << reference_value
                << std::endl
                << "GPU result: "
                << result_value
                << std::endl
                << "Error     : "
                << error
                << std::endl;

            return false;
        }


    }
    return true;
}

double calculate_gflops(
    int M,
    int K,
    int N,
    float time_ms
){
    if(time_ms < 0.0f){
        return 0.0;
    }

    double number_of_flops = 2 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);

    double time_seconds = static_cast<double>(time_ms) / 1000.0;

    return number_of_flops / time_seconds / 1.0e9;
}

int main(
    int argc,
    char** argv
){
    int M = 512;
    int N = 512;
    int K = 512;

    int repeat_count = 20;

    if (argc >= 2) {
        M = std::stoi(argv[1]);
    }

    if (argc >= 3) {
        K = std::stoi(argv[2]);
    }

    if (argc >= 4) {
        N = std::stoi(argv[3]);
    }

    if (argc >= 5) {
        repeat_count = std::stoi(argv[4]);
    }

    if(M <= 0
    || K <= 0
    || N <= 0
    || repeat_count <= 0){
        std::cerr
            << "M, K, N and repeat_count "
            << "must all be positive."
            << std::endl;

        return EXIT_FAILURE;
    }

    std::cout
        << "Tiled matrix multiplication"
        << std::endl
        << std::endl
        << "Matrix A      : "
        << M << " x " << K
        << std::endl
        << "Matrix B      : "
        << K << " x " << N
        << std::endl
        << "Matrix C      : "
        << M << " x " << N
        << std::endl
        << "Tile size     : "
        << TILE_WIDTH
        << " x "
        << TILE_WIDTH
        << std::endl
        << "Repeat count  : "
        << repeat_count
        << std::endl
        << std::endl;


    std::size_t A_element_count = 
    static_cast<std::size_t>(M) * static_cast<std::size_t>(K);

    std::size_t B_element_count = 
    static_cast<std::size_t>(K) * static_cast<std::size_t>(N);

    std::size_t C_element_count = 
    static_cast<std::size_t>(M) * static_cast<std::size_t>(N);

    std::size_t A_bytes = A_element_count * sizeof(float);

    std::size_t B_bytes = B_element_count * sizeof(float);

    std::size_t C_bytes = C_element_count * sizeof(float);

    std::vector<float> host_A(A_element_count);

    std::vector<float> host_B(B_element_count);

    std::vector<float> host_C_cpu(C_element_count, 0.0f);

    std::vector<float> host_C_naive(C_element_count, 0.0f);
    
    std::vector<float> host_C_tiled(C_element_count,0.0f);

    std::mt19937 random_generator(12345);

    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for(float& value : host_A){
        value = distribution(random_generator);
    }

    for(float& value : host_B){
        value = distribution(random_generator);
    }

    std::cout
        << "Computing CPU reference..."
        << std::endl;

    matrix_mul_cpu(host_A, host_B, host_C_cpu, M, K, N);

    float* device_A = nullptr;

    float* device_B = nullptr;

    float* device_C_naive = nullptr;

    float* device_C_tiled = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_A), A_bytes));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_B), B_bytes));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_C_naive), C_bytes));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_C_tiled), C_bytes));

    CUDA_CHECK(cudaMemcpy(device_A, host_A.data(), A_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(device_B, host_B.data(), B_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(device_C_naive, 0, C_bytes));
    
    CUDA_CHECK(cudaMemset(device_C_tiled, 0, C_bytes));

    dim3 block(TILE_WIDTH, TILE_WIDTH);

    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1)/ TILE_WIDTH);

    std::cout
        << "Block          : ("
        << block.x << ", "
        << block.y << ", "
        << block.z << ")"
        << std::endl
        << "Grid           : ("
        << grid.x << ", "
        << grid.y << ", "
        << grid.z << ")"
        << std::endl
        << std::endl;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    CUDA_CHECK(
        cudaEventCreate(&start_event)
    );

    CUDA_CHECK(
        cudaEventCreate(&stop_event)
    );

    matrix_mul_naive_kernel <<<grid, block>>>(device_A, device_B, device_C_naive,M, K, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start_event));

    for(int a = 0; a < repeat_count; ++ a){
        matrix_mul_naive_kernel <<<grid, block>>>(device_A, device_B, device_C_naive,M, K, N);
    }

    CUDA_CHECK(cudaEventRecord(stop_event));

    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float naive_total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&naive_total_ms, start_event, stop_event));

    float naive_average_ms = naive_total_ms / static_cast<float>(repeat_count);

    CUDA_CHECK(cudaGetLastError());

    matrix_mul_tiled_kernel<<<grid, block>>>(device_A, device_B, device_C_tiled, M, K, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaEventRecord(start_event)
    );

    for (int repeat = 0;
         repeat < repeat_count;
         ++repeat)
    {
        matrix_mul_tiled_kernel
            <<<grid, block>>>(
                device_A,
                device_B,
                device_C_tiled,
                M,
                K,
                N
            );
    }

    CUDA_CHECK(
        cudaEventRecord(stop_event)
    );

    CUDA_CHECK(
        cudaEventSynchronize(stop_event)
    );


    float tiled_total_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &tiled_total_ms,
            start_event,
            stop_event
        )
    );


    float tiled_average_ms =
        tiled_total_ms
        /
        static_cast<float>(repeat_count);


    CUDA_CHECK(cudaGetLastError());


    CUDA_CHECK(
        cudaMemcpy(
            host_C_naive.data(),
            device_C_naive,
            C_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            host_C_tiled.data(),
            device_C_tiled,
            C_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    bool naive_correct =
        check_result(
            host_C_cpu,
            host_C_naive
        );

    bool tiled_correct =
        check_result(
            host_C_cpu,
            host_C_tiled
        );

    
    double naive_gflops(calculate_gflops(M, K, N, naive_average_ms));

    double tiled_gflops(calculate_gflops(M, K, N, tiled_average_ms));

    std::cout<<std::fixed<<std::setprecision(4);

    std::cout
        << "Naive kernel"
        << std::endl
        << "  Average time : "
        << naive_average_ms
        << " ms"
        << std::endl
        << "  Throughput   : "
        << naive_gflops
        << " GFLOP/s"
        << std::endl
        << "  Correctness  : "
        << (
            naive_correct
            ? "PASSED"
            : "FAILED"
        )
        << std::endl
        << std::endl;


    std::cout
        << "Tiled kernel"
        << std::endl
        << "  Average time : "
        << tiled_average_ms
        << " ms"
        << std::endl
        << "  Throughput   : "
        << tiled_gflops
        << " GFLOP/s"
        << std::endl
        << "  Correctness  : "
        << (
            tiled_correct
            ? "PASSED"
            : "FAILED"
        )
        << std::endl
        << std::endl;


    

    if (tiled_average_ms > 0.0f) {

        double speedup =
            static_cast<double>(
                naive_average_ms
            )
            /
            static_cast<double>(
                tiled_average_ms
            );

        std::cout
            << "Speedup"
            << std::endl
            << "  Naive / tiled: "
            << speedup
            << "x"
            << std::endl;
    }


    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    CUDA_CHECK(cudaFree(device_A));
    CUDA_CHECK(cudaFree(device_B));

    CUDA_CHECK(cudaFree(device_C_naive));

    CUDA_CHECK(cudaFree(device_C_tiled));

    return(naive_correct && tiled_correct)

    ? EXIT_SUCCESS
    : EXIT_FAILURE;

}