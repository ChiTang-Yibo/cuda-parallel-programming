#include <cuda_runtime.h>

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <vector>
#include <utility>


#define cuda_check(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__       \
                      << " : " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                      \
    } while (0)


class CpuTimer {
public:
    CpuTimer() {
        reset();
    }

    void reset() {
        start_time_ = clock_type::now();
    }

    double elapsed_ms() const {
        auto end_time = clock_type::now();
        std::chrono::duration<double, std::milli> elapsed =
            end_time - start_time_;
        return elapsed.count();
    }

private:
    using clock_type = std::chrono::high_resolution_clock;
    std::chrono::time_point<clock_type> start_time_;
};

__global__
void matrixMulKernel(const float*M,
                     const float*N,
                     float* P,
                    int width)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < width && col < width){
        float pValue = 0.0f;

        for(int k = 0; k < width; ++k){
            pValue += M[ row * width + k] * N[k * width + col];
        }

        P[row * width + col] = pValue;
    }
}

void initialize_matirx(std::vector<float>& mat, int width, int seed_offset){
    for(int i = 0; i < width * width; ++i){
        int value = (i + seed_offset) % 100;
        mat[i] = static_cast<float>(value) / 100.0f;
    }
}

float cpu_reference_element(const std::vector<float>& M,
                            const std::vector<float>& N,
                            int width,
                            int row,
                            int col)
{
    float value = 0.0f;
    for(int k = 0; k < width; ++k){
        value += M[row * width + k] * N[k * width + col];
    }

    return value;
}

bool sample_check_result(const std::vector<float>& M,
                         const std::vector<float>& N,
                         const std::vector<float>& P,
                         int width)
{
    std::vector<std::pair<int, int>> samples ={
        {0,0},
        {0, width - 1},
        {width - 1, 0},
        {width-1, width - 1},
        {width / 2, width / 2}
    };

    const float tolerance = 1.0e-3f;

    for(const auto& item : samples){
        int row = item.first;
        int col = item.second;

        float cpu_value = cpu_reference_element(M, N, width, row, col);
        float gpu_value = P[row * width + col];

        float diff = std::fabs(cpu_value - gpu_value);

        if(diff > tolerance){
            std::cout << "Check failed at P[" << row << "][" << col << "]"
                      << " CPU = " << cpu_value
                      << " GPU = " << gpu_value
                      << " diff = " << diff << std::endl;
            return false;
        }

    }
    return true;
}

void run_matmul_case(int width, int repeat){
    std::cout << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Matrix size: " << width << " x " << width << std::endl;
    std::cout << "Repeat: " << repeat << std::endl;
    std::cout << "============================================================" << std::endl;

    CpuTimer totalTimer;

    size_t numElements = static_cast<size_t>(width) * width;
    size_t numBytes = static_cast<size_t>(numElements) * sizeof(float);

    CpuTimer initTimer;

    std::vector<float> h_M(numElements);
    std::vector<float> h_N(numElements);
    std::vector<float> h_P(numElements, 0.0f);

    initialize_matirx(h_M, width, 0);
    initialize_matirx(h_N, width, 17);

    double hostInitMs = initTimer.elapsed_ms();

    float* d_M = nullptr;
    float* d_N = nullptr;
    float* d_P = nullptr;

    CpuTimer mallocTimer;

    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_M), numBytes));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_N), numBytes));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_P), numBytes));

    double mallocMs = mallocTimer.elapsed_ms();
    CpuTimer h2dTimer;

    cuda_check(cudaMemcpy(d_M, h_M.data(), numBytes, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_N, h_N.data(), numBytes, cudaMemcpyHostToDevice));
    cuda_check(cudaMemset(d_P, 0, numBytes));

    double h2dMs = h2dTimer.elapsed_ms();

    dim3 dimBlock(16, 16, 1);

    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (width + dimBlock.y - 1) / dimBlock.y, 
                  1);

    std::cout << "dimBlock = ("
              << dimBlock.x << ", "
              << dimBlock.y << ", "
              << dimBlock.z << ")" << std::endl;

    std::cout << "dimGrid  = ("
              << dimGrid.x << ", "
              << dimGrid.y << ", "
              << dimGrid.z << ")" << std::endl;

    matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    cudaEvent_t startEvent;
    cudaEvent_t endEvent;

    cuda_check(cudaEventCreate(&startEvent));
    cuda_check(cudaEventCreate(&endEvent));

    cuda_check(cudaEventRecord(startEvent));

    for(int i = 0; i < repeat; ++ i){
        matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
    }

    cuda_check(cudaEventRecord(endEvent));
    cuda_check(cudaEventSynchronize(endEvent));

    cuda_check(cudaGetLastError());

    float kernelTotalMs = 0.0f;

    cuda_check(cudaEventElapsedTime(&kernelTotalMs, startEvent, endEvent));

    double kernelAvgMs = static_cast<double>(kernelTotalMs) / repeat;
    cuda_check(cudaEventDestroy(startEvent));
    cuda_check(cudaEventDestroy(endEvent));

    CpuTimer d2hTimer;

    cuda_check(cudaMemcpy(h_P.data(), d_P, numBytes, cudaMemcpyDeviceToHost));
    double d2hMs = d2hTimer.elapsed_ms();

    bool checkPassed = sample_check_result(h_M,h_N, h_P, width);

    cuda_check(cudaFree(d_M));
    cuda_check(cudaFree(d_N));
    cuda_check(cudaFree(d_P));

    double totalMs = totalTimer.elapsed_ms();

    size_t totalThreads =
        static_cast<size_t>(dimGrid.x) *
        dimGrid.y *
        dimGrid.z *
        dimBlock.x *
        dimBlock.y *
        dimBlock.z;

    double operations =
        2.0 * static_cast<double>(width) *
        static_cast<double>(width) *
        static_cast<double>(width);

    double kernelSeconds = kernelAvgMs / 1000.0;
    double gflops = operations / kernelSeconds / 1.0e9;

    std::cout << std::fixed << std::setprecision(4);

    std::cout << std::endl;
    std::cout << "Timing result" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Host init time       : " << hostInitMs << " ms" << std::endl;
    std::cout << "cudaMalloc time      : " << mallocMs << " ms" << std::endl;
    std::cout << "Host to Device copy  : " << h2dMs << " ms" << std::endl;
    std::cout << "Kernel total time    : " << kernelTotalMs << " ms" << std::endl;
    std::cout << "Kernel average time  : " << kernelAvgMs << " ms" << std::endl;
    std::cout << "Device to Host copy  : " << d2hMs << " ms" << std::endl;
    std::cout << "Total case time      : " << totalMs << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Matrix elements      : " << numElements << std::endl;
    std::cout << "Total CUDA threads   : " << totalThreads << std::endl;
    std::cout << "Sample check         : "
              << (checkPassed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Estimated GFLOPS     : " << gflops << std::endl;
}


int main(){
    CpuTimer cudaInitTimer;

    cuda_check(cudaFree(0));

    double cudaInitMs = cudaInitTimer.elapsed_ms();

    std::cout << "CUDA basic matrix multiplication benchmark" << std::endl;
    std::cout << "CUDA init time: " << cudaInitMs << " ms" << std::endl;


    run_matmul_case(10, 1000);
    run_matmul_case(100, 200);
    run_matmul_case(1000, 20);

    return 0;
}