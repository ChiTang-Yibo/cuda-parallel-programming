#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define cuda_check(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                    \
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
void blurKernelRGB(const unsigned char* Pin,
                   unsigned char* Pout,
                   int width,
                   int height,
                   int blurRadius)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    const int channels = 3;

    if (col < width && row < height) {
        int sumR = 0;
        int sumG = 0;
        int sumB = 0;
        int pixels = 0;

        // Average all valid pixels in the local blur window.
        for (int blurRow = -blurRadius; blurRow <= blurRadius; ++blurRow) {
            for (int blurCol = -blurRadius; blurCol <= blurRadius; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow >= 0 && curRow < height &&
                    curCol >= 0 && curCol < width) {

                    int pixelOffset = curRow * width + curCol;
                    int rgbOffset = pixelOffset * channels;

                    sumR += Pin[rgbOffset];
                    sumG += Pin[rgbOffset + 1];
                    sumB += Pin[rgbOffset + 2];

                    ++pixels;
                }
            }
        }

        int outOffset = (row * width + col) * channels;

        Pout[outOffset]     = static_cast<unsigned char>(sumR / pixels);
        Pout[outOffset + 1] = static_cast<unsigned char>(sumG / pixels);
        Pout[outOffset + 2] = static_cast<unsigned char>(sumB / pixels);
    }
}

int main(int argc, char** argv)
{
    std::string inputPath =
        R"(C:\Users\12504\Desktop\6-8\CUDA\chapter3\blur\sample.png)";

    std::string outputPath =
        R"(C:\Users\12504\Desktop\6-8\CUDA\chapter3\blur\sample_blur_cuda.png)";

    int repeat = 100;
    int blurRadius = 3;

    if (argc >= 2) {
        inputPath = argv[1];
    }

    if (argc >= 3) {
        outputPath = argv[2];
    }

    if (argc >= 4) {
        repeat = std::atoi(argv[3]);
        if (repeat < 1) {
            repeat = 1;
        }
    }

    if (argc >= 5) {
        blurRadius = std::atoi(argv[4]);
        if (blurRadius < 0) {
            blurRadius = 0;
        }
    }

    const int channels = 3;

    CpuTimer totalTimer;

    // Initialize CUDA runtime before formal timing.
    CpuTimer cudaInitTimer;
    cuda_check(cudaFree(0));
    double cudaInitMs = cudaInitTimer.elapsed_ms();

    CpuTimer loadTimer;

    int width = 0;
    int height = 0;
    int originalChannels = 0;

    unsigned char* h_Pin = stbi_load(
        inputPath.c_str(),
        &width,
        &height,
        &originalChannels,
        channels
    );

    double loadMs = loadTimer.elapsed_ms();

    if (h_Pin == nullptr) {
        std::cerr << "Failed to load image: " << inputPath << std::endl;
        return EXIT_FAILURE;
    }

    size_t imageBytes =
        static_cast<size_t>(width) * height * channels;

    std::vector<unsigned char> h_Pout(imageBytes);

    std::cout << "Input image: " << inputPath << std::endl;
    std::cout << "Width  = " << width << std::endl;
    std::cout << "Height = " << height << std::endl;
    std::cout << "Original channels = " << originalChannels << std::endl;
    std::cout << "Loaded channels   = " << channels << std::endl;
    std::cout << "Repeat = " << repeat << std::endl;
    std::cout << "Blur radius = " << blurRadius << std::endl;
    std::cout << "Blur window = "
              << (2 * blurRadius + 1)
              << " x "
              << (2 * blurRadius + 1)
              << std::endl;

    unsigned char* d_Pin = nullptr;
    unsigned char* d_Pout = nullptr;

    CpuTimer mallocTimer;

    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Pin), imageBytes));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Pout), imageBytes));

    double mallocMs = mallocTimer.elapsed_ms();

    CpuTimer h2dTimer;

    cuda_check(cudaMemcpy(
        d_Pin,
        h_Pin,
        imageBytes,
        cudaMemcpyHostToDevice
    ));

    double h2dMs = h2dTimer.elapsed_ms();

    dim3 dimBlock(16, 16, 1);

    dim3 dimGrid(
        (width  + dimBlock.x - 1) / dimBlock.x,
        (height + dimBlock.y - 1) / dimBlock.y,
        1
    );

    std::cout << "dimBlock = ("
              << dimBlock.x << ", "
              << dimBlock.y << ", "
              << dimBlock.z << ")" << std::endl;

    std::cout << "dimGrid  = ("
              << dimGrid.x << ", "
              << dimGrid.y << ", "
              << dimGrid.z << ")" << std::endl;

    // Warm-up launch.
    blurKernelRGB<<<dimGrid, dimBlock>>>(
        d_Pin,
        d_Pout,
        width,
        height,
        blurRadius
    );

    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    cuda_check(cudaEventCreate(&startEvent));
    cuda_check(cudaEventCreate(&stopEvent));

    cuda_check(cudaEventRecord(startEvent));

    for (int i = 0; i < repeat; ++i) {
        blurKernelRGB<<<dimGrid, dimBlock>>>(
            d_Pin,
            d_Pout,
            width,
            height,
            blurRadius
        );
    }

    cuda_check(cudaEventRecord(stopEvent));
    cuda_check(cudaEventSynchronize(stopEvent));

    cuda_check(cudaGetLastError());

    float kernelTotalMs = 0.0f;

    cuda_check(cudaEventElapsedTime(
        &kernelTotalMs,
        startEvent,
        stopEvent
    ));

    double kernelAvgMs =
        static_cast<double>(kernelTotalMs) / repeat;

    cuda_check(cudaEventDestroy(startEvent));
    cuda_check(cudaEventDestroy(stopEvent));

    CpuTimer d2hTimer;

    cuda_check(cudaMemcpy(
        h_Pout.data(),
        d_Pout,
        imageBytes,
        cudaMemcpyDeviceToHost
    ));

    double d2hMs = d2hTimer.elapsed_ms();

    CpuTimer saveTimer;

    int success = stbi_write_png(
        outputPath.c_str(),
        width,
        height,
        channels,
        h_Pout.data(),
        width * channels
    );

    double saveMs = saveTimer.elapsed_ms();

    if (!success) {
        std::cerr << "Failed to save image: " << outputPath << std::endl;
    } else {
        std::cout << "Saved blurred image: " << outputPath << std::endl;
    }

    cuda_check(cudaFree(d_Pin));
    cuda_check(cudaFree(d_Pout));

    stbi_image_free(h_Pin);

    double totalMs = totalTimer.elapsed_ms();

    size_t totalThreads =
        static_cast<size_t>(dimGrid.x) *
        dimGrid.y *
        dimGrid.z *
        dimBlock.x *
        dimBlock.y *
        dimBlock.z;

    size_t pixels =
        static_cast<size_t>(width) * height;

    std::cout << std::endl;
    std::cout << "Timing result" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "CUDA init time        : " << cudaInitMs << " ms" << std::endl;
    std::cout << "Image load time       : " << loadMs << " ms" << std::endl;
    std::cout << "cudaMalloc time       : " << mallocMs << " ms" << std::endl;
    std::cout << "Host to Device copy   : " << h2dMs << " ms" << std::endl;
    std::cout << "Kernel total time     : " << kernelTotalMs << " ms" << std::endl;
    std::cout << "Kernel average time   : " << kernelAvgMs << " ms" << std::endl;
    std::cout << "Device to Host copy   : " << d2hMs << " ms" << std::endl;
    std::cout << "Image save time       : " << saveMs << " ms" << std::endl;
    std::cout << "Total program time    : " << totalMs << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Pixels                : " << pixels << std::endl;
    std::cout << "Total CUDA threads    : " << totalThreads << std::endl;

    return 0;
}