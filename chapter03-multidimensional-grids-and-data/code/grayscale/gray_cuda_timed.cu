#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iomanip>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define check_cuda(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " : " << cudaGetErrorString(err) << std::endl;      \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                    \
    } while (0)

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(const Clock::time_point& start,
                         const Clock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================
// CUDA kernel: RGB image to grayscale image
// ============================================================
__global__
void colorToGrayscaleConversion(
    const unsigned char* Pin,
    unsigned char* Pout,
    int width,
    int height
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * 3;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        float gray = 0.21f * r + 0.71f * g + 0.07f * b;
        Pout[grayOffset] = static_cast<unsigned char>(gray);
    }
}

int main(int argc, char** argv) {
    auto totalStart = Clock::now();

    // ========================================================
    // 1. Input and output paths
    // ========================================================
    std::string inputPath = R"(C:\Users\12504\Desktop\6-8\CUDA\chapter3\gray\sample.png)";
    std::string outputPath = R"(C:\Users\12504\Desktop\6-8\CUDA\chapter3\gray\sample_gray_cuda_timed.png)";
    int repeat = 100;

    // Optional command-line override:
    // .\gray_cuda_timed.exe input.png output.png 100
    if (argc >= 2) {
        inputPath = argv[1];
    }
    if (argc >= 3) {
        outputPath = argv[2];
    }
    if (argc >= 4) {
        repeat = std::atoi(argv[3]);
        if (repeat <= 0) {
            repeat = 1;
        }
    }

    // ========================================================
    // 2. Load image on host
    // ========================================================
    auto loadStart = Clock::now();

    int width = 0;
    int height = 0;
    int originalChannels = 0;
    const int channels = 3;

    unsigned char* h_Pin = stbi_load(
        inputPath.c_str(),
        &width,
        &height,
        &originalChannels,
        channels
    );

    if (h_Pin == nullptr) {
        std::cerr << "Failed to load the image: " << inputPath << std::endl;
        return EXIT_FAILURE;
    }

    auto loadEnd = Clock::now();

    size_t rgbBytes = static_cast<size_t>(width) * height * channels * sizeof(unsigned char);
    size_t grayPixels = static_cast<size_t>(width) * height;
    size_t grayBytes = grayPixels * sizeof(unsigned char);

    std::vector<unsigned char> h_Pout(grayPixels);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Input image: " << inputPath << std::endl;
    std::cout << "Width  = " << width << std::endl;
    std::cout << "Height = " << height << std::endl;
    std::cout << "Original channels = " << originalChannels << std::endl;
    std::cout << "Loaded channels   = " << channels << std::endl;
    std::cout << "Repeat = " << repeat << std::endl;

    // ========================================================
    // 3. Allocate device memory
    // ========================================================
    unsigned char* d_Pin = nullptr;
    unsigned char* d_Pout = nullptr;

    auto mallocStart = Clock::now();
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_Pin), rgbBytes));
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_Pout), grayBytes));
    auto mallocEnd = Clock::now();

    // ========================================================
    // 4. Configure CUDA grid and block
    // ========================================================
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

    // ========================================================
    // 5. Create CUDA events for GPU-side timing
    // ========================================================
    cudaEvent_t startEvent, stopEvent;
    check_cuda(cudaEventCreate(&startEvent));
    check_cuda(cudaEventCreate(&stopEvent));

    float h2dMs = 0.0f;
    float kernelTotalMs = 0.0f;
    float d2hMs = 0.0f;

    // ========================================================
    // 6. Copy input image from host to device
    // ========================================================
    check_cuda(cudaEventRecord(startEvent));
    check_cuda(cudaMemcpy(d_Pin, h_Pin, rgbBytes, cudaMemcpyHostToDevice));
    check_cuda(cudaEventRecord(stopEvent));
    check_cuda(cudaEventSynchronize(stopEvent));
    check_cuda(cudaEventElapsedTime(&h2dMs, startEvent, stopEvent));

    // ========================================================
    // 7. Warm-up kernel
    //    This avoids mixing first-launch overhead with kernel timing.
    // ========================================================
    colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(d_Pin, d_Pout, width, height);
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    // ========================================================
    // 8. Timed kernel execution
    //    CUDA event timing measures GPU execution time on the stream.
    //    It does not include image loading, saving, or CPU launch overhead.
    // ========================================================
    check_cuda(cudaEventRecord(startEvent));
    for (int i = 0; i < repeat; ++i) {
        colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(d_Pin, d_Pout, width, height);
    }
    check_cuda(cudaEventRecord(stopEvent));
    check_cuda(cudaEventSynchronize(stopEvent));
    check_cuda(cudaGetLastError());
    check_cuda(cudaEventElapsedTime(&kernelTotalMs, startEvent, stopEvent));

    // ========================================================
    // 9. Copy output image from device to host
    // ========================================================
    check_cuda(cudaEventRecord(startEvent));
    check_cuda(cudaMemcpy(h_Pout.data(), d_Pout, grayBytes, cudaMemcpyDeviceToHost));
    check_cuda(cudaEventRecord(stopEvent));
    check_cuda(cudaEventSynchronize(stopEvent));
    check_cuda(cudaEventElapsedTime(&d2hMs, startEvent, stopEvent));

    // ========================================================
    // 10. Save grayscale image
    // ========================================================
    auto saveStart = Clock::now();

    int success = stbi_write_png(
        outputPath.c_str(),
        width,
        height,
        1,
        h_Pout.data(),
        width
    );

    auto saveEnd = Clock::now();

    if (!success) {
        std::cerr << "Failed to save the image: " << outputPath << std::endl;
    } else {
        std::cout << "Saved grayscale image: " << outputPath << std::endl;
    }

    // ========================================================
    // 11. Report timing
    // ========================================================
    auto totalEnd = Clock::now();

    double loadMs = elapsed_ms(loadStart, loadEnd);
    double mallocMs = elapsed_ms(mallocStart, mallocEnd);
    double saveMs = elapsed_ms(saveStart, saveEnd);
    double totalMs = elapsed_ms(totalStart, totalEnd);

    std::cout << "\nTiming result" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Image load time        : " << loadMs << " ms" << std::endl;
    std::cout << "cudaMalloc time        : " << mallocMs << " ms" << std::endl;
    std::cout << "Host to Device copy    : " << h2dMs << " ms" << std::endl;
    std::cout << "Kernel total time      : " << kernelTotalMs << " ms" << std::endl;
    std::cout << "Kernel average time    : " << kernelTotalMs / repeat << " ms" << std::endl;
    std::cout << "Device to Host copy    : " << d2hMs << " ms" << std::endl;
    std::cout << "Image save time        : " << saveMs << " ms" << std::endl;
    std::cout << "Total program time     : " << totalMs << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Pixels                 : " << grayPixels << std::endl;
    std::cout << "Total CUDA threads     : "
              << static_cast<size_t>(dimGrid.x) * dimGrid.y * dimBlock.x * dimBlock.y
              << std::endl;

    // ========================================================
    // 12. Free memory
    // ========================================================
    check_cuda(cudaEventDestroy(startEvent));
    check_cuda(cudaEventDestroy(stopEvent));
    check_cuda(cudaFree(d_Pin));
    check_cuda(cudaFree(d_Pout));
    stbi_image_free(h_Pin);

    return 0;
}
