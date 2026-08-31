#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
                                                                           \
        if (error != cudaSuccess) {                                        \
            std::cerr                                                      \
                << "CUDA error at "                                        \
                << __FILE__ << ":" << __LINE__                             \
                << std::endl                                               \
                << "Error message: "                                       \
                << cudaGetErrorString(error)                               \
                << std::endl;                                              \
                                                                           \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

double bytes_to_mib(std::size_t bytes){
    return static_cast<double>(bytes) /(1024.0 * 1024.0);
}

double bytes_to_Gib(std::size_t bytes){
    return static_cast<double> (bytes) / (1024.0 * 1024.0 * 1024.0);
}

void print_device_properties(int device_id){
    cudaDeviceProp devProp{};

    CUDA_CHECK(cudaGetDeviceProperties(&devProp, device_id));

    int clock_rate_khz = 0;

    CUDA_CHECK(
        cudaDeviceGetAttribute(
            &clock_rate_khz,
            cudaDevAttrClockRate,
            device_id
        )
    );

    std::cout
        << "\n==================================================\n";

    std::cout
        << "CUDA Device " << device_id << '\n';

    std::cout
        << "==================================================\n";

    std::cout
        <<std::left
        <<std::setw(42)
        <<"Device name:"
        <<devProp.name
        <<'\n';

    std::cout
        <<std::left
        <<std::setw(42)
        <<"Compute capability:"
        <<devProp.major
        <<"."
        <<devProp.minor
        <<'\n';

    std::cout   
        <<std::left
        <<std::setw(42)
        <<"Number of SM:"
        <<devProp.multiProcessorCount
        <<'\n';

    std::cout   
        <<std::left
        <<std::setw(42)
        <<"Clock Rate:"
        <<std::fixed
        <<std::setprecision(1)
        <<static_cast<double>(clock_rate_khz) / 1000.0
        <<" MHz"
        <<'\n';

    std::cout
        <<std::left
        <<std::setw(42)
        <<"Total global memory:"
        <<std::fixed
        <<std::setprecision(2)
        <<bytes_to_Gib(devProp.totalGlobalMem)
        <<" GiB"
        <<'\n';

    std::cout << "\n[Warp Information] \n";

    std::cout
        <<std::left
        <<std::setw(42)
        <<"Warp size:"
        <<devProp.warpSize
        <<" threads"
        <<'\n';

    int max_warp_per_SM = devProp.maxThreadsPerMultiProcessor / devProp.warpSize;

    std::cout
        <<std::left
        <<std::setw(42)
        <<"Maximum warps per SM:"
        <<max_warp_per_SM
        <<'\n';

    std::cout
        << "\n[Thread-block limits]\n";


    std::cout
        << std::left
        << std::setw(42)
        << "Maximum threads per block:"
        << devProp.maxThreadsPerBlock
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Maximum block dimension x:"
        << devProp.maxThreadsDim[0]
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Maximum block dimension y:"
        << devProp.maxThreadsDim[1]
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Maximum block dimension z:"
        << devProp.maxThreadsDim[2]
        << '\n';

    std::cout
        << "\n[Grid limits]\n";


    std::cout
        << std::left
        << std::setw(42)
        << "Maximum grid dimension x:"
        << devProp.maxGridSize[0]
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Maximum grid dimension y:"
        << devProp.maxGridSize[1]
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Maximum grid dimension z:"
        << devProp.maxGridSize[2]
        << '\n';

    std::cout
        << "\n[SM resource limits]\n";


    std::cout
        << std::left
        << std::setw(42)
        << "Maximum resident threads per SM:"
        << devProp.maxThreadsPerMultiProcessor
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Maximum resident blocks per SM:"
        << devProp.maxBlocksPerMultiProcessor
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Registers per block:"
        << devProp.regsPerBlock
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Registers per SM:"
        << devProp.regsPerMultiprocessor
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Shared memory per block:"
        << std::fixed
        << std::setprecision(2)
        << bytes_to_mib(devProp.sharedMemPerBlock)
        << " MiB"
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Shared memory per SM:"
        << std::fixed
        << std::setprecision(2)
        << bytes_to_mib(
               devProp.sharedMemPerMultiprocessor
           )
        << " MiB"
        << '\n';

    std::cout
        << "\n[Derived information]\n";


    int registers_per_thread_at_full_occupancy =
        devProp.regsPerMultiprocessor
        / devProp.maxThreadsPerMultiProcessor;


    std::cout
        << std::left
        << std::setw(42)
        << "Registers/thread at full occupancy:"
        << registers_per_thread_at_full_occupancy
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Warps in a 128-thread block:"
        << (128 + devProp.warpSize - 1)
           / devProp.warpSize
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Warps in a 256-thread block:"
        << (256 + devProp.warpSize - 1)
           / devProp.warpSize
        << '\n';


    std::cout
        << std::left
        << std::setw(42)
        << "Warps in a 512-thread block:"
        << (512 + devProp.warpSize - 1)
           / devProp.warpSize
        << '\n';
    
}

int main(){
    int device_count = 0;

    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::cout<<"Number of CUDA-capable devices:" << device_count << '\n';

    if(device_count == 0){
        std::cerr << "No CUDA-capable NVIDIA GPU was detected." <<std::endl;

        return EXIT_FAILURE;
    }

    for(int device_id = 0; device_id < device_count; ++ device_id){
        print_device_properties(device_id);
    }

    std::cout<<"\nGPU property query completed successfully."<<std::endl;

    return EXIT_SUCCESS;
}