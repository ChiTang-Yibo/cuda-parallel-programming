# CUDA Device Properties — NVIDIA GeForce RTX 5070 Laptop GPU

This small Chapter 4 utility queries CUDA runtime and hardware properties and connects the CUDA programming model to the resources of the actual GPU.

The accompanying CUDA program reports device information with `cudaGetDeviceCount`, `cudaGetDeviceProperties`, and selected `cudaDeviceGetAttribute` queries.

## Files

- `device_properties.cu` — CUDA device-property query program
- `README.md` — interpretation of the measured RTX 5070 Laptop GPU properties

## What the Program Queries

The program reports:

- CUDA runtime and driver-supported versions;
- device name and compute capability;
- number of streaming multiprocessors (SMs);
- reported core and memory clocks;
- total global memory;
- memory-bus width and L2 cache size;
- warp size;
- maximum block and grid dimensions;
- maximum resident threads, warps, and blocks per SM;
- register resources;
- shared-memory resources;
- unified-addressing and managed-memory support;
- several simple occupancy-related quantities.

The clock-rate queries use `cudaDeviceGetAttribute` so the program remains compatible with CUDA versions where some clock-related fields are no longer exposed directly through `cudaDeviceProp`.

## Measured Device

The tested system contains one CUDA-capable device:

- **GPU:** NVIDIA GeForce RTX 5070 Laptop GPU
- **Compute capability:** 12.0
- **Streaming multiprocessors:** 36
- **Reported core clock:** 1425.0 MHz
- **Total global memory:** 7.96 GiB

These values are runtime-reported properties of the tested machine rather than general specifications for every RTX 5070 Laptop GPU configuration.

## Warp and Thread Organization

The reported warp size is 32 threads.

```math
1\ \text{warp} = 32\ \text{threads}
```

Each SM supports at most 48 resident warps, so the maximum resident thread count is

```math
48 \times 32 = 1536\ \text{threads/SM}.
```

For common block sizes:

| Threads per block | Warps per block |
|---:|---:|
| 128 | 4 |
| 256 | 8 |
| 512 | 16 |
| 1024 | 32 |

In general,

```math
\text{warps per block}
=
\left\lceil
\frac{\text{threads per block}}{32}
\right\rceil.
```

## Thread-Block Limits

The maximum number of threads in one block is 1024.

The reported maximum block dimensions are:

```math
\begin{aligned}
\texttt{blockDim.x} &\le 1024,\\
\texttt{blockDim.y} &\le 1024,\\
\texttt{blockDim.z} &\le 64.
\end{aligned}
```

A valid block must also satisfy

```math
\texttt{blockDim.x}
\times
\texttt{blockDim.y}
\times
\texttt{blockDim.z}
\le 1024.
```

Examples:

```text
dim3 block(16, 16)  ->  256 threads   -> valid
dim3 block(32, 32)  -> 1024 threads   -> valid
dim3 block(64, 32)  -> 2048 threads   -> invalid
```

## Grid Limits

The reported maximum grid dimensions are:

```math
\begin{aligned}
\texttt{gridDim.x} &\le 2{,}147{,}483{,}647,\\
\texttt{gridDim.y} &\le 65{,}535,\\
\texttt{gridDim.z} &\le 65{,}535.
\end{aligned}
```

These values limit the number of blocks that may be launched along each grid dimension.

## Per-SM Resource Limits

The measured device reports up to:

- **1536 resident threads per SM**
- **48 resident warps per SM**
- **24 resident blocks per SM**
- **65,536 32-bit registers per SM**
- **approximately 0.10 MiB shared memory per SM**

The default per-block resources reported in the benchmark are approximately:

- **65,536 registers per block**
- **0.05 MiB shared memory per block**

The CUDA program also prints `sharedMemPerBlockOptin` separately. The opt-in limit can differ from the default per-block shared-memory limit, so the runtime output should be treated as the authoritative value when configuring a kernel that explicitly requests additional dynamic shared memory.

## Occupancy from Thread and Block Limits

Ignoring register and shared-memory constraints, the thread/block limits give:

| Block size | Maximum resident blocks | Resident threads | Theoretical thread occupancy |
|---:|---:|---:|---:|
| 32 | 24 | 768 | 50% |
| 64 | 24 | 1536 | 100% |
| 128 | 12 | 1536 | 100% |
| 256 | 6 | 1536 | 100% |
| 512 | 3 | 1536 | 100% |
| 1024 | 1 | 1024 | 66.7% |

Using resident threads,

```math
\text{occupancy}
=
\frac{\text{resident threads per SM}}{1536}.
```

A 32-thread block reaches the 24-block limit before filling all thread slots. A 1024-thread block leaves thread capacity unused because only one complete 1024-thread block fits under the 1536-thread limit.

This table is only a simplified theoretical calculation. Real kernel occupancy must also account for register allocation, shared-memory usage, warp limits, and hardware allocation granularity.

## Register Constraint

The device reports 65,536 registers per SM. Dividing by the maximum 1536 resident threads gives an average full-thread-occupancy budget of

```math
\frac{65{,}536}{1536}
\approx 42.67
```

registers per thread.

The utility prints 42 because the corresponding helper calculation uses integer division.

This value is useful as an initial intuition, but it is not a strict per-thread threshold: actual occupancy depends on block size and the hardware's register-allocation granularity.

For a simplified example with 64 registers per thread:

```math
\left\lfloor
\frac{65{,}536}{64}
\right\rfloor
=
1024
```

threads can be supported by register capacity alone, corresponding to roughly

```math
\frac{1024}{1536}
\approx 66.7\%.
```

## Shared-Memory Constraint

The reported shared-memory capacity is approximately:

- **0.10 MiB per SM**
- **0.05 MiB per block** under the default per-block limit

A kernel that uses a large amount of shared memory per block can therefore reduce the number of resident blocks even when the thread limit would allow more.

This directly connects Chapter 4 resource partitioning to kernel design:

```text
more shared memory per block
        ↓
fewer resident blocks
        ↓
fewer resident warps
        ↓
potentially less latency hiding
```

## Practical Interpretation

For this GPU, 128-thread and 256-thread blocks are reasonable starting points for experiments:

- 128 threads = 4 warps per block
- 256 threads = 8 warps per block

Both can theoretically fill all 1536 thread slots when considering only thread and block limits.

However, block size should not be selected from occupancy alone. Real performance also depends on:

- registers per thread;
- shared memory per block;
- global-memory access patterns;
- warp divergence;
- instruction mix;
- arithmetic intensity;
- available instruction-level parallelism.

## Build and Run

Compile with NVCC:

```bash
nvcc device_properties.cu -o device_properties
```

On Windows PowerShell:

```powershell
nvcc .\device_properties.cu -o .\device_properties.exe
.\device_properties.exe
```

The program does not launch a computation kernel. It queries the CUDA runtime and prints the properties of every CUDA-capable NVIDIA device detected in the system.

## Chapter 4 Connection

This utility turns the architectural concepts from Chapter 4 into concrete hardware limits:

```text
CUDA program
    ↓
device properties
    ↓
SM count
warp size
thread/block limits
register capacity
shared-memory capacity
    ↓
block residency and occupancy
```

The important lesson is that launch configuration and occupancy are constrained by the real resources of the GPU, not only by the logical CUDA grid/block model.
