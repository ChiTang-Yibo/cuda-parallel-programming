# Chapter 2 — Heterogeneous Data Parallelism

Chapter 2 introduces the basic CUDA programming model by mapping independent data elements to GPU threads. The main example is vector addition, which connects the host–device workflow, kernel launches, thread indexing, device memory management, and CUDA compilation.

## Key Topics

- Heterogeneous CPU–GPU execution
- Data parallelism
- CUDA kernels
- Grid, block, and thread hierarchy
- Global thread indexing
- Kernel launch configuration
- Boundary checking
- `__host__`, `__device__`, and `__global__`
- Device global memory
- `cudaMalloc`, `cudaMemcpy`, and `cudaFree`
- CUDA error checking
- `nvcc`, PTX, and the host/device compilation flow

## Core Relations

### Global Thread Index

```math
i = \texttt{blockIdx.x}\times\texttt{blockDim.x}+\texttt{threadIdx.x}
```

Each thread computes a unique global index and uses it to select the data element it processes.

### Number of Blocks

For a vector of length \(n\),

```math
N_{\mathrm{blocks}}
=
\left\lceil
\frac{n}{N_{\mathrm{threads/block}}}
\right\rceil
```

In integer C/C++ arithmetic this is commonly written as:

```cpp
int numBlocks = (n + blockSize - 1) / blockSize;
```

Because the launched grid may contain slightly more threads than valid data elements, the kernel protects memory accesses with:

```cpp
if (i < n) {
    C[i] = A[i] + B[i];
}
```

## CUDA Workflow

The basic host–device workflow introduced in this chapter is:

```text
allocate device memory
        ↓
copy input data to the device
        ↓
launch the CUDA kernel
        ↓
execute many GPU threads
        ↓
copy results back to the host
        ↓
free device memory
```

The corresponding CUDA Runtime API functions are mainly:

```text
cudaMalloc
cudaMemcpy
kernel<<<grid, block>>>(...)
cudaFree
```

## Vector Addition Example

The CUDA implementation in [`code/vec_add.cu`](code/vec_add.cu) demonstrates the complete workflow:

- host memory initialization;
- device memory allocation;
- host-to-device memory transfer;
- one-dimensional kernel launch;
- global thread indexing;
- boundary checking;
- device-to-host result transfer;
- CUDA API and kernel error checking;
- CPU-side correctness verification.

The kernel itself is intentionally simple:

```cpp
__global__ void vecAddKernel(
    const float* A,
    const float* B,
    float* C,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

A typical compilation command is:

```bash
nvcc code/vec_add.cu -o vec_add
```

Generated files such as `.exe`, `.obj`, `.exp`, and `.lib` are build artifacts and are excluded from version control.

## CUDA Compilation Model

A CUDA `.cu` file may contain both host and device code. `nvcc` coordinates the compilation of these two parts.

```text
CUDA C/C++ source (.cu)
        ↓
       nvcc
      /    \
host code   device code
   ↓            ↓
host compiler   PTX / GPU code
   ↓            ↓
 CPU            GPU
```

PTX stands for **Parallel Thread Execution** and acts as an intermediate representation for CUDA device code. The NVIDIA driver can translate PTX into machine code for the target GPU.

## Files

- [`notes/chapter02_notes.tex`](notes/chapter02_notes.tex) — LaTeX source
- [`notes/chapter02_notes.pdf`](notes/chapter02_notes.pdf) — compiled chapter notes
- [`code/vec_add.cu`](code/vec_add.cu) — basic CUDA vector-addition implementation
- [`code/vec_add_experiment.cu`](code/vec_add_experiment.cu) — configurable vector-addition experiment
- [`code/vec_add_experiment.md`](code/vec_add_experiment.md) — experiment setup, results, and analysis
- [`exercises/cuda_ch2_exercises_answers.md`](exercises/cuda_ch2_exercises_answers.md) — exercise answers and explanations
- [`../reference/cuda_glossary.pdf`](../reference/cuda_glossary.pdf) — CUDA and parallel-computing terminology reference

## Chapter Takeaway

The central idea of Chapter 2 is the mapping

```text
independent data elements
        ↓
independent CUDA threads
        ↓
parallel GPU execution
```

The CPU remains responsible for program control and kernel launches, while the GPU executes the data-parallel portion using a grid of threads.
