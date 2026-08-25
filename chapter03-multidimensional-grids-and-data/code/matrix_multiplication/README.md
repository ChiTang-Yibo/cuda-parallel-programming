# CUDA vs. NumPy CPU Matrix Multiplication Benchmark

## Experiment Setup

This benchmark compares a basic CUDA matrix multiplication kernel with a NumPy CPU implementation. The CUDA version follows the Chapter 3 thread-to-data mapping strategy:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

Each CUDA thread computes one output matrix element:

\[
P_{row,col} = \sum_{k=0}^{N-1} M_{row,k}N_{k,col}
\]

The CUDA implementation is a naive global-memory version. It does not use shared memory, tiling, Tensor Cores, or cuBLAS.

The NumPy CPU implementation uses NumPy matrix multiplication, which is usually backed by an optimized BLAS library. Therefore, it is not a pure Python triple-loop implementation.

## Timing Results

| Matrix Size | Repeat | CUDA Kernel Avg Time | CUDA GFLOPS | NumPy CPU Avg Time | NumPy CPU GFLOPS | Faster Method |
|---:|---:|---:|---:|---:|---:|---|
| \(10 \times 10\) | 1000 | 0.0046 ms | 0.4365 | 0.0008 ms | 2.4468 | NumPy CPU |
| \(100 \times 100\) | 200 | 0.0091 ms | 220.2488 | 0.9369 ms | 2.1347 | CUDA |
| \(1000 \times 1000\) | 20 | 2.9221 ms | 684.4316 | 5.6271 ms | 355.4200 | CUDA |

## Full Workflow Time

| Matrix Size | CUDA Total Case Time | NumPy CPU Total Case Time | Faster Method |
|---:|---:|---:|---|
| \(10 \times 10\) | 92.5241 ms | 1.5749 ms | NumPy CPU |
| \(100 \times 100\) | 3.5810 ms | 188.2256 ms | CUDA |
| \(1000 \times 1000\) | 68.2246 ms | 129.8646 ms | CUDA |

The full CUDA case time includes host initialization, `cudaMalloc`, host-to-device copy, repeated kernel launches, device-to-host copy, memory release, and sample correctness checking. Therefore, for very small matrices, CUDA overhead dominates the total time.

## Speedup Based on Compute Time

For \(100 \times 100\):

\[
\text{Speedup} = \frac{0.9369}{0.0091} \approx 103
\]

For \(1000 \times 1000\):

\[
\text{Speedup} = \frac{5.6271}{2.9221} \approx 1.93
\]

For \(10 \times 10\), NumPy CPU is faster:

\[
\frac{0.0046}{0.0008} \approx 5.75
\]

So NumPy CPU is about 5--6 times faster for the very small matrix case.

## Why the \(10 \times 10\) Case Is Faster on CPU

The \(10 \times 10\) matrix multiplication has only:

\[
2N^3 = 2 \times 10^3 = 2000
\]

floating-point operations. This is too small to utilize the GPU effectively.

The CUDA launch uses one \(16 \times 16\) block, so it creates 256 threads, but only 100 output elements are valid. The remaining 156 threads are removed by the boundary check:

```cpp
if (row < width && col < width)
```

For such a small computation, kernel launch overhead, thread scheduling, and memory-management overhead dominate. Therefore, the CPU version is faster.

## Why \(100 \times 100\) Shows a Large CUDA Advantage

For \(100 \times 100\), the operation count is:

\[
2N^3 = 2 \times 100^3 = 2,000,000
\]

This is large enough for GPU parallelism to become useful. CUDA launches \(7 \times 7\) blocks with \(16 \times 16\) threads per block, giving 12,544 threads, of which 10,000 correspond to valid output elements.

At this size, the GPU can expose much more parallelism than the CPU. The CUDA kernel average time is 0.0091 ms, while the NumPy CPU average time is 0.9369 ms, giving a speedup of about 103 times.

## Why \(1000 \times 1000\) Is Only About 1.9 Times Faster on CUDA

For \(1000 \times 1000\), the operation count is:

\[
2N^3 = 2 \times 1000^3 = 2 \times 10^9
\]

This is a much larger workload, and CUDA achieves about 684 GFLOPS. However, the speedup over NumPy CPU is only about 1.9 times because the CUDA implementation is still naive.

The current CUDA kernel has several limitations:

1. It reads all matrix elements directly from global memory.
2. Neighboring threads repeatedly load many of the same elements from matrices \(M\) and \(N\).
3. Accessing \(N[k \times width + col]\) follows a column-wise pattern, which is not contiguous in row-major memory layout.
4. It does not use shared memory tiling.
5. It does not use cuBLAS or Tensor Cores.

By contrast, NumPy matrix multiplication is usually backed by an optimized CPU BLAS library, which uses cache blocking, SIMD vectorization, and multi-threading. Therefore, the comparison is effectively between a naive CUDA kernel and an optimized CPU BLAS implementation.

## Explanation of the CPU Timing Phenomenon

At first glance, it may look surprising that \(1000 \times 1000\) takes only about 5.6271 ms on CPU while \(100 \times 100\) takes 0.9369 ms. However, this is reasonable when absolute time and computational efficiency are separated.

The \(1000 \times 1000\) case has 1000 times more arithmetic work than the \(100 \times 100\) case:

\[
\frac{2 \times 1000^3}{2 \times 100^3} = 1000
\]

But the runtime only increases by about:

\[
\frac{5.6271}{0.9369} \approx 6.0
\]

This means the CPU BLAS backend becomes much more efficient for larger matrices. For small matrices, function-call overhead, BLAS dispatch overhead, and threading overhead are significant relative to the amount of computation. For large matrices, the computation dominates and optimized BLAS can better use cache blocking, SIMD, and multi-core execution.

Therefore:

- \(100 \times 100\) is faster in absolute time because it has much less work.
- \(1000 \times 1000\) is much more efficient in GFLOPS because the CPU BLAS backend is better utilized.

## Conclusion

This benchmark shows three important points.

First, very small matrix multiplications are not suitable for GPU acceleration because CUDA overhead dominates the computation.

Second, once the matrix size becomes moderately large, CUDA can provide significant acceleration. In this experiment, the \(100 \times 100\) case achieved about 103 times speedup over NumPy CPU compute time.

Third, for large matrices such as \(1000 \times 1000\), even the naive CUDA kernel is faster than NumPy CPU, but the speedup is limited because NumPy uses optimized CPU BLAS while the CUDA implementation does not yet use shared memory tiling or cuBLAS.

The next optimization step is to implement tiled matrix multiplication using shared memory. That version should reduce repeated global memory access and significantly improve CUDA performance.
