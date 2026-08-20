# Chapter 1 — Introduction to Heterogeneous Parallel Computing

Chapter 1 introduces the motivation for GPU computing and the basic heterogeneous CPU–GPU execution model.

## Key Topics

- Transition from single-thread performance scaling to parallel execution
- CPU as a **latency-oriented** processor
- GPU as a **throughput-oriented** processor
- CUDA host–device execution model
- Data parallelism
- Speedup and Amdahl's Law
- Sequential and data-parallel portions of applications
- Memory bandwidth as a performance limit
- CPU–GPU division of work
- Work efficiency, arithmetic intensity, load imbalance, synchronization, and atomic operations

## Core Relations

### Speedup

```math
S = \frac{T_{\mathrm{old}}}{T_{\mathrm{new}}}
```

### Amdahl's Law

```math
S_{\mathrm{total}} = \frac{1}{(1-p)+\frac{p}{s}}
```

where $p$ is the parallelizable fraction of the original execution time and $s$ is the speedup of that parallel portion.

As $s \rightarrow \infty$,

```math
S_{\mathrm{total,max}} = \frac{1}{1-p}
```

so the sequential fraction sets an upper bound on total application speedup.

### Arithmetic Intensity

```math
\text{Arithmetic Intensity}
=
\frac{\text{Arithmetic Operations}}{\text{Bytes Transferred from Memory}}
```

Low arithmetic intensity often indicates a memory-bandwidth bottleneck, while higher arithmetic intensity makes it more likely that performance is limited by computation.

## Files

- [`notes/chapter01_notes.tex`](notes/chapter01_notes.tex) — LaTeX source
- [`notes/chapter01_notes.pdf`](notes/chapter01_notes.pdf) — compiled notes
- [`../reference/cuda_glossary.pdf`](../reference/cuda_glossary.pdf) — CUDA and parallel-computing terminology reference

Chapter 1 is intentionally introductory. Its purpose is to establish the motivation, heterogeneous execution model, and performance vocabulary used throughout the later CUDA chapters.
