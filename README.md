# CUDA Parallel Programming

Notes, exercises, and CUDA implementations developed while studying GPU architecture, heterogeneous computing, and massively parallel programming.

Primary reference:

> Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj,  
> *Programming Massively Parallel Processors: A Hands-on Approach*, 4th Edition.

## Goals

This repository focuses on:

- heterogeneous CPU–GPU computing;
- CUDA execution and memory models;
- GPU architecture and scheduling;
- memory locality and data reuse;
- performance analysis and optimization;
- common parallel programming patterns;
- CUDA deep-learning operators;
- NVIDIA Nsight and Triton;
- Transformer inference profiling.

The long-term learning path is:

```text
CUDA fundamentals
        ↓
GPU architecture
        ↓
kernel optimization
        ↓
deep-learning operators
        ↓
Nsight / Triton
        ↓
Transformer inference profiling
```

## Repository Structure

```text
cuda-parallel-programming/
├── README.md
├── reference/
│   ├── cuda_glossary.tex
│   └── cuda_glossary.pdf
│
├── chapter01-introduction/
│   ├── README.md
│   └── notes/
│       ├── chapter01_notes.tex
│       └── chapter01_notes.pdf
│
├── chapter02-heterogeneous-data-parallelism/
├── chapter03-multidimensional-grids/
├── chapter04-gpu-architecture/
├── chapter05-memory-architecture/
└── ...
```

## Current Progress

- [x] Chapter 1 — Introduction to Heterogeneous Parallel Computing
- [x] Chapter 2 — Heterogeneous Data Parallelism
- [ ] Chapter 3 — Multidimensional Grids and Data
- [ ] Chapter 4 — GPU Architecture and Scheduling
- [ ] Chapter 5 — Memory Architecture and Data Locality
- [ ] Chapter 6 — Performance Considerations

## Workflow

For programming topics, the preferred workflow is:

```text
concept
   ↓
CPU reference
   ↓
CUDA implementation
   ↓
correctness check
   ↓
benchmark
   ↓
profiling
   ↓
optimization
```

Most notes are provided in both LaTeX source and compiled PDF form. The notes are intentionally concise and are intended as technical study references rather than reproductions of the textbook.

## Disclaimer

This is a personal educational repository based on my own study, implementations, and summaries. It is not affiliated with or endorsed by the authors or publisher of the primary reference.
