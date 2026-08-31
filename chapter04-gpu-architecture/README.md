# Chapter 4 — GPU Architecture and Scheduling

Chapter 4 connects the CUDA programming model to the hardware execution model of an NVIDIA GPU.

The central hierarchy is:

```text
kernel
  ↓
grid
  ↓
blocks
  ↓
SMs
  ↓
warps
  ↓
threads
```

## Key Topics

- GPU organization and streaming multiprocessors (SMs)
- block-to-SM scheduling
- block-level synchronization with `__syncthreads()`
- transparent scalability
- warp organization
- warp ID and lane ID
- SIMT execution
- instruction fetch, decode, and dispatch
- control divergence
- warp scheduling and latency hiding
- resource partitioning
- occupancy
- registers and shared-memory constraints
- CUDA device-property queries

## Notes

- [`notes/chapter04_notes.tex`](notes/chapter04_notes.tex) — LaTeX source
- [`notes/chapter04_notes.pdf`](notes/chapter04_notes.pdf) — compiled notes

The notes are intentionally concise and replace most textbook diagrams with short architectural explanations and concrete CUDA examples.

## Device Properties

[`code/device_properties/`](code/device_properties/)

This utility uses the CUDA Runtime API to inspect the resources of the available GPU.

It reports properties such as:

- number of SMs;
- compute capability;
- warp size;
- thread and block limits;
- grid limits;
- register capacity;
- shared-memory capacity;
- global-memory information;
- simple occupancy-related quantities.

The included README interprets the measured properties of an NVIDIA GeForce RTX 5070 Laptop GPU and connects them to Chapter 4 concepts such as block residency, warps, and occupancy.

## Exercises

[`exercises/chapter04_exercises.md`](exercises/chapter04_exercises.md)

The exercise notes contain complete questions, final answers, and explanations covering:

- warp counting;
- active and divergent warps;
- SIMD efficiency;
- divergence in branches and loops;
- boundary-check divergence;
- barrier synchronization;
- block-size constraints;
- occupancy limits;
- register constraints.

## Main Takeaway

The logical CUDA hierarchy

```text
grid -> blocks -> threads
```

is executed through the hardware hierarchy

```text
GPU -> SMs -> warps -> execution lanes
```

A block is assigned as a complete unit to one SM, then divided into 32-thread warps. Performance depends on how effectively the SM can keep its execution resources busy while respecting limits from threads, warps, blocks, registers, and shared memory.
