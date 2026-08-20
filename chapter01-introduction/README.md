# Chapter 1 — Introduction to Heterogeneous Parallel Computing

Chapter 1 introduces the motivation for GPU computing and the basic heterogeneous CPU–GPU execution model.

## Key Topics

- Transition from single-thread performance scaling to parallel execution
- CPU as a latency-oriented processor
- GPU as a throughput-oriented processor
- CUDA host–device execution model
- Data parallelism
- Speedup and Amdahl's Law
- Sequential and data-parallel portions of applications
- Memory bandwidth as a performance limit
- CPU–GPU division of work
- Work efficiency, arithmetic intensity, load imbalance, synchronization, and atomic operations

## Core Relations

Speedup:

\[
S = \frac{T_{\mathrm{old}}}{T_{\mathrm{new}}}
\]

Amdahl's Law:

\[
S_{\mathrm{total}}
=
\frac{1}{(1-p)+p/s}
\]

Arithmetic intensity:

\[
\text{Arithmetic Intensity}
=
\frac{\text{arithmetic operations}}
{\text{bytes transferred from memory}}
\]

## Files

- [`notes/chapter01_notes.tex`](notes/chapter01_notes.tex) — LaTeX source
- [`notes/chapter01_notes.pdf`](notes/chapter01_notes.pdf) — compiled notes
- [`../reference/cuda_glossary.pdf`](../reference/cuda_glossary.pdf) — terminology reference

Chapter 1 is intentionally introductory. The purpose is to establish the motivation, execution model, and performance vocabulary used throughout the later CUDA chapters.
