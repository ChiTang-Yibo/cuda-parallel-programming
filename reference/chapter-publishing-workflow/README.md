# CUDA Chapter Publishing Workflow

This document defines the standard workflow for adding a completed chapter to the `cuda-parallel-programming` repository.

The goal is to keep every chapter consistent, easy to browse on GitHub, and easy to extend later with additional CUDA experiments.

---

## 1. Recommended Chapter Structure

Use one top-level folder for each chapter.

Example:

```text
cuda-parallel-programming/
├── README.md
├── reference/
│   └── ...
├── chapter01-introduction/
├── chapter02-heterogeneous-data-parallelism/
├── ...
└── chapterXX-short-topic-name/
    ├── README.md
    ├── notes/
    │   ├── chapterXX_notes.tex
    │   └── chapterXX_notes.pdf
    ├── exercises/
    │   ├── chapterXX_exercises.md
    │   └── exercise_or_benchmark.cu
    └── code/
        ├── main_experiment.cu
        └── README.md
```

The structure does not need to be artificially expanded. Only create files that are actually useful.

A typical completed chapter should contain:

- a root `README.md`;
- a `notes/` folder;
- an `exercises/` folder;
- a `code/` folder when the chapter contains CUDA implementations or benchmarks.

---

## 2. File Naming Rules

### Chapter folder

Use:

```text
chapterXX-short-topic-name
```

Examples:

```text
chapter05-memory-architecture
chapter06-performance-considerations
```

Use lowercase words separated by hyphens.

### Notes

Use matching names for the LaTeX source and compiled PDF:

```text
notes/
├── chapter06_notes.tex
└── chapter06_notes.pdf
```

### Exercises

Use:

```text
exercises/chapter06_exercises.md
```

If an exercise contains an implementation or benchmark, place the source file in the same `exercises/` folder when it directly belongs to that exercise:

```text
exercises/
├── chapter06_exercises.md
└── corner_turning_benchmark.cu
```

### Code

Use descriptive names:

```text
coarse_matmul_benchmark.cu
tiled_matmul.cu
vector_add.cu
```

Avoid names such as:

```text
test1.cu
new.cu
code_final2.cu
```

---

## 3. Root README for Each Chapter

Every chapter folder should contain:

```text
chapterXX-topic/
└── README.md
```

GitHub automatically renders `README.md` when the folder is opened.

The chapter README should be concise and act as an index rather than duplicate the full notes.

A useful structure is:

```markdown
# Chapter X — Topic Name

Short description of the chapter.

## Topics

- Topic A
- Topic B
- Topic C

---

## Contents

```text
chapterXX-topic/
├── README.md
├── notes/
├── exercises/
└── code/
```

---

## Notes

- [PDF Notes](notes/chapterXX_notes.pdf)
- [LaTeX Source](notes/chapterXX_notes.tex)

---

## Exercises

- [Exercise Summary](exercises/chapterXX_exercises.md)

---

## Code

- [`example_kernel.cu`](code/example_kernel.cu)
- [Experiment Summary](code/README.md)

---

## Key Takeaways

- Main idea 1
- Main idea 2
- Main idea 3
```

Do not copy the entire chapter note into the chapter README.

The chapter README should answer:

1. What is this chapter about?
2. What files are included?
3. Which implementation or benchmark is important?
4. What are the main technical conclusions?

---

## 4. Markdown Style for GitHub

All Markdown files should use GitHub-Flavored Markdown and should be checked directly on GitHub after pushing.

### 4.1 Headings

Use:

```markdown
# Main Title

## Section

### Subsection
```

Do not manually number every paragraph unless numbering is useful for the document.

---

### 4.2 Section Separators

Use:

```markdown
---
```

between major sections when it improves readability.

Example:

```markdown
## Experimental Setup

...

---

## Results
```

The `---` separator is only a visual horizontal rule. It is not responsible for rendering mathematical equations.

---

### 4.3 Inline Code

Use single backticks:

```markdown
`threadIdx.x`
`cudaMalloc`
`TILE_WIDTH`
```

Example:

```markdown
The block contains `TILE_WIDTH x TILE_WIDTH` threads.
```

---

### 4.4 Code Blocks

Always specify the language when possible.

CUDA / C++:

````markdown
```cpp
__global__ void kernel(float* x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
}
```
````

Shell commands:

````markdown
```bash
nvcc -std=c++17 example.cu -o example
```
````

Plain terminal output or directory trees:

````markdown
```text
Verification: PASSED
GPU time: 0.1534 ms
```
````

---

## 5. Mathematical Equations on GitHub

For this repository, use the same style that rendered correctly in Chapters 5 and 6.

### 5.1 Inline Math

Inline math can use:

```markdown
$N \times N$
```

Example:

```markdown
For an $N \times N$ matrix, the useful work is approximately $2N^3$ FLOPs.
```

---

### 5.2 Display Math

For important standalone equations, use a fenced `math` block:

````markdown
```math
F_{\text{useful}} = 2N^3
```
````

For multi-line derivations:

````markdown
```math
AI
=
\frac{2CT^3}
{4(C+1)T^2}
=
\frac{CT}{2(C+1)}
```
````

For ceiling division:

````markdown
```math
\texttt{gridDim.x}
=
\left\lceil
\frac{W}{CT}
\right\rceil
```
````

This style should be preferred over:

```text
$$
...
$$
```

because the fenced `math` format has rendered more reliably in this repository.

---

### 5.3 Avoid Overly Complex LaTeX Environments

For GitHub Markdown, prefer simple expressions.

Prefer:

````markdown
```math
T = 16,\qquad C = 4
```
````

instead of relying on large LaTeX environments such as:

```text
\begin{align}
...
\end{align}
```

The full LaTeX note can contain more sophisticated formatting. The Markdown version should prioritize reliable GitHub rendering.

---

## 6. Tables

Use standard Markdown tables.

Example:

```markdown
| N | GPU Time (ms) | GFLOP/s | Check |
|---:|---:|---:|:---:|
| 256 | 0.0250 | 1342.18 | PASS |
| 512 | 0.1477 | 1817.74 | PASS |
```

Recommended alignment:

- labels: left-aligned;
- numerical values: right-aligned;
- short status values such as `PASS`: centered.

Do not make tables unnecessarily wide. If a table becomes difficult to read, split it into two smaller tables.

---

## 7. Links

Use relative repository paths.

Correct:

```markdown
[Chapter Notes](notes/chapter06_notes.pdf)
```

```markdown
[`coarse_matmul_benchmark.cu`](code/coarse_matmul_benchmark.cu)
```

Avoid local Windows paths such as:

```text
C:\Users\...\chapter06_notes.pdf
```

Relative links continue to work after the repository is cloned on another machine.

---

## 8. Exercise Markdown

The exercise file should contain the reasoning and final answers, not only the answers.

Recommended structure:

```markdown
# CUDA Chapter X Exercises

## Exercise 1 — Short Title

Problem interpretation.

```math
...
```

Reasoning.

### Result

Short conclusion.

---

## Exercise 2 — Short Title

...
```

If an exercise contains a benchmark, keep the Markdown explanation concise and link the source file:

```markdown
Implementation:

[`corner_turning_benchmark.cu`](corner_turning_benchmark.cu)
```

The Markdown should explain:

- what was tested;
- why the experiment was useful;
- the main numerical result;
- the architectural interpretation.

---

## 9. Code Experiment README

When a chapter contains a substantial CUDA experiment, place a `README.md` next to the code.

Example:

```text
code/
├── coarse_matmul_benchmark.cu
└── README.md
```

Recommended experiment README structure:

```markdown
# Experiment Title

## 1. Purpose

What question is being tested?

---

## 2. Configuration

Kernel parameters and matrix sizes.

---

## 3. Results

Markdown table with measured values.

---

## 4. Analysis

Explain the measured behavior using CUDA concepts.

---

## 5. Conclusion

State the main performance lesson.
```

The experiment README should not reproduce the entire CUDA source code. Link to the `.cu` file instead.

---

## 10. What Should Not Be Uploaded

Do not commit generated executable or temporary build files.

Normally exclude:

```text
*.exe
*.obj
*.pdb
*.ilk
*.exp
*.lib
```

Upload source files:

```text
*.cu
*.cpp
*.h
*.md
*.tex
*.pdf
```

Before committing, run:

```bash
git status
```

and check the file list.

---

## 11. Update the Repository Root README

After finishing a chapter, update the repository-level `README.md`.

Example:

```markdown
- [x] [Chapter 5 — Memory Architecture and Data Locality](chapter05-memory-architecture/)
- [x] [Chapter 6 — Performance Considerations](chapter06-performance-considerations/)
- [ ] Chapter 7 — Next Topic
```

The chapter title should link to the chapter directory.

---

## 12. Standard Upload Workflow

### Step 1 — Finish the local folder

Check:

```text
chapterXX-topic/
├── README.md
├── notes/
├── exercises/
└── code/
```

Only keep files that belong to the chapter.

### Step 2 — Check Markdown locally

Check:

- headings;
- relative links;
- tables;
- code fences;
- `math` fences;
- spelling;
- filenames.

### Step 3 — Check Git status

```bash
git status
```

### Step 4 — Stage the chapter and root README

```bash
git add chapterXX-topic
git add README.md
```

### Step 5 — Check the staged files

```bash
git status
```

Confirm that no executable or temporary files are included.

### Step 6 — Commit

Example:

```bash
git commit -m "Add Chapter X topic notes and experiments"
```

### Step 7 — Push

```bash
git push
```

or:

```bash
git push origin main
```

### Step 8 — Verify directly on GitHub

After pushing, open the repository in the browser and check:

1. the chapter root `README.md` renders automatically;
2. all relative links work;
3. mathematical equations render;
4. Markdown tables render correctly;
5. code blocks have syntax highlighting;
6. PDF and LaTeX links open correctly;
7. no `.exe` or temporary build files were uploaded.

Do not treat the chapter as finished until the GitHub-rendered version has been checked.

---

## 13. Final Chapter Checklist

Before marking a chapter as complete:

- [ ] Chapter directory has a clear name.
- [ ] Chapter root `README.md` exists.
- [ ] `notes/` contains the final `.tex` and `.pdf`.
- [ ] `exercises/` contains the exercise summary.
- [ ] Relevant exercise code is included.
- [ ] `code/` contains the main implementation or benchmark when applicable.
- [ ] Experiment README contains results and analysis.
- [ ] Markdown uses GitHub-compatible syntax.
- [ ] Display equations use fenced `math` blocks.
- [ ] Code blocks specify a language.
- [ ] Tables render correctly.
- [ ] All links use relative paths.
- [ ] No executable or temporary files are committed.
- [ ] Repository root README is updated.
- [ ] `git status` is clean after the commit.
- [ ] The final GitHub page has been visually checked.

---

## 14. Recommended Principle

Keep the repository useful as a technical study record rather than trying to archive every intermediate file.

For each chapter, preserve three things:

```text
conceptual understanding
        +
correct implementation
        +
measured evidence
```

A concise chapter with clear notes, verified code, and one meaningful experiment is more useful than a large folder containing many redundant files.
