# PyCUDA Compiler: Technical Report

## 1. Introduction

Developing high-performance GPU applications typically involves writing CUDA C++, a process that offers control but comes with significant complexity and boilerplate. While Python ecosystem tools like Numba, CuPy, and Taichi have made strides in accessibility, they often still require developers to understand low-level GPU concepts or use specialized domain-specific languages.

PyCUDA Compiler addresses this challenge by providing **seamless Python-to-CUDA compilation**. It allows developers to write standard, statically-typed Python code that is automatically compiled into optimized CUDA kernels. Key differentiators include automatic parallelization of native loops and first-class support for high-level data structures—such as `List`, `Dict`, and `Set`—directly within GPU kernels.

## 2. Architecture Overview

The compiler follows a traditional multi-pass architecture:

1.  **Parsing (`parser.py`)**: Converts Python source code into a custom Intermediate Representation (IR). It uses Python's built-in `ast` module to traverse the code structure.
2.  **Type Inference (`type_inference.py`)**: Analyzes the IR to assign types to all variables and expressions. It resolves types for function arguments, local variables, and return values, ensuring type safety before C++ generation.
3.  **Optimization**: Includes automatic parallelization and dead code analysis.
4.  **Code Generation (`codegen.py`)**: Translates the typed IR into optimizing CUDA C++ code. It handles C++ struct definitions, memory access patterns, and intrinsic mapping.
5.  **Compilation & Execution (`compiler.py`)**: Uses `cupy.RawModule` or NVRTC to compile the generated C++ code into a binary kernel and launches it on the GPU.

## 3. Automatic Parallelization

One of the compiler's most powerful features is identifying and parallelizing loops.

### Detection
The detection logic resides in `parser.py`. When parsing a `for` loop, the compiler marks it for parallelization if:
*   It is a top-level loop within a `@kernel` function.
*   It iterates over a `range()`.
*   The range is bound by `len(array)` (or explicit size).

### Transformation
When a loop is marked parallel, it is converted into an `IRParallelFor` node.
The code generator (`codegen.py`) translates this into a **Grid-Stride Loop**, a robust CUDA pattern that handles arrays of any size regardless of grid dimensions.

**Python Source:**
```python
for i in range(len(data)):
    data[i] = data[i] * 2.0
```

**Generated CUDA C++:**
```cpp
// THREAD_ID = blockIdx.x * blockDim.x + threadIdx.x
// GRID_STRIDE = blockDim.x * gridDim.x

for (int i = THREAD_ID; i < _size_data; i += GRID_STRIDE) {
    data[i] = data[i] * 2.0f;
}
```

## 4. Code Generation & Type System

### Data Structures
Python classes and data structures are mapped directly to C++ structs.

*   **`List[T]`**: Implemented as a clean struct `{ T* data; int* size; int capacity; }`.
    *   Supports `append(val)`, `pop()`, `len()`, and random access `operator[]`.
    *   **Note**: Requires pre-allocated memory currently.
*   **`Dict[K, V]`**: Experimental implementation using open-addressing hash table within the kernel.
    *   Supports `get()`, `put()`, `contains()`.
    *   Collision resolution: Linear probing.
*   **`Set[T]`**: Similar to Dict, but stores only keys.
*   **`Optional[T]`**: Mapped to `{ bool has_value; T value; }`.
    *   Efficient for sparse data or error handling within kernels.

### Intrinsics
Standard Python math functions are remapped to their CUDA float-precision equivalents (e.g., `math.sin` -> `sinf`, `math.pow` -> `powf`) to avoid double-precision performance penalties on consumer GPUs.

## 5. Comparison with Existing Tools

| Feature | PyCUDA Compiler | Numba (CUDA) | CuPy (Elementwise) | Taichi |
| :--- | :--- | :--- | :--- | :--- |
| **Syntax** | Pure Python + Hints | Python subset | String / limited Python | Python DSL |
| **Auto-Parallel** | **Yes (loops)** | No (explicit thread idx) | Yes (elementwise) | Yes (struct-for) |
| **Dynamic Structures** | **List, Dict, Set** | No | No | No |
| **Type System** | Static (Type Hints) | Infer / Sig | Templated C++ | Static infer |
| **Learning Curve** | Low | Medium | Medium (C++ knowledge) | Medium |

**Key Differentiator**: PyCUDA Compiler allows using `List` and `Dict` inside kernels, enabling algorithms that are hard to express in pure array-based frameworks (e.g., graph traversal, sparse accumulation, filtering).

## 6. Results / Benchmarks

Tests were performed on various NVIDIA GPUs.

### Speedup vs CPU (NVIDIA RTX 3080)
| Operation | Speedup | Note |
| :--- | :--- | :--- |
| Vector Add | **14x** | Memory bandwidth bound |
| Math Ops (sin/exp) | **94x** | Compute bound, highly scalable |
| Conditional Logic | **121x** | efficient branch handling |
| Complex Calc | **280x** | Massive parallelism win |

### Performance Scaling
*   **Break-even point**: ~100k elements. Below this, PCI-E transfer overhead dominates.
*   **Optimal range**: >1M elements.
*   **A100 Performance**: Up to **747x** speedup on complex compute kernels.

## 7. Limitations

1.  **Recursion**: Not supported (CUDA limitation).
2.  **Dynamic Memory Allocation**: `List` and `Dict` currently rely on pre-allocated buffers. `malloc` inside kernels is not used to ensure stability and performance.
3.  **Python Standard Library**: Only `math` module and basic types are supported. You cannot import generic Python modules (os, sys, json) inside a kernel.
4.  **Dictionary Performance**: The current linear-probing `Dict` implementation may suffer from collisions at high load factors (>70%).
5.  **Classes**: Class methods are supported, but inheritance/polymorphism is not yet implemented.

## 8. Future Work
*   Shared memory support (`__shared__`) via Python context managers.
*   More aggressive loop fusion and unrolling.
*   Support for multi-file projects and linking.
*   Improved `Dict` implementation (Cuckoo hashing or Robin Hood hashing).
