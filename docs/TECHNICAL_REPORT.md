# PyCUDA Compiler: Technical Report

This document details the internal architecture and implementation of the PyCUDA Compiler.

## 1. Architecture Overview

The compiler follows a traditional multi-pass architecture:

1.  **Parsing (`parser.py`)**: Converts Python source code into a custom Intermediate Representation (IR). It uses Python's built-in `ast` module to traverse the code structure.
2.  **Type Inference (`type_inference.py`)**: Analyzes the IR to assign types to all variables and expressions. It resolves types for function arguments, local variables, and return values, ensuring type safety before C++ generation.
3.  **Optimization**: Includes automatic parallelization and dead code analysis (partially integrated into parsing and inference).
4.  **Code Generation (`codegen.py`)**: Translates the typed IR into optimizing CUDA C++ code. It handles C++ struct definitions, memory access patterns, and intrinsic mapping.
5.  **Compilation & Execution (`compiler.py`)**: Uses `cupy.RawModule` or NVRTC to compile the generated C++ code into a binary kernel and launches it on the GPU.

## 2. Automatic Parallelization

One of the compiler's most powerful features is identifying and parallelizing loops.

### Detection
The detection logic resides in `parser.py`. When parsing a `for` loop, the compiler checks:
*   Is this a top-level loop within a `@kernel` function?
*   Is it iterating over a `range()`?
*   Is the range bound by `len(array)`?

If these conditions are met, the loop is tagged for parallelization.

### Transformation
When a loop is marked parallel, it is converted into an `IRParallelFor` node instead of a standard `IRFor`.

The code generator (`codegen.py`) translates `IRParallelFor` into a **Grid-Stride Loop**. This is a robust CUDA pattern that allows a kernel to handle arrays of any size, regardless of the grid dimensions.

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

This ensures efficient memory coalescing and load balancing across GPU threads.

## 3. Code Generation & Type System

### Data Structures
Python classes and data structures are mapped directly to C++ structs.

*   `List[T]`: Mapped to a struct containing a data pointer, size pointer, and capacity. Supports `append()`, `pop()`, and indexing.
*   `Optional[T]`: Mapped to a struct with a `bool has_value` flag and the value storage.
*   `Dict[K, V]`: Implemented as a hash map with open addressing within the kernel (experimental).

### Intrinsics
Standard Python math functions are remapped to their CUDA float-precision equivalents (e.g., `math.sin` -> `sinf`, `math.pow` -> `powf`) to avoid double-precision performance penalties on consumer GPUs.

## 4. Future Work
*   Shared memory support (`__shared__`) via Python context managers.
*   More aggressive loop fusion and unrolling.
*   Support for multi-file projects and linking.
