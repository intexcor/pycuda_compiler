# PyCUDA Compiler User Guide

PyCUDA Compiler is a high-level Python-to-CUDA compiler that allows you to write CUDA kernels using standard Python syntax and type hints. It automates the complexity of writing C++ CUDA code, handling type inference, memory management, and parallelization for you.

## Features

*   **Python Syntax**: Write kernels as standard Python functions decorated with `@cuda_compile`.
*   **Type Statically Typed**: Uses Python type hints (`Array`, `float32`, `int32`, `List`, `Dict`, `Set`) for robust code generation.
*   **Automatic Parallelization**: Automatically converts top-level loops over arrays into highly optimized CUDA Grid-Stride loops.
*   **Rich Standard Library**: Supports common math functions (`sin`, `cos`, `sqrt`, etc.), `print` debugging, and complex data structures.
*   **Seamless Integration**: Works directly with `cupy` arrays.

## Installation

Requirements:
*   Python 3.8+
*   NVIDIA GPU with CUDA support
*   `cupy` (for array management and kernel launching)

```bash
pip install cupy-cuda12x  # Adjust for your CUDA version
# Clone this repository
git clone https://github.com/yourusername/pycuda_compiler.git
cd pycuda_compiler
```

## Quick Start

Here is a simple example that computes the square root of an array of numbers.

```python
import cupy as cp
from compiler import cuda_compile, Array, float32
from math import sqrt

@cuda_compile
def square_root_kernel(data: Array[float32], result: Array[float32]):
    # Automatic parallelization handles the loop distributions!
    for i in range(len(data)):
        result[i] = sqrt(data[i])

# Create data on GPU
data = cp.array([1.0, 4.0, 9.0, 16.0], dtype=cp.float32)
result = cp.zeros_like(data)

# Run kernel
square_root_kernel(data, result)

print(result) # [1. 2. 3. 4.]
```

## Advanced Features

### Tensor & Structured Types

PyCUDA Compiler supports multi-dimensional explicit Tensors and structured types (Classes).

```python
from pycuda_compiler import cuda_compile, Array, Tensor, float32

class Particle:
    x: float32
    y: float32
    vx: float32
    vy: float32

@cuda_compile
def update_particles(particles: Array[Particle], dt: float32):
    for i in range(len(particles)):
        p = particles[i]
        p.x += p.vx * dt
        p.y += p.vy * dt
        particles[i] = p
```

### Dynamic Data Structures

We support experimental dynamic data structures like `List`, `Dict`, and `Set` directly in CUDA kernels.

```python
from pycuda_compiler import cuda_compile, List, int32

@cuda_compile
def filter_positive(input: List[int32], output: List[int32]):
    for i in range(input.len()):
        val = input[i]
        if val > 0:
            output.append(val)
```
