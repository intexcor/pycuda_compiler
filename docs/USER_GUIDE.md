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
cd python cli.py input.py -o kernel.cu
```

### Компиляция с NVCC

Сгенерированный `.cu` файл содержит только ядра (kernels). Чтобы использовать их:

**1. Компиляция в PTX (для загрузки в Python):**
```bash
nvcc -ptx kernel.cu -o kernel.ptx
```
Затем можно загрузить через `cupy.RawModule(filename='kernel.ptx')`.

**2. Компиляция в Shared Library (для C++):**
```bash
nvcc -shared -Xcompiler -fPIC kernel.cu -o kernel.so
```

### Как использовать kernel.ptx (Python)

Скомпилированный PTX файл можно загрузить и исполнить в любом Python скрипте с помощью `cupy.RawModule`, даже без установленного PyCUDA Compiler.

```python
import cupy as cp

# Загрузка модуля
module = cp.RawModule(path='kernel.ptx')
kernel = module.get_function('square_root_kernel')

# Подготовка данных
data = cp.array([1.0, 4.0, 9.0, 16.0], dtype=cp.float32)
result = cp.zeros_like(data)

# Аргументы для C++ ядра (pointers и размеры)
# void square_root_kernel(float* data, int size_data, float* result, int size_result)
args = (
    data, 
    cp.int32(data.size), 
    result, 
    cp.int32(result.size)
)

# Запуск
kernel((1,), (256,), args)
print(result)
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
