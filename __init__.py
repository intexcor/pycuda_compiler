"""
PyCUDA Compiler - Compile Python to CUDA

A compiler that transforms Python code into CUDA kernels,
allowing you to run arbitrary Python-like code on NVIDIA GPUs.

Quick Start:
    
    from pycuda_compiler import cuda_compile, Array, float32
    
    @cuda_compile
    def my_kernel(data: Array[float32]):
        for i in range(len(data)):
            if data[i] > 0:
                data[i] = sqrt(data[i])
            else:
                data[i] = data[i] * data[i]
    
    # Create GPU array and run
    data = my_kernel.array([4.0, -2.0, 9.0, -3.0])
    my_kernel(data)
    print(my_kernel.to_numpy(data))  # [2.0, 4.0, 3.0, 9.0]

Features:
    - Automatic parallelization of for loops
    - Classes → CUDA structs
    - Methods → device functions
    - Type inference
    - Automatic grid/block sizing
    - Math functions (sqrt, sin, cos, etc.)

Supported Python Constructs:
    - Functions with @kernel decorator
    - Classes (become CUDA structs)
    - for loops (auto-parallelized in kernels)
    - while loops
    - if/elif/else
    - Arithmetic operations (+, -, *, /, //, %, **)
    - Comparisons (==, !=, <, <=, >, >=)
    - Boolean operations (and, or, not)
    - Bitwise operations (&, |, ^, ~, <<, >>)
    - Ternary operator (x if cond else y)
    - Array indexing (arr[i])
    - Attribute access (obj.field)
    - Method calls (obj.method())
    - Math functions (sqrt, sin, cos, exp, log, etc.)

Limitations:
    - No dynamic memory allocation
    - No recursion (limited support)
    - No strings
    - No exceptions
    - No imports inside kernels
    - No closures

Requirements:
    - NVIDIA GPU with CUDA support
    - CuPy (pip install cupy-cuda12x)
    - NumPy
"""

from .compiler import (
    # Main classes
    CUDAProgram,
    CUDAKernel,
    CompileError,
    
    # Decorators
    cuda_compile,
    kernel,
    
    # Functions
    compile_module,
    
    # Type hints
    Array,
    Tensor,
    float32,
    float64,
    int32,
    int64,
)

from .types_ir import (
    CUDAType,
    TypeKind,
    VOID, BOOL,
    INT8, INT16, INT32, INT64,
    UINT8, UINT16, UINT32, UINT64,
    FLOAT16, FLOAT32, FLOAT64,
)

from .runtime import (
    CUDARuntime,
    CUDAArray,
    CUDAStruct,
    CUDATimer,
    GPUMemoryManager,
    auto_launch_config,
    auto_launch_config_2d,
)

__version__ = '0.1.0'
__author__ = 'PyCUDA Compiler'

__all__ = [
    # Main API
    'CUDAProgram',
    'CUDAKernel',
    'cuda_compile',
    'kernel',
    'compile_module',
    'CompileError',
    
    # Type hints
    'Array',
    'Tensor',
    'float32',
    'float64', 
    'int32',
    'int64',
    
    # Types
    'CUDAType',
    'TypeKind',
    'VOID', 'BOOL',
    'INT8', 'INT16', 'INT32', 'INT64',
    'UINT8', 'UINT16', 'UINT32', 'UINT64',
    'FLOAT16', 'FLOAT32', 'FLOAT64',
    
    # Runtime
    'CUDARuntime',
    'CUDAArray',
    'CUDAStruct',
    'CUDATimer',
    'GPUMemoryManager',
    'auto_launch_config',
    'auto_launch_config_2d',
]
