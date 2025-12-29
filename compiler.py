"""
PyCUDA Compiler - Main API

Высокоуровневый API для компиляции и запуска Python кода на CUDA.

Пример использования:
    
    from pycuda_compiler import cuda_compile, CUDAProgram
    
    @cuda_compile
    def my_kernel(data: Array[float32]):
        for i in range(len(data)):
            if data[i] > 0:
                data[i] = sqrt(data[i])
            else:
                data[i] = 0.0
    
    # Запуск
    data = my_kernel.array([1.0, 4.0, 9.0, -1.0])
    my_kernel(data)
    print(data)  # [1.0, 2.0, 3.0, 0.0]
"""

from __future__ import annotations
import inspect
import textwrap
from typing import Dict, List as PyList, Optional, Callable, Any, Tuple

from types_ir import (
    IRModule, IRFunctionDef
)
from parser import PythonParser
from type_inference import TypeInference
from codegen import CUDACodeGen

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


# ============================================================================
# Type aliases for annotations
# ============================================================================

class Array:
    """Type hint для GPU массивов."""
    def __class_getitem__(cls, item):
        return f'Array[{item}]'

class List:
    """Type hint для GPU списков."""
    def __class_getitem__(cls, item):
        return f'List[{item}]'

class Tensor:
    """Type hint для GPU тензоров."""
    def __class_getitem__(cls, item):
        return f'Tensor[{item}]'

# Type aliases
float32 = 'float32'
float64 = 'float64'
int32 = 'int32'
int64 = 'int64'


# ============================================================================
# Main Compiler Class
# ============================================================================

class CUDAProgram:
    """
    Компилирует Python код в CUDA и предоставляет API для запуска.
    """
    
    def __init__(self, source: str, debug: bool = False):
        """
        Args:
            source: Python исходный код
            debug: показывать отладочную информацию
        """
        self.source = textwrap.dedent(source)
        self.debug = debug
        
        # Компиляция
        self._ir: Optional[IRModule] = None
        self._cuda_code: Optional[str] = None
        self._kernels: Dict[str, Any] = {}
        self._kernel_info: Dict[str, IRFunctionDef] = {}
        self.functions: PyList[IRFunctionDef] = []
        
        self._compile()
    
    def _compile(self):
        """Компилирует исходный код."""
        # 1. Parse Python → IR
        parser = PythonParser()
        self._ir = parser.parse(self.source)
        
        if self.debug:
            print("=== Parsed IR ===")
            self._print_ir()
        
        # 2. Type inference
        type_inf = TypeInference()
        self._ir = type_inf.infer(self._ir)
        
        if self.debug:
            print("\n=== After Type Inference ===")
            self._print_ir()
        
        # 3. Generate CUDA code
        codegen = CUDACodeGen()
        self._cuda_code = codegen.generate(self._ir)
        
        # 4. Собираем информацию о kernels
        for func in self._ir.functions:
            if func.is_kernel:
                self._kernel_info[func.name] = func
    
    def _print_ir(self):
        """Выводит IR для отладки."""
        for struct in self._ir.structs:
            print(f"Struct: {struct.name}")
            for name, t in struct.fields:
                print(f"  {name}: {t.name}")
            for method in struct.methods:
                print(f"  method {method.name}()")
        
        for func in self._ir.functions:
            kind = "kernel" if func.is_kernel else "device"
            print(f"Function [{kind}]: {func.name}")
            for name, t in func.params:
                print(f"  param {name}: {t.name if t else '?'}")
            for name, t in func.local_vars.items():
                print(f"  local {name}: {t.name if t else '?'}")
            print(f"  returns: {func.return_type.name if func.return_type else 'void'}")
    
    @property
    def cuda_code(self) -> str:
        """Возвращает сгенерированный CUDA код."""
        return self._cuda_code
    
    def show_cuda(self):
        """Выводит сгенерированный CUDA код."""
        print(self._cuda_code)
    
    def get_kernel(self, name: str):
        """Компилирует и возвращает kernel."""
        if not HAS_CUPY:
            raise ImportError("CuPy required. Install: pip install cupy-cuda12x")
        
        if name not in self._kernels:
            if name not in self._kernel_info:
                raise ValueError(f"Kernel '{name}' not found. Available: {list(self._kernel_info.keys())}")
            
            # Компилируем CUDA код
            # Пробуем несколько подходов для совместимости
            original_code = self._cuda_code
            
            # Подготовка кода с extern "C" для RawModule
            lines = original_code.split('\n')
            header_end = None
            
            # Находим первую __global__ функцию (kernel)
            for i, line in enumerate(lines):
                stripped = line.strip()
                if '__global__' in stripped:
                    header_end = i
                    break
            
            # Если нашли kernel функции, разделяем код и оборачиваем функции в extern "C"
            wrapped_code = original_code
            if header_end is not None and header_end > 0:
                header = '\n'.join(lines[:header_end])
                functions = '\n'.join(lines[header_end:])
                
                # Оборачиваем функции в extern "C" блок
                wrapped_code = f'''{header}

extern "C" {{
{functions}
}}'''
            
            try:
                # Используем RawModule с extern "C" - это более надежный подход
                module = cp.RawModule(code=wrapped_code, backend='nvrtc', options=('--std=c++14',))
                
                # Пытаемся получить функцию
                try:
                    self._kernels[name] = module.get_function(name)
                except Exception as func_error:
                    # Если функция не найдена, пытаемся получить больше информации
                    error_msg = str(func_error)
                    
                    # Пытаемся получить PTX чтобы увидеть какие символы есть
                    ptx_info = ""
                    try:
                        if hasattr(module, 'get_ptx'):
                            ptx = module.get_ptx()
                            ptx_str = ptx.decode() if isinstance(ptx, bytes) else str(ptx)
                            
                            # Ищем все .visible .entry функции в PTX
                            import re as re2
                            entries = re2.findall(r'\.visible\s+\.entry\s+(\w+)', ptx_str)
                            if entries:
                                ptx_info = f"\nAvailable kernel symbols in PTX: {entries}"
                            else:
                                # Попробуем найти любые entry точки
                                all_entries = re2.findall(r'\.entry\s+(\w+)', ptx_str)
                                if all_entries:
                                    ptx_info = f"\nAll entry points in PTX: {all_entries}"
                                else:
                                    # Покажем начало PTX для отладки
                                    ptx_lines = ptx_str.split('\n')[:50]
                                    ptx_info = "\nFirst 50 lines of PTX:\n" + '\n'.join(ptx_lines)
                    except Exception as ptx_err:
                        ptx_info = f"\nCould not extract PTX: {ptx_err}"
                    
                    print("=== CUDA Code ===")
                    print(wrapped_code)
                    print(f"\n=== Trying to find function: {name} ===")
                    print(ptx_info)
                    raise CompileError(f"Function '{name}' not found in compiled module: {error_msg}{ptx_info}")
                    
            except cp.cuda.compiler.CompileException as e:
                # Ошибка компиляции
                error_msg = str(e)
                if hasattr(e, 'log'):
                    error_msg += f"\nCompilation log:\n{e.log}"
                print("=== CUDA Code ===")
                print(wrapped_code)
                raise CompileError(f"CUDA compilation failed: {error_msg}")
            except Exception as e:
                # Другие ошибки
                error_msg = str(e)
                print("=== CUDA Code ===")
                print(wrapped_code)
                raise CompileError(f"CUDA compilation failed: {error_msg}")
        
        return self._kernels[name]
    
    def run(
        self,
        kernel_name: str,
        *args,
        grid: Optional[Tuple[int, ...]] = None,
        block: Optional[Tuple[int, ...]] = None
    ):
        """
        Запускает kernel.
        
        Args:
            kernel_name: имя kernel функции
            *args: аргументы (CuPy arrays)
            grid: размер grid (автоматически если None)
            block: размер block (по умолчанию 256)
        """
        kernel = self.get_kernel(kernel_name)
        kernel_info = self._kernel_info[kernel_name]
        
        # Собираем аргументы с размерами массивов
        cuda_args = []
        size = None
        
        # Проверяем соответствие аргументов
        params = kernel_info.params
        
        # Post-processing tasks (write-back for numpy arrays)
        write_back_tasks = []
        
        # Args loop
        for i, arg in enumerate(args):
            # Check for NumPy array -> convert to CuPy
            is_numpy = HAS_NUMPY and np is not None and isinstance(arg, np.ndarray)
            if is_numpy:
                # Creates a copy on device
                # Note: cp.asarray(arg) might be more efficient if already on device? 
                # But here it is definitely host numpy.
                d_arg = cp.array(arg)
                
                # We will use this device array for the kernel
                # And we need to copy it back later to support in-place modifications
                write_back_tasks.append((d_arg, arg))
                
                # Replace arg with device arg for subsequent processing
                arg = d_arg
            
            if i < len(params):
                param_name, param_type = params[i]
            else:
                param_type = None

            # Tensor support
            if param_type and param_type.is_tensor():
                 if not isinstance(arg, cp.ndarray):
                     raise TypeError(f"Argument '{param_name}' must be a Cupy array (or Numpy array) (expected Tensor)")
                 
                 cuda_args.append(arg) # ptr
                 # Push shapes and strides
                 rank = param_type.rank
                 if arg.ndim != rank:
                      pass
                 
                 for d in range(rank):
                     if d < arg.ndim:
                         cuda_args.append(arg.shape[d])
                         cuda_args.append(arg.strides[d] // arg.itemsize)
                     else:
                         cuda_args.append(1)
                         cuda_args.append(0)
                 
                 # Use first dimension for default grid size if not set
                 if size is None and rank > 0:
                     size = arg.shape[0]

            # List support
            elif param_type and param_type.is_list():
                # arg should be a tuple (buffer, size_ptr) or just buffer (ifsize=0)?
                # Actually, the user should pass a pre-allocated buffer for the list content.
                # And we need to manage the size.
                # Ideally, we pass a special List object from python side.
                # But for now, let's accept a tuple: (buffer_array, size_array)
                # buffer_array: where elements are stored. capacity = len(buffer_array)
                # size_array: 1-element int32 array storing current size (indices count).
                
                if isinstance(arg, tuple) and len(arg) == 2:
                    buf, sz = arg
                else:
                    # Provide defaults? Or expect user to pass (buffer, size_arr)
                    # Let's verify types
                    raise ValueError(f"Argument '{param_name}' (List) must be a tuple (buffer_array, size_array)")
                
                if not isinstance(buf, cp.ndarray) or not isinstance(sz, cp.ndarray):
                    raise TypeError("List argument components must be cupy arrays")
                
                # Check types?
                # We need to construct the C++ struct on the fly? No, the kernel signature expects struct by value?
                # Actually, in CUDA C++, structs are passed by value.
                # We need to pass the fields of the struct as separate kernel arguments?
                # Or pass a single pointer to a struct in global memory?
                # Easier: CodeGen generates struct. Arg is struct by value.
                # But struct by value in kernel args?
                # "Kernel arguments are passed by value". 
                # If the struct contains pointers, we pass the struct (which is small) by value.
                # So we need to pass: buf_ptr, size_ptr, capacity (int).
                
                # Let's check how CodeGen generates the function signature for List.
                # Currently it calls `type_to_cuda`, which for struct returns name.
                # But we haven't updated CodeGen yet! CodeGen needs to expand List args.
                # Wait, if `type_to_cuda` returns a struct name, then the kernel expects that struct.
                # In PyCUDA/CuPy call, we can't pass a raw C struct easily unless we pack it into bytes.
                # Alternatively, we can assume CodeGen expands it to (T* data, int* size, int capacity).
                # Expanding is easier for Python interoperability.
                # Let's assume CodeGen WILL expand List arguments.
                
                cuda_args.append(buf) # data ptr
                cuda_args.append(sz)  # size ptr
                cuda_args.append(cp.int32(buf.size)) # capacity
                
            # Support for manual size override: (array, logical_size)
            elif isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], cp.ndarray):
                arr, logical_size = arg
                cuda_args.append(arr)
                cuda_args.append(logical_size)
                if size is None:
                    size = logical_size
            elif isinstance(arg, cp.ndarray):
                cuda_args.append(arg)
                cuda_args.append(arg.size)  # размер массива
                if size is None:
                    size = arg.size
            else:
                cuda_args.append(arg)
        
        if size is None:
            size = 1
        
        # Автоматический расчёт grid/block
        if block is None:
            block = (256,)
        if grid is None:
            grid = ((size + block[0] - 1) // block[0],)
        
        # Запуск
        kernel(grid, block, tuple(cuda_args))
        
        # Write back changes to NumPy arrays (synchronize first)
        if write_back_tasks:
            cp.cuda.Stream.null.synchronize()
            for d_arr, h_arr in write_back_tasks:
                # In-place update of host array
                # h_arr[:] = d_arr.get()  # .get() returns numpy array
                try:
                    # Optimize: use copyto?
                    # np.copyto(h_arr, d_arr.get())
                    # or element-wise assign
                    h_arr[:] = d_arr.get()
                except Exception as e:
                    print(f"Warning: Failed to copy back NumPy array argument: {e}")

    
    # ========== Convenience methods ==========
    
    def array(self, data, dtype=None) -> 'cp.ndarray':
        """Создаёт GPU массив."""
        if not HAS_CUPY:
            raise ImportError("CuPy required")
        
        if dtype is None:
            dtype = cp.float32
        
        if isinstance(data, cp.ndarray):
            return data.astype(dtype)
        return cp.array(data, dtype=dtype)
    
    def zeros(self, shape, dtype=None) -> 'cp.ndarray':
        """Создаёт нулевой GPU массив."""
        if not HAS_CUPY:
            raise ImportError("CuPy required")
        return cp.zeros(shape, dtype=dtype or cp.float32)
    
    def ones(self, shape, dtype=None) -> 'cp.ndarray':
        """Создаёт единичный GPU массив."""
        if not HAS_CUPY:
            raise ImportError("CuPy required")
        return cp.ones(shape, dtype=dtype or cp.float32)
    
    def to_numpy(self, arr: 'cp.ndarray') -> 'np.ndarray':
        """Копирует GPU массив на CPU."""
        return cp.asnumpy(arr)


class CompileError(Exception):
    """Ошибка компиляции."""
    pass


# ============================================================================
# Decorator API
# ============================================================================

class CUDAKernel:
    """Обёртка над скомпилированным kernel."""
    
    def __init__(self, func: Callable, debug: bool = False):
        self._func = func
        self._name = func.__name__
        self._debug = debug
        
        # Получаем исходный код функции
        source = inspect.getsource(func)
        # Убираем декоратор
        lines = source.split('\n')
        filtered = []
        skip_decorator = True
        for line in lines:
            stripped = line.strip()
            if skip_decorator:
                if stripped.startswith('@'):
                    continue
                skip_decorator = False
            filtered.append(line)
        
        # Добавляем @kernel декоратор для нашего парсера
        source = '@kernel\n' + textwrap.dedent('\n'.join(filtered))
        
        # Компилируем
        self._program = CUDAProgram(source, debug=debug)
        
        # Копируем docstring
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__
    
    def __call__(self, *args, grid=None, block=None):
        """Запускает kernel."""
        self._program.run(self._name, *args, grid=grid, block=block)
    
    @property
    def cuda_code(self) -> str:
        """Возвращает CUDA код."""
        return self._program.cuda_code
    
    def show_cuda(self):
        """Показывает CUDA код."""
        self._program.show_cuda()
    
    def array(self, data, dtype=None):
        """Создаёт GPU массив."""
        return self._program.array(data, dtype)
    
    def zeros(self, shape, dtype=None):
        return self._program.zeros(shape, dtype)
    
    def ones(self, shape, dtype=None):
        return self._program.ones(shape, dtype)
    
    def to_numpy(self, arr):
        return self._program.to_numpy(arr)


def cuda_compile(func: Callable = None, *, debug: bool = False):
    """
    Декоратор для компиляции Python функции в CUDA kernel.
    
    Пример:
        @cuda_compile
        def process(data: Array[float32]):
            for i in range(len(data)):
                data[i] = sqrt(data[i])
        
        data = process.array([1.0, 4.0, 9.0])
        process(data)
        print(process.to_numpy(data))  # [1.0, 2.0, 3.0]
    """
    def decorator(f):
        return CUDAKernel(f, debug=debug)
    
    if func is not None:
        return decorator(func)
    return decorator


def kernel(func: Callable):
    """Маркер для kernel функций (используется внутри CUDAProgram)."""
    func._is_kernel = True
    return func


# ============================================================================
# Multi-kernel programs
# ============================================================================

def compile_module(source: str, debug: bool = False) -> CUDAProgram:
    """
    Компилирует модуль с несколькими kernels и функциями.
    
    Пример:
        program = compile_module('''
        class Particle:
            x: float32
            y: float32
            vx: float32
            vy: float32
        
        @kernel
        def update_positions(particles: Array[Particle], dt: float32):
            for i in range(len(particles)):
                particles[i].x += particles[i].vx * dt
                particles[i].y += particles[i].vy * dt
        
        @kernel
        def compute_forces(particles: Array[Particle], forces: Array[float32]):
            for i in range(len(particles)):
                # ... compute forces ...
                pass
        ''')
        
        particles = program.zeros(1000, dtype=Particle)
        program.run('update_positions', particles, 0.01)
    """
    return CUDAProgram(source, debug=debug)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main classes
    'CUDAProgram',
    'CUDAKernel',
    'CompileError',
    
    # Decorators
    'cuda_compile',
    'kernel',
    
    # Functions
    'compile_module',
    
    # Type hints
    'Array',
    'Tensor',
    'float32',
    'float64',
    'int32',
    'int64',
]

