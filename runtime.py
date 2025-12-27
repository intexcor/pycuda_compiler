"""
PyCUDA Compiler - Runtime

Компилирует и запускает CUDA код через CuPy.
"""

from __future__ import annotations
import hashlib
from typing import Dict, Optional, Tuple, Any, Union

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


class CUDARuntime:
    """Runtime для компиляции и запуска CUDA кода."""
    
    def __init__(self):
        if not HAS_CUPY:
            raise ImportError("CuPy is required. Install with: pip install cupy-cuda12x")
        
        self._kernel_cache: Dict[str, Any] = {}
        self._module_cache: Dict[str, Any] = {}
    
    def compile_kernel(self, cuda_code: str, kernel_name: str) -> Any:
        """Компилирует CUDA код и возвращает kernel."""
        # Кеш по хешу кода
        code_hash = hashlib.md5(cuda_code.encode()).hexdigest()
        cache_key = f"{code_hash}_{kernel_name}"
        
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]
        
        # Компиляция
        try:
            # Обёртываем в extern "C" для nvrtc
            wrapped_code = f'extern "C" {{\n{cuda_code}\n}}'
            
            module = cp.RawModule(
                code=wrapped_code,
                backend='nvrtc',
                options=('--std=c++14',)
            )
            kernel = module.get_function(kernel_name)
            
            self._kernel_cache[cache_key] = kernel
            return kernel
        
        except cp.cuda.compiler.CompileException as e:
            raise CompileError(f"CUDA compilation failed:\n{e}")
    
    def launch_kernel(
        self,
        kernel: Any,
        args: Tuple[Any, ...],
        grid: Optional[Tuple[int, ...]] = None,
        block: Optional[Tuple[int, ...]] = None
    ):
        """Запускает kernel."""
        # Определяем размеры по первому массиву
        size = None
        for arg in args:
            if isinstance(arg, cp.ndarray):
                size = arg.size
                break
        
        if size is None:
            size = 1
        
        # Автоматический расчёт grid/block
        if block is None:
            block = (256,)
        if grid is None:
            grid = ((size + block[0] - 1) // block[0],)
        
        kernel(grid, block, args)
    
    def to_gpu(self, data: Any) -> cp.ndarray:
        """Переносит данные на GPU."""
        if isinstance(data, cp.ndarray):
            return data
        if HAS_NUMPY and isinstance(data, np.ndarray):
            return cp.asarray(data)
        if isinstance(data, (list, tuple)):
            return cp.array(data, dtype=cp.float32)
        return cp.array([data], dtype=cp.float32)
    
    def to_cpu(self, data: Any) -> np.ndarray:
        """Переносит данные на CPU."""
        if isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        if HAS_NUMPY and isinstance(data, np.ndarray):
            return data
        return np.array(data)
    
    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype=cp.float32) -> cp.ndarray:
        """Создаёт нулевой массив на GPU."""
        return cp.zeros(shape, dtype=dtype)
    
    def ones(self, shape: Union[int, Tuple[int, ...]], dtype=cp.float32) -> cp.ndarray:
        """Создаёт единичный массив на GPU."""
        return cp.ones(shape, dtype=dtype)
    
    def empty(self, shape: Union[int, Tuple[int, ...]], dtype=cp.float32) -> cp.ndarray:
        """Создаёт пустой массив на GPU."""
        return cp.empty(shape, dtype=dtype)
    
    def synchronize(self):
        """Синхронизирует GPU."""
        cp.cuda.Stream.null.synchronize()


class CompileError(Exception):
    """Ошибка компиляции CUDA."""
    pass


# ============================================================================
# Высокоуровневый API
# ============================================================================

class CUDAArray:
    """Обёртка над GPU массивом с удобным API."""
    
    def __init__(self, data=None, shape=None, dtype=cp.float32):
        if data is not None:
            if isinstance(data, cp.ndarray):
                self._data = data
            elif HAS_NUMPY and isinstance(data, np.ndarray):
                self._data = cp.asarray(data)
            else:
                self._data = cp.array(data, dtype=dtype)
        elif shape is not None:
            self._data = cp.zeros(shape, dtype=dtype)
        else:
            self._data = cp.array([], dtype=dtype)
    
    @property
    def data(self) -> cp.ndarray:
        """Возвращает underlying CuPy array."""
        return self._data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
    
    @property
    def size(self) -> int:
        return self._data.size
    
    @property
    def dtype(self):
        return self._data.dtype
    
    def to_numpy(self) -> np.ndarray:
        """Копирует данные на CPU."""
        return cp.asnumpy(self._data)
    
    def __len__(self) -> int:
        return self._data.size
    
    def __repr__(self) -> str:
        return f"CUDAArray({self.to_numpy()})"
    
    def __str__(self) -> str:
        return str(self.to_numpy())


class CUDAStruct:
    """Представление структуры для GPU."""
    
    def __init__(self, struct_def: Dict[str, type], count: int = 1):
        """
        struct_def: {'x': float, 'y': float, 'vx': float, 'vy': float}
        count: количество структур в массиве
        """
        self._fields = list(struct_def.keys())
        self._types = struct_def
        self._count = count
        
        # Создаём структурированный dtype для NumPy/CuPy
        dtype_list = []
        for name, t in struct_def.items():
            if t is float or t == 'float32':
                dtype_list.append((name, cp.float32))
            elif t is int or t == 'int32':
                dtype_list.append((name, cp.int32))
            elif t is bool or t == 'bool':
                dtype_list.append((name, cp.bool_))
            else:
                dtype_list.append((name, cp.float32))
        
        self._dtype = cp.dtype(dtype_list)
        self._data = cp.zeros(count, dtype=self._dtype)
    
    @property
    def data(self) -> cp.ndarray:
        return self._data
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]
        return self._data[key]
    
    def __setitem__(self, key, value):
        if isinstance(key, str):
            self._data[key] = value
        else:
            self._data[key] = value
    
    def to_numpy(self) -> np.ndarray:
        return cp.asnumpy(self._data)
    
    def __len__(self) -> int:
        return self._count
    
    def __repr__(self) -> str:
        return f"CUDAStruct({self._fields}, count={self._count})"


# ============================================================================
# Автоматический подбор grid/block размеров
# ============================================================================

def auto_launch_config(size: int, max_threads: int = 256) -> Tuple[Tuple[int], Tuple[int]]:
    """Автоматически подбирает оптимальные grid и block размеры."""
    # Ограничиваем количество потоков на блок
    threads_per_block = min(max_threads, size)
    
    # Округляем до степени двойки для эффективности
    if threads_per_block > 1:
        threads_per_block = 1 << (threads_per_block - 1).bit_length()
        threads_per_block = min(threads_per_block, max_threads)
    
    # Количество блоков
    num_blocks = (size + threads_per_block - 1) // threads_per_block
    
    return (num_blocks,), (threads_per_block,)


def auto_launch_config_2d(
    shape: Tuple[int, int],
    max_threads: int = 256
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Автоматически подбирает grid и block для 2D."""
    height, width = shape
    
    # Квадратный блок
    block_dim = int(max_threads ** 0.5)
    block = (block_dim, block_dim)
    
    grid_x = (width + block[0] - 1) // block[0]
    grid_y = (height + block[1] - 1) // block[1]
    grid = (grid_x, grid_y)
    
    return grid, block


# ============================================================================
# Memory management helpers
# ============================================================================

class GPUMemoryManager:
    """Управление памятью GPU."""
    
    @staticmethod
    def get_free_memory() -> int:
        """Возвращает свободную память GPU в байтах."""
        mempool = cp.get_default_memory_pool()
        return mempool.free_bytes()
    
    @staticmethod
    def get_used_memory() -> int:
        """Возвращает используемую память GPU в байтах."""
        mempool = cp.get_default_memory_pool()
        return mempool.used_bytes()
    
    @staticmethod
    def get_total_memory() -> int:
        """Возвращает общую память GPU в байтах."""
        mempool = cp.get_default_memory_pool()
        return mempool.total_bytes()
    
    @staticmethod
    def clear_cache():
        """Очищает кеш памяти."""
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    
    @staticmethod
    def memory_info() -> Dict[str, int]:
        """Возвращает информацию о памяти."""
        return {
            'free': GPUMemoryManager.get_free_memory(),
            'used': GPUMemoryManager.get_used_memory(),
            'total': GPUMemoryManager.get_total_memory()
        }


# ============================================================================
# Profiling helpers
# ============================================================================

class CUDATimer:
    """Таймер для измерения производительности GPU."""
    
    def __init__(self):
        self._start_event = None
        self._end_event = None
    
    def start(self):
        """Начинает измерение."""
        self._start_event = cp.cuda.Event()
        self._end_event = cp.cuda.Event()
        self._start_event.record()
    
    def stop(self) -> float:
        """Останавливает измерение и возвращает время в миллисекундах."""
        self._end_event.record()
        self._end_event.synchronize()
        return cp.cuda.get_elapsed_time(self._start_event, self._end_event)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        elapsed = self.stop()
        print(f"GPU time: {elapsed:.3f} ms")
