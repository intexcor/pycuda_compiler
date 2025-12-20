#!/usr/bin/env python3
"""
PyCUDA Compiler - Benchmark CPU vs GPU

Сравнение производительности CPU (NumPy) vs GPU (CUDA).
"""

import sys
import time
sys.path.insert(0, '/home/claude/pycuda_compiler')

try:
    import cupy as cp
    HAS_GPU = True

    try:
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(0)

        name = props["name"].decode()
        cc = f"{props['major']}.{props['minor']}"
        free_mem, total_mem = device.mem_info

        print(f"✅ GPU найден: {name}")
        print(f"   Compute Capability: {cc}")
        print(
            f"   Memory: {total_mem / 1024**3:.1f} GB total, "
            f"{free_mem / 1024**3:.1f} GB free"
        )

    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"❌ CUDA runtime error: {e}")
        HAS_GPU = False
except ImportError:
    print("❌ CuPy не установлен")
    HAS_GPU = False

import numpy as np
from compiler import CUDAProgram

# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_cpu_numpy(func, *args, warmup=2, runs=10):
    """Бенчмарк CPU с NumPy."""
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def benchmark_gpu_cuda(program, kernel_name, *args, warmup=2, runs=10):
    """Бенчмарк GPU с CUDA."""
    if not HAS_GPU:
        return None
    
    # Конвертируем в CuPy arrays
    gpu_args = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            gpu_args.append(cp.asarray(arg))
        else:
            gpu_args.append(arg)
    
    # Warmup
    for _ in range(warmup):
        program.run(kernel_name, *gpu_args)
        cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    times = []
    for _ in range(runs):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        program.run(kernel_name, *gpu_args)
        end.record()
        end.synchronize()
        
        times.append(cp.cuda.get_elapsed_time(start, end))
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def print_benchmark_result(name, cpu_result, gpu_result, size):
    """Выводит результаты бенчмарка."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {name}")
    print(f"Data size: {size:,} elements")
    print(f"{'=' * 60}")
    
    print(f"\n  CPU (NumPy):")
    print(f"    Mean: {cpu_result['mean']:.3f} ms")
    print(f"    Std:  {cpu_result['std']:.3f} ms")
    print(f"    Min:  {cpu_result['min']:.3f} ms")
    print(f"    Max:  {cpu_result['max']:.3f} ms")
    
    if gpu_result:
        print(f"\n  GPU (CUDA):")
        print(f"    Mean: {gpu_result['mean']:.3f} ms")
        print(f"    Std:  {gpu_result['std']:.3f} ms")
        print(f"    Min:  {gpu_result['min']:.3f} ms")
        print(f"    Max:  {gpu_result['max']:.3f} ms")
        
        speedup = cpu_result['mean'] / gpu_result['mean']
        print(f"\n  🚀 Speedup: {speedup:.1f}x")
        
        if speedup > 1:
            print(f"     GPU быстрее на {(speedup - 1) * 100:.1f}%")
        else:
            print(f"     CPU быстрее на {(1 - speedup) * 100:.1f}%")
    else:
        print(f"\n  GPU: недоступен")


# ============================================================================
# Test 1: Vector Addition
# ============================================================================

def test_vector_add(size=10_000_000):
    """Сложение векторов."""
    print("\n" + "=" * 60)
    print("TEST 1: Vector Addition (c = a + b)")
    print("=" * 60)
    
    # CPU version
    def cpu_vector_add(a, b, c):
        np.add(a, b, out=c)
    
    # GPU version
    program = CUDAProgram("""
@kernel
def vector_add(a: Array[float32], b: Array[float32], c: Array[float32]):
    for i in range(len(a)):
        c[i] = a[i] + b[i]
""")
    
    # Data
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros(size, dtype=np.float32)
    
    cpu_result = benchmark_cpu_numpy(cpu_vector_add, a, b, c)
    gpu_result = benchmark_gpu_cuda(program, 'vector_add', a, b, c.copy()) if HAS_GPU else None
    
    print_benchmark_result("Vector Addition", cpu_result, gpu_result, size)
    
    # Verify correctness
    if HAS_GPU:
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        c_gpu = cp.zeros(size, dtype=cp.float32)
        program.run('vector_add', a_gpu, b_gpu, c_gpu)
        
        c_expected = a + b
        c_actual = cp.asnumpy(c_gpu)
        
        if np.allclose(c_expected, c_actual):
            print("  ✅ Results match!")
        else:
            print("  ❌ Results differ!")


# ============================================================================
# Test 2: Element-wise Math Operations
# ============================================================================

def test_math_ops(size=10_000_000):
    """Математические операции."""
    print("\n" + "=" * 60)
    print("TEST 2: Math Operations (sqrt, sin, exp)")
    print("=" * 60)
    
    # CPU version
    def cpu_math_ops(x, out):
        np.sqrt(np.abs(x) + 1, out=out)
        np.sin(out, out=out)
        np.exp(out * 0.1, out=out)
    
    # GPU version
    program = CUDAProgram("""
@kernel
def math_ops(x: Array[float32], out: Array[float32]):
    for i in range(len(x)):
        val = sqrt(abs(x[i]) + 1.0)
        val = sin(val)
        out[i] = exp(val * 0.1)
""")
    
    # Data
    x = np.random.randn(size).astype(np.float32) * 10
    out = np.zeros(size, dtype=np.float32)
    
    cpu_result = benchmark_cpu_numpy(cpu_math_ops, x, out)
    gpu_result = benchmark_gpu_cuda(program, 'math_ops', x, out.copy()) if HAS_GPU else None
    
    print_benchmark_result("Math Operations", cpu_result, gpu_result, size)


# ============================================================================
# Test 3: Conditional Processing
# ============================================================================

def test_conditional(size=10_000_000):
    """Условная обработка."""
    print("\n" + "=" * 60)
    print("TEST 3: Conditional Processing (if/else)")
    print("=" * 60)
    
    # CPU version
    def cpu_conditional(x, out):
        mask_pos = x > 0
        mask_neg = ~mask_pos
        out[mask_pos] = np.sqrt(x[mask_pos])
        out[mask_neg] = x[mask_neg] ** 2
    
    # GPU version
    program = CUDAProgram("""
@kernel
def conditional(x: Array[float32], out: Array[float32]):
    for i in range(len(x)):
        if x[i] > 0:
            out[i] = sqrt(x[i])
        else:
            out[i] = x[i] * x[i]
""")
    
    # Data
    x = np.random.randn(size).astype(np.float32)
    out = np.zeros(size, dtype=np.float32)
    
    cpu_result = benchmark_cpu_numpy(cpu_conditional, x, out)
    gpu_result = benchmark_gpu_cuda(program, 'conditional', x, out.copy()) if HAS_GPU else None
    
    print_benchmark_result("Conditional Processing", cpu_result, gpu_result, size)


# ============================================================================
# Test 4: Reduction-like (but parallel)
# ============================================================================

def test_normalize(size=10_000_000):
    """Нормализация (каждый элемент делим на его abs + 1)."""
    print("\n" + "=" * 60)
    print("TEST 4: Normalize (x / (|x| + 1))")
    print("=" * 60)
    
    # CPU version
    def cpu_normalize(x, out):
        np.divide(x, np.abs(x) + 1, out=out)
    
    # GPU version
    program = CUDAProgram("""
@kernel
def normalize(x: Array[float32], out: Array[float32]):
    for i in range(len(x)):
        out[i] = x[i] / (abs(x[i]) + 1.0)
""")
    
    # Data
    x = np.random.randn(size).astype(np.float32) * 100
    out = np.zeros(size, dtype=np.float32)
    
    cpu_result = benchmark_cpu_numpy(cpu_normalize, x, out)
    gpu_result = benchmark_gpu_cuda(program, 'normalize', x, out.copy()) if HAS_GPU else None
    
    print_benchmark_result("Normalize", cpu_result, gpu_result, size)


# ============================================================================
# Test 5: Saxpy (a*x + y)
# ============================================================================

def test_saxpy(size=10_000_000):
    """SAXPY: y = a*x + y."""
    print("\n" + "=" * 60)
    print("TEST 5: SAXPY (y = a*x + y)")
    print("=" * 60)
    
    a = 2.5
    
    # CPU version
    def cpu_saxpy(x, y, a):
        y += a * x
    
    # GPU version
    program = CUDAProgram("""
@kernel
def saxpy(x: Array[float32], y: Array[float32], a: float32):
    for i in range(len(x)):
        y[i] = a * x[i] + y[i]
""")
    
    # Data
    x = np.random.randn(size).astype(np.float32)
    y = np.random.randn(size).astype(np.float32)
    
    cpu_result = benchmark_cpu_numpy(cpu_saxpy, x, y.copy(), a)
    gpu_result = benchmark_gpu_cuda(program, 'saxpy', x, y.copy(), np.float32(a)) if HAS_GPU else None
    
    print_benchmark_result("SAXPY", cpu_result, gpu_result, size)


# ============================================================================
# Test 6: Complex computation
# ============================================================================

def test_complex(size=5_000_000):
    """Сложные вычисления."""
    print("\n" + "=" * 60)
    print("TEST 6: Complex (sin²(x) + cos²(x) * exp(-|x|))")
    print("=" * 60)
    
    # CPU version
    def cpu_complex(x, out):
        s = np.sin(x)
        c = np.cos(x)
        out[:] = s * s + c * c * np.exp(-np.abs(x))
    
    # GPU version
    program = CUDAProgram("""
@kernel
def complex_calc(x: Array[float32], out: Array[float32]):
    for i in range(len(x)):
        s = sin(x[i])
        c = cos(x[i])
        out[i] = s * s + c * c * exp(-abs(x[i]))
""")
    
    # Data
    x = np.random.randn(size).astype(np.float32)
    out = np.zeros(size, dtype=np.float32)
    
    cpu_result = benchmark_cpu_numpy(cpu_complex, x, out)
    gpu_result = benchmark_gpu_cuda(program, 'complex_calc', x, out.copy()) if HAS_GPU else None
    
    print_benchmark_result("Complex Computation", cpu_result, gpu_result, size)


# ============================================================================
# Test 7: Different sizes
# ============================================================================

def test_scaling():
    """Тест масштабирования по размеру данных."""
    print("\n" + "=" * 60)
    print("TEST 7: Scaling Analysis")
    print("=" * 60)
    
    program = CUDAProgram("""
@kernel
def scale_test(x: Array[float32]):
    for i in range(len(x)):
        x[i] = sqrt(x[i] * x[i] + 1.0)
""")
    
    def cpu_scale(x):
        x[:] = np.sqrt(x * x + 1.0)
    
    sizes = [1_000, 10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]
    
    print(f"\n{'Size':>12} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'Speedup':>10}")
    print("-" * 50)
    
    for size in sizes:
        x = np.random.randn(size).astype(np.float32)
        
        cpu_result = benchmark_cpu_numpy(cpu_scale, x.copy(), runs=5)
        gpu_result = benchmark_gpu_cuda(program, 'scale_test', x.copy(), runs=5) if HAS_GPU else None
        
        if gpu_result:
            speedup = cpu_result['mean'] / gpu_result['mean']
            print(f"{size:>12,} | {cpu_result['mean']:>10.3f} | {gpu_result['mean']:>10.3f} | {speedup:>10.1f}x")
        else:
            print(f"{size:>12,} | {cpu_result['mean']:>10.3f} | {'N/A':>10} | {'N/A':>10}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("PyCUDA Compiler - CPU vs GPU Benchmark")
    print("=" * 60)
    
    if not HAS_GPU:
        print("\n⚠️  GPU недоступен, будут показаны только CPU результаты")
        print("   Для полного теста нужна NVIDIA GPU + CuPy")
    
    # Run benchmarks
    test_vector_add()
    test_math_ops()
    test_conditional()
    test_normalize()
    test_saxpy()
    test_complex()
    test_scaling()
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
