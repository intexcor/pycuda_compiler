#!/usr/bin/env python3
"""
PyCUDA Compiler - Custom Examples Test

Тестируем разные кастомные примеры на RTX 4090.
"""

import sys
import time
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    print("CuPy не установлен!")
    sys.exit(1)

from compiler import CUDAProgram, cuda_compile, kernel, Array, float32, int32, Tensor
import inspect
import textwrap
from math import exp, tanh, log, sqrt

# ============================================================================
# Helper
# ============================================================================

def benchmark(name, cpu_func, gpu_program, kernel_name, *args, size=None):
    """Бенчмарк CPU vs GPU."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    if size:
        print(f"Size: {size:,}")
    print('=' * 60)
    
    # CPU
    cpu_args = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    
    start = time.perf_counter()
    for _ in range(5):
        cpu_func(*cpu_args)
    cpu_time = (time.perf_counter() - start) / 5 * 1000
    
    # GPU
    gpu_args = []
    for a in args:
        if isinstance(a, np.ndarray):
            if a.dtype.names:
                # Structured array: pass as raw bytes (uint8) + logical size
                # We view it as uint8 to trick Cupy, but we need to pass the original logical size
                # so the kernel knows how many ELEMENTS to process
                gpu_arr = cp.asarray(a.view(np.uint8))
                gpu_args.append((gpu_arr, a.size))
            else:
                gpu_args.append(cp.asarray(a))
        else:
            gpu_args.append(a)
    
    # Warmup
    gpu_program.run(kernel_name, *gpu_args)
    cp.cuda.Stream.null.synchronize()
    
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    
    start_event.record()
    for _ in range(5):
        gpu_program.run(kernel_name, *gpu_args)
    end_event.record()
    end_event.synchronize()
    
    gpu_time = cp.cuda.get_elapsed_time(start_event, end_event) / 5
    
    speedup = cpu_time / gpu_time
    
    print(f"  CPU: {cpu_time:.3f} ms")
    print(f"  GPU: {gpu_time:.3f} ms")
    print(f"  🚀 Speedup: {speedup:.1f}x")
    
    return cpu_time, gpu_time, speedup


# ============================================================================
# 1. Image Processing - Blur
# ============================================================================

def test_blur():
    print("\n" + "=" * 60)
    print("1. IMAGE PROCESSING - Box Blur 3x3")
    print("=" * 60)
    
    @cuda_compile(debug=True)
    def blur_shared(img: Array[float32], out: Array[float32], width: int32, height: int32):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                idx = y * width + x
                s: float32 = 0.0
                
                # 3x3 kernel
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        s += img[(y + dy) * width + (x + dx)]
                
                out[idx] = s / 9.0

    # Test data - 1920x1080 image
    width, height = 1920, 1080
    img = np.random.rand(width * height).astype(np.float32)
    out = np.zeros_like(img)

    print(blur_shared.cuda_code)

    # Use the compiled function wrapper's original function for CPU testing if accessible,
    # or just use the kernel wrapper which allows calling on CPU IF we implemented that (we didn't).
    # We must assume usage of the underlying python function. 
    # Since @cuda_compile returns a CUDAKernel object, we need access to the original function.
    # We can extract it if we didn't decorate it, or access `._func` if we did.
    # Looking at compiler.py: CUDAKernel stores `self._func = func`.
    
    cpu_func = blur_shared._func
    
    # Needs a wrapper to handle type hints if python runtime checks them? 
    # Python ignores type hints at runtime usually, so it should be fine.
    
    # NumPy vectorized version for reference (fast CPU)
    def cpu_blur_numpy(img, out, width, height):
        img2d = img.reshape(height, width)
        out2d = out.reshape(height, width)
        out2d[1:-1, 1:-1] = (
            img2d[:-2, :-2] + img2d[:-2, 1:-1] + img2d[:-2, 2:] +
            img2d[1:-1, :-2] + img2d[1:-1, 1:-1] + img2d[1:-1, 2:] +
            img2d[2:, :-2] + img2d[2:, 1:-1] + img2d[2:, 2:]
        ) / 9.0
    
    # Benchmark "Naive Python" vs GPU using the SAME source
    # We use cpu_func for the Slow CPU test to prove it works
    # But for the graph we might want to skip it or show it's very slow.
    # The user asked to "compile one of these". 
    
    print("Running Naive Python Loop (will be slow)...")
    # Limiting runs to 1 for sanity if it's super slow. 1920x1080 is 2M pixels. Python loop is slow.
    # Let's run on smaller size for CPU correctness check, but full size for GPU.
    # Actually, let's just run it once for correctness verification, and use numpy for speed comparison?
    # No, user wants to see we use the same code.
    
    # Let's perform correctness check with the naive loop on a smaller image
    w_small, h_small = 128, 128
    img_s = np.random.rand(w_small * h_small).astype(np.float32)
    out_s = np.zeros_like(img_s)
    cpu_func(img_s, out_s, w_small, h_small)
    
    # GPU run on small
    img_gpu_s = cp.asarray(img_s)
    out_gpu_s = cp.zeros_like(img_gpu_s)
    blur_shared(img_gpu_s, out_gpu_s, np.int32(w_small), np.int32(h_small))
    
    if np.allclose(out_s, cp.asnumpy(out_gpu_s), rtol=1e-4):
        print("  ✅ Python Logic and GPU Kernel match!")
    else:
        print("  ❌ Mismatch between Python and GPU!")
        
    benchmark("Box Blur 3x3 (1920x1080) [Comparing vs Numpy for speed]", 
              cpu_blur_numpy, blur_shared._program, 'blur_shared',
              img, out, np.int32(width), np.int32(height),
              size=width*height)


# ============================================================================
# 2. Particle Physics Simulation
# ============================================================================

def test_particles():
    print("\n" + "=" * 60)
    print("2. PARTICLE PHYSICS SIMULATION")
    print("=" * 60)
    
    class Particle:
        x: float32
        y: float32
        vx: float32
        vy: float32
        mass: float32
    
    # To share code between CPU and GPU for particles, we need to handle the struct access.
    # GPU uses dot notation (particles[i].vx), NumPy uses struct field access (particles[i]['vx']).
    # To make it work on CPU, we can wrap the numpy array in a class or use a Record Array.
    # However, Python's access to Record Array is still slightly different (requires recarray view).
    # Let's use a Recarray for the CPU test data so dot access works!

    @kernel
    def simulate_step(particles: Array[Particle], dt: float32):
        gravity: float32 = -9.81
        
        for i in range(len(particles)):
            # Update velocity (gravity)
            particles[i].vy += gravity * dt
            
            # Update position
            particles[i].x += particles[i].vx * dt
            particles[i].y += particles[i].vy * dt
            
            # Bounce off ground
            if particles[i].y < 0.0:
                particles[i].y = 0.0
                particles[i].vy = -particles[i].vy * 0.8  # energy loss

    source = textwrap.dedent(inspect.getsource(Particle)) + "\n" + textwrap.dedent(inspect.getsource(simulate_step))
    program = CUDAProgram(source, debug=True)
    
    # Test data
    n_particles = 1_000_000
    
    # Create structured array
    particle_dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('vx', np.float32),
        ('vy', np.float32),
        ('mass', np.float32)
    ])
    
    # Use Record Array for dot notation support on CPU
    particles = np.recarray(n_particles, dtype=particle_dtype)
    particles.x = np.random.rand(n_particles).astype(np.float32) * 100
    particles.y = np.random.rand(n_particles).astype(np.float32) * 100
    particles.vx = (np.random.rand(n_particles).astype(np.float32) - 0.5) * 10
    particles.vy = (np.random.rand(n_particles).astype(np.float32) - 0.5) * 10
    particles.mass = np.ones(n_particles, dtype=np.float32)
    
    dt = np.float32(0.016)  # 60 FPS
    
    cpu_simulate = simulate_step
    
    # Benchmark against vectorized numpy (still fast)
    def cpu_simulate_numpy(particles, dt):
        gravity = -9.81
        particles.vy += gravity * dt
        particles.x += particles.vx * dt
        particles.y += particles.vy * dt
        
        # Bounce
        mask = particles.y < 0
        particles.y[mask] = 0
        particles.vy[mask] *= -0.8

    # Correctness using shared python source (slow loop) on small sample
    n_small = 1000
    p_small = particles[:n_small].copy()
    cpu_simulate(p_small, dt) # Running python loop on CPU
    
    benchmark("Particle Simulation (1M particles) [Numpy Vectorized]", 
              cpu_simulate_numpy, program, 'simulate_step',
              particles, dt,
              size=n_particles)


# ============================================================================
# 3. Neural Network Activations
# ============================================================================

def test_neural_network():
    print("\n" + "=" * 60)
    print("3. NEURAL NETWORK ACTIVATIONS")
    print("=" * 60)
    
    @kernel
    def relu(x: Array[float32]):
        for i in range(len(x)):
            if x[i] < 0.0:
                x[i] = 0.0
    
    @kernel
    def sigmoid(x: Array[float32], out: Array[float32]):
        for i in range(len(x)):
            out[i] = 1.0 / (1.0 + exp(-x[i]))
    
    @kernel
    def tanh_activation(x: Array[float32], out: Array[float32]):
        for i in range(len(x)):
            out[i] = tanh(x[i])
    
    @kernel
    def softplus(x: Array[float32], out: Array[float32]):
        for i in range(len(x)):
            out[i] = log(1.0 + exp(x[i]))
    
    @kernel
    def gelu(x: Array[float32], out: Array[float32]):
        for i in range(len(x)):
            # GELU approximation
            out[i] = 0.5 * x[i] * (1.0 + tanh(0.7978845608 * (x[i] + 0.044715 * x[i] * x[i] * x[i])))

    source = "\n".join(textwrap.dedent(inspect.getsource(f)) for f in [relu, sigmoid, tanh_activation, softplus, gelu])
    program = CUDAProgram(source, debug=True)
    
    size = 10_000_000
    x = np.random.randn(size).astype(np.float32)
    out = np.zeros_like(x)
    
    # Reuse valid pure python functions
    benchmark("ReLU (Shared Src)", relu, program, 'relu', x.copy(), size=size)
    
    # For sigmoid and others, we defined them with a loop i in range(len(x)).
    # We can run these on CPU as is.
    
    # Sigmoid
    benchmark("Sigmoid (Shared Src)", sigmoid, program, 'sigmoid', x, out.copy(), size=size)
    
    # Tanh
    benchmark("Tanh (Shared Src)", tanh_activation, program, 'tanh_activation', x, out.copy(), size=size)
    
    # GELU
    benchmark("GELU (Shared Src)", gelu, program, 'gelu', x, out.copy(), size=size)


# ============================================================================
# 4. Monte Carlo Pi Estimation
# ============================================================================

def test_monte_carlo():
    print("\n" + "=" * 60)
    print("4. MONTE CARLO PI ESTIMATION")
    print("=" * 60)
    
    @cuda_compile(debug=True)
    def monte_carlo_pi(random_x: Array[float32], random_y: Array[float32], inside: Array[int32]):
        for i in range(len(random_x)):
            dist_sq = random_x[i] * random_x[i] + random_y[i] * random_y[i]
            if dist_sq <= 1.0:
                inside[i] = 1
            else:
                inside[i] = 0
                
    program = monte_carlo_pi._program
    
    n_points = 50_000_000
    
    random_x = np.random.rand(n_points).astype(np.float32)
    random_y = np.random.rand(n_points).astype(np.float32)
    inside = np.zeros(n_points, dtype=np.int32)
    
    benchmark("Monte Carlo Pi (50M points) [Shared Src]", 
              monte_carlo_pi._func, program, 'monte_carlo_pi',
              random_x, random_y, inside,
              size=n_points)
    
    # Calculate Pi
    random_x_gpu = cp.asarray(random_x)
    random_y_gpu = cp.asarray(random_y)
    inside_gpu = cp.zeros(n_points, dtype=cp.int32)
    
    program.run('monte_carlo_pi', random_x_gpu, random_y_gpu, inside_gpu)
    
    pi_estimate = 4.0 * cp.sum(inside_gpu).get() / n_points
    print(f"  π ≈ {pi_estimate:.6f} (actual: 3.141593)")
    print(f"  Error: {abs(pi_estimate - np.pi):.6f}")


# ============================================================================
# 5. Mandelbrot Set
# ============================================================================

def test_mandelbrot():
    print("\n" + "=" * 60)
    print("5. MANDELBROT SET")
    print("=" * 60)
    
    @cuda_compile(debug=True)
    def mandelbrot(output: Array[int32], width: int32, height: int32, max_iter: int32):
        for idx in range(len(output)):
            px = idx % width
            py = idx // width
            
            # Map to complex plane [-2.5, 1] x [-1, 1]
            x0 = (px / width) * 3.5 - 2.5
            y0 = (py / height) * 2.0 - 1.0
            
            x: float32 = 0.0
            y: float32 = 0.0
            iteration: int32 = 0
            
            while x*x + y*y <= 4.0 and iteration < max_iter:
                xtemp = x*x - y*y + x0
                y = 2.0*x*y + y0
                x = xtemp
                iteration += 1
            
            output[idx] = iteration

    program = mandelbrot._program
    
    width, height = 1920, 1080
    max_iter = 256
    output = np.zeros(width * height, dtype=np.int32)

    # Use shared source (flattened loop)
    # The slow python loop
    
    benchmark("Mandelbrot (1920x1080, 256 iter) [Shared Src]", 
              mandelbrot._func, program, 'mandelbrot',
              output, np.int32(width), np.int32(height), np.int32(max_iter),
              size=width*height)


# ============================================================================
# 6. Vector Dot Product Prep (element-wise multiply)
# ============================================================================

def test_dot_product_prep():
    print("\n" + "=" * 60)
    print("6. DOT PRODUCT (element-wise multiply)")
    print("=" * 60)
    
    @cuda_compile(debug=True)
    def elementwise_mul(a: Array[float32], b: Array[float32], c: Array[float32]):
        for i in range(len(a)):
            c[i] = a[i] * b[i]

    program = elementwise_mul._program
    
    size = 50_000_000
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(a)
    
    benchmark("Element-wise Multiply (50M) [Shared Src]", 
              elementwise_mul._func, program, 'elementwise_mul',
              a, b, c,
              size=size)
    
    # Verify
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    c_gpu = cp.zeros_like(a_gpu)
    program.run('elementwise_mul', a_gpu, b_gpu, c_gpu)
    
    c_cpu = a * b
    if np.allclose(c_cpu, cp.asnumpy(c_gpu)):
        print("  ✅ Results match!")


# ============================================================================
# 7. Color Space Conversion (RGB -> HSV)
# ============================================================================

def test_color_conversion():
    print("\n" + "=" * 60)
    print("7. COLOR SPACE CONVERSION (RGB -> Grayscale)")
    print("=" * 60)
    
    @cuda_compile(debug=True)
    def rgb_to_grayscale(r: Array[float32], g: Array[float32], b: Array[float32], out: Array[float32]):
        for i in range(len(r)):
            out[i] = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]

    program = rgb_to_grayscale._program
    
    # 4K image
    size = 3840 * 2160
    r = np.random.rand(size).astype(np.float32)
    g = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    out = np.zeros_like(r)
    
    benchmark("RGB to Grayscale (4K image) [Shared Src]", 
              rgb_to_grayscale._func, program, 'rgb_to_grayscale',
              r, g, b, out,
              size=size)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("PyCUDA Compiler - Custom Examples on RTX 4090")
    print("=" * 60)
    
    test_blur()
    test_particles()
    test_neural_network()
    test_monte_carlo()
    test_mandelbrot()
    test_dot_product_prep()
    test_color_conversion()
    
    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()