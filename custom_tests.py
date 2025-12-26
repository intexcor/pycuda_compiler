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

from compiler import CUDAProgram

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
    
    program = CUDAProgram("""
@kernel
def blur(img: Array[float32], out: Array[float32], width: int32, height: int32):
    for idx in range(len(img)):
        x = idx % width
        y = idx // width
        
        if x > 0 and x < width - 1 and y > 0 and y < height - 1:
            sum: float32 = 0.0
            
            # 3x3 kernel
            sum += img[(y-1) * width + (x-1)]
            sum += img[(y-1) * width + x]
            sum += img[(y-1) * width + (x+1)]
            sum += img[y * width + (x-1)]
            sum += img[y * width + x]
            sum += img[y * width + (x+1)]
            sum += img[(y+1) * width + (x-1)]
            sum += img[(y+1) * width + x]
            sum += img[(y+1) * width + (x+1)]
            
            out[idx] = sum / 9.0
        else:
            out[idx] = img[idx]
""", debug=True)
    
    # Test data - 1920x1080 image
    width, height = 1920, 1080
    img = np.random.rand(width * height).astype(np.float32)
    out = np.zeros_like(img)
    
    def cpu_blur(img, out, width, height):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                idx = y * width + x
                s = 0.0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        s += img[(y + dy) * width + (x + dx)]
                out[idx] = s / 9.0
    
    # NumPy vectorized version for fair comparison
    def cpu_blur_numpy(img, out, width, height):
        img2d = img.reshape(height, width)
        out2d = out.reshape(height, width)
        out2d[1:-1, 1:-1] = (
            img2d[:-2, :-2] + img2d[:-2, 1:-1] + img2d[:-2, 2:] +
            img2d[1:-1, :-2] + img2d[1:-1, 1:-1] + img2d[1:-1, 2:] +
            img2d[2:, :-2] + img2d[2:, 1:-1] + img2d[2:, 2:]
        ) / 9.0
    
    benchmark("Box Blur 3x3 (1920x1080)", 
              cpu_blur_numpy, program, 'blur',
              img, out, np.int32(width), np.int32(height),
              size=width*height)
    
    # Verify
    img_gpu = cp.asarray(img)
    out_gpu = cp.zeros_like(img_gpu)
    program.run('blur', img_gpu, out_gpu, np.int32(width), np.int32(height))
    
    out_cpu = np.zeros_like(img)
    cpu_blur_numpy(img, out_cpu, width, height)
    
    if np.allclose(out_cpu, cp.asnumpy(out_gpu), rtol=1e-4):
        print("  ✅ Results match!")
    else:
        diff = np.abs(out_cpu - cp.asnumpy(out_gpu)).max()
        print(f"  ⚠️  Max diff: {diff}")


# ============================================================================
# 2. Particle Physics Simulation
# ============================================================================

def test_particles():
    print("\n" + "=" * 60)
    print("2. PARTICLE PHYSICS SIMULATION")
    print("=" * 60)
    
    program = CUDAProgram("""
class Particle:
    x: float32
    y: float32
    vx: float32
    vy: float32
    mass: float32

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
""", debug=True)
    
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
    
    particles = np.zeros(n_particles, dtype=particle_dtype)
    particles['x'] = np.random.rand(n_particles).astype(np.float32) * 100
    particles['y'] = np.random.rand(n_particles).astype(np.float32) * 100
    particles['vx'] = (np.random.rand(n_particles).astype(np.float32) - 0.5) * 10
    particles['vy'] = (np.random.rand(n_particles).astype(np.float32) - 0.5) * 10
    particles['mass'] = np.ones(n_particles, dtype=np.float32)
    
    dt = np.float32(0.016)  # 60 FPS
    
    def cpu_simulate(particles, dt):
        gravity = -9.81
        particles['vy'] += gravity * dt
        particles['x'] += particles['vx'] * dt
        particles['y'] += particles['vy'] * dt
        
        # Bounce
        mask = particles['y'] < 0
        particles['y'][mask] = 0
        particles['vy'][mask] *= -0.8
    
    benchmark("Particle Simulation (1M particles)", 
              cpu_simulate, program, 'simulate_step',
              particles, dt,
              size=n_particles)


# ============================================================================
# 3. Neural Network Activations
# ============================================================================

def test_neural_network():
    print("\n" + "=" * 60)
    print("3. NEURAL NETWORK ACTIVATIONS")
    print("=" * 60)
    
    program = CUDAProgram("""
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
""", debug=True)
    
    size = 10_000_000
    x = np.random.randn(size).astype(np.float32)
    out = np.zeros_like(x)
    
    # ReLU
    def cpu_relu(x):
        x[x < 0] = 0
    
    benchmark("ReLU", cpu_relu, program, 'relu', x.copy(), size=size)
    
    # Sigmoid
    def cpu_sigmoid(x, out):
        np.divide(1.0, 1.0 + np.exp(-x), out=out)
    
    benchmark("Sigmoid", cpu_sigmoid, program, 'sigmoid', x, out.copy(), size=size)
    
    # Tanh
    def cpu_tanh(x, out):
        np.tanh(x, out=out)
    
    benchmark("Tanh", cpu_tanh, program, 'tanh_activation', x, out.copy(), size=size)
    
    # GELU
    def cpu_gelu(x, out):
        out[:] = 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    
    benchmark("GELU", cpu_gelu, program, 'gelu', x, out.copy(), size=size)


# ============================================================================
# 4. Monte Carlo Pi Estimation
# ============================================================================

def test_monte_carlo():
    print("\n" + "=" * 60)
    print("4. MONTE CARLO PI ESTIMATION")
    print("=" * 60)
    
    program = CUDAProgram("""
@kernel
def monte_carlo_pi(random_x: Array[float32], random_y: Array[float32], inside: Array[int32]):
    for i in range(len(random_x)):
        dist_sq = random_x[i] * random_x[i] + random_y[i] * random_y[i]
        if dist_sq <= 1.0:
            inside[i] = 1
        else:
            inside[i] = 0
""", debug=True)
    
    n_points = 50_000_000
    
    random_x = np.random.rand(n_points).astype(np.float32)
    random_y = np.random.rand(n_points).astype(np.float32)
    inside = np.zeros(n_points, dtype=np.int32)
    
    def cpu_monte_carlo(random_x, random_y, inside):
        dist_sq = random_x**2 + random_y**2
        inside[:] = (dist_sq <= 1.0).astype(np.int32)
    
    benchmark("Monte Carlo Pi (50M points)", 
              cpu_monte_carlo, program, 'monte_carlo_pi',
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
    
    program = CUDAProgram("""
@kernel
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
""", debug=True)
    
    width, height = 1920, 1080
    max_iter = 256
    output = np.zeros(width * height, dtype=np.int32)
    
    def cpu_mandelbrot(output, width, height, max_iter):
        for py in range(height):
            for px in range(width):
                x0 = (px / width) * 3.5 - 2.5
                y0 = (py / height) * 2.0 - 1.0
                x, y = 0.0, 0.0
                iteration = 0
                while x*x + y*y <= 4.0 and iteration < max_iter:
                    x, y = x*x - y*y + x0, 2*x*y + y0
                    iteration += 1
                output[py * width + px] = iteration
    
    # NumPy vectorized (still slow for this)
    def cpu_mandelbrot_numpy(output, width, height, max_iter):
        px = np.arange(width)
        py = np.arange(height)
        px, py = np.meshgrid(px, py)
        
        x0 = (px / width) * 3.5 - 2.5
        y0 = (py / height) * 2.0 - 1.0
        
        c = x0 + 1j * y0
        z = np.zeros_like(c)
        output_2d = output.reshape(height, width)
        output_2d[:] = max_iter
        
        for i in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = z[mask]**2 + c[mask]
            output_2d[mask & (np.abs(z) > 2)] = i
    
    benchmark("Mandelbrot (1920x1080, 256 iter)", 
              cpu_mandelbrot_numpy, program, 'mandelbrot',
              output, np.int32(width), np.int32(height), np.int32(max_iter),
              size=width*height)


# ============================================================================
# 6. Vector Dot Product Prep (element-wise multiply)
# ============================================================================

def test_dot_product_prep():
    print("\n" + "=" * 60)
    print("6. DOT PRODUCT (element-wise multiply)")
    print("=" * 60)
    
    program = CUDAProgram("""
@kernel
def elementwise_mul(a: Array[float32], b: Array[float32], c: Array[float32]):
    for i in range(len(a)):
        c[i] = a[i] * b[i]
""", debug=True)
    
    size = 50_000_000
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(a)
    
    def cpu_mul(a, b, c):
        np.multiply(a, b, out=c)
    
    benchmark("Element-wise Multiply (50M)", 
              cpu_mul, program, 'elementwise_mul',
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
    
    program = CUDAProgram("""
@kernel
def rgb_to_grayscale(r: Array[float32], g: Array[float32], b: Array[float32], out: Array[float32]):
    for i in range(len(r)):
        out[i] = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]
""", debug=True)
    
    # 4K image
    size = 3840 * 2160
    r = np.random.rand(size).astype(np.float32)
    g = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    out = np.zeros_like(r)
    
    def cpu_grayscale(r, g, b, out):
        out[:] = 0.299 * r + 0.587 * g + 0.114 * b
    
    benchmark("RGB to Grayscale (4K image)", 
              cpu_grayscale, program, 'rgb_to_grayscale',
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