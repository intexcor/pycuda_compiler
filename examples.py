#!/usr/bin/env python3
"""
PyCUDA Compiler - Examples

Примеры использования компилятора Python → CUDA.
"""

# ============================================================================
# Example 1: Simple kernel with decorator
# ============================================================================

example1_simple = '''
from pycuda_compiler import cuda_compile, Array, float32

@cuda_compile
def process_data(data: Array[float32]):
    """Обрабатывает каждый элемент массива."""
    for i in range(len(data)):
        if data[i] > 0:
            data[i] = sqrt(data[i])
        else:
            data[i] = data[i] * data[i]

# Использование:
data = process_data.array([4.0, -2.0, 9.0, -3.0, 16.0])
process_data(data)
print(process_data.to_numpy(data))
# Output: [2.0, 4.0, 3.0, 9.0, 4.0]
'''


# ============================================================================
# Example 2: Multiple arrays
# ============================================================================

example2_multiple_arrays = '''
from pycuda_compiler import cuda_compile, Array, float32

@cuda_compile
def vector_add(a: Array[float32], b: Array[float32], c: Array[float32]):
    """Складывает два вектора: c = a + b"""
    for i in range(len(a)):
        c[i] = a[i] + b[i]

@cuda_compile  
def vector_mul(a: Array[float32], b: Array[float32], c: Array[float32]):
    """Умножает два вектора поэлементно: c = a * b"""
    for i in range(len(a)):
        c[i] = a[i] * b[i]

# Использование:
import cupy as cp
a = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
b = cp.array([4.0, 5.0, 6.0], dtype=cp.float32)
c = cp.zeros(3, dtype=cp.float32)

vector_add(a, b, c)
print(cp.asnumpy(c))  # [5.0, 7.0, 9.0]
'''


# ============================================================================
# Example 3: Complex math
# ============================================================================

example3_math = '''
from pycuda_compiler import cuda_compile, Array, float32

@cuda_compile
def apply_sigmoid(x: Array[float32]):
    """Применяет сигмоид: sigmoid(x) = 1 / (1 + exp(-x))"""
    for i in range(len(x)):
        x[i] = 1.0 / (1.0 + exp(-x[i]))

@cuda_compile
def apply_relu(x: Array[float32]):
    """Применяет ReLU: max(0, x)"""
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0.0

@cuda_compile
def apply_softplus(x: Array[float32]):
    """Применяет Softplus: log(1 + exp(x))"""
    for i in range(len(x)):
        x[i] = log(1.0 + exp(x[i]))
'''


# ============================================================================
# Example 4: Classes (structs)
# ============================================================================

example4_classes = '''
from pycuda_compiler import CUDAProgram

program = CUDAProgram("""
class Particle:
    x: float32
    y: float32
    vx: float32
    vy: float32

def update_position(p: Particle, dt: float32):
    p.x += p.vx * dt
    p.y += p.vy * dt

def compute_distance(p1: Particle, p2: Particle) -> float32:
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return sqrt(dx * dx + dy * dy)

@kernel
def simulate_step(particles: Array[Particle], dt: float32):
    for i in range(len(particles)):
        update_position(particles[i], dt)
""")

# Показать сгенерированный CUDA код
program.show_cuda()
'''


# ============================================================================
# Example 5: Physics simulation
# ============================================================================

example5_physics = '''
from pycuda_compiler import CUDAProgram
import cupy as cp
import numpy as np

program = CUDAProgram("""
class Body:
    x: float32
    y: float32
    vx: float32
    vy: float32
    mass: float32

@kernel
def compute_gravity(bodies: Array[Body], forces_x: Array[float32], forces_y: Array[float32]):
    G: float32 = 6.674e-11
    softening: float32 = 1e-9
    
    for i in range(len(bodies)):
        fx: float32 = 0.0
        fy: float32 = 0.0
        
        for j in range(len(bodies)):
            if i != j:
                dx = bodies[j].x - bodies[i].x
                dy = bodies[j].y - bodies[i].y
                dist_sq = dx * dx + dy * dy + softening
                dist = sqrt(dist_sq)
                force = G * bodies[i].mass * bodies[j].mass / dist_sq
                
                fx += force * dx / dist
                fy += force * dy / dist
        
        forces_x[i] = fx
        forces_y[i] = fy

@kernel
def update_velocities(bodies: Array[Body], forces_x: Array[float32], forces_y: Array[float32], dt: float32):
    for i in range(len(bodies)):
        ax = forces_x[i] / bodies[i].mass
        ay = forces_y[i] / bodies[i].mass
        bodies[i].vx += ax * dt
        bodies[i].vy += ay * dt

@kernel
def update_positions(bodies: Array[Body], dt: float32):
    for i in range(len(bodies)):
        bodies[i].x += bodies[i].vx * dt
        bodies[i].y += bodies[i].vy * dt
""")

program.show_cuda()
'''


# ============================================================================
# Example 6: Image processing
# ============================================================================

example6_image = '''
from pycuda_compiler import CUDAProgram

program = CUDAProgram("""
@kernel
def grayscale(r: Array[float32], g: Array[float32], b: Array[float32], out: Array[float32]):
    \"\"\"Конвертирует RGB в grayscale.\"\"\"
    for i in range(len(r)):
        out[i] = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]

@kernel
def brightness(pixels: Array[float32], factor: float32):
    \"\"\"Изменяет яркость.\"\"\"
    for i in range(len(pixels)):
        pixels[i] = pixels[i] * factor
        if pixels[i] > 1.0:
            pixels[i] = 1.0
        if pixels[i] < 0.0:
            pixels[i] = 0.0

@kernel
def contrast(pixels: Array[float32], factor: float32):
    \"\"\"Изменяет контраст.\"\"\"
    for i in range(len(pixels)):
        pixels[i] = (pixels[i] - 0.5) * factor + 0.5
        if pixels[i] > 1.0:
            pixels[i] = 1.0
        if pixels[i] < 0.0:
            pixels[i] = 0.0

@kernel
def threshold(pixels: Array[float32], thresh: float32):
    \"\"\"Бинаризация.\"\"\"
    for i in range(len(pixels)):
        if pixels[i] > thresh:
            pixels[i] = 1.0
        else:
            pixels[i] = 0.0
""")

program.show_cuda()
'''


# ============================================================================
# Example 7: Reduction (sum)
# ============================================================================

example7_reduction = '''
from pycuda_compiler import CUDAProgram

program = CUDAProgram("""
@kernel
def partial_sum(data: Array[float32], partial: Array[float32], chunk_size: int32):
    \"\"\"
    Вычисляет частичные суммы для редукции.
    Каждый поток суммирует chunk_size элементов.
    \"\"\"
    for i in range(len(partial)):
        start = i * chunk_size
        end = start + chunk_size
        if end > len(data):
            end = len(data)
        
        sum_val: float32 = 0.0
        for j in range(start, end):
            sum_val += data[j]
        
        partial[i] = sum_val
""")

program.show_cuda()
'''


# ============================================================================
# Example 8: Monte Carlo Pi estimation
# ============================================================================

example8_monte_carlo = '''
from pycuda_compiler import CUDAProgram

program = CUDAProgram("""
@kernel
def monte_carlo_pi(random_x: Array[float32], random_y: Array[float32], inside: Array[int32]):
    \"\"\"
    Оценка числа Pi методом Монте-Карло.
    inside[i] = 1 если точка внутри единичного круга.
    \"\"\"
    for i in range(len(random_x)):
        x = random_x[i]
        y = random_y[i]
        dist_sq = x * x + y * y
        
        if dist_sq <= 1.0:
            inside[i] = 1
        else:
            inside[i] = 0
""")

# Использование:
# import cupy as cp
# n = 10_000_000
# x = cp.random.uniform(0, 1, n).astype(cp.float32)
# y = cp.random.uniform(0, 1, n).astype(cp.float32)
# inside = cp.zeros(n, dtype=cp.int32)
# 
# program.run('monte_carlo_pi', x, y, inside)
# pi_estimate = 4.0 * inside.sum() / n
# print(f"Pi ≈ {pi_estimate}")

program.show_cuda()
'''


# ============================================================================
# Example 9: Neural network layer
# ============================================================================

example9_neural = '''
from pycuda_compiler import CUDAProgram

program = CUDAProgram("""
@kernel
def dense_forward(
    input_data: Array[float32],
    weights: Array[float32],
    bias: Array[float32],
    output: Array[float32],
    input_size: int32,
    output_size: int32,
    batch_size: int32
):
    \"\"\"
    Прямой проход полносвязного слоя.
    input_data: [batch_size, input_size]
    weights: [input_size, output_size]
    bias: [output_size]
    output: [batch_size, output_size]
    \"\"\"
    for idx in range(len(output)):
        batch_idx = idx / output_size
        out_idx = idx % output_size
        
        sum_val: float32 = bias[out_idx]
        
        for in_idx in range(input_size):
            input_val = input_data[batch_idx * input_size + in_idx]
            weight_val = weights[in_idx * output_size + out_idx]
            sum_val += input_val * weight_val
        
        # ReLU activation
        if sum_val < 0.0:
            sum_val = 0.0
        
        output[idx] = sum_val
""")

program.show_cuda()
'''


# ============================================================================
# Example 10: Custom DSP
# ============================================================================

example10_dsp = '''
from pycuda_compiler import CUDAProgram

program = CUDAProgram("""
@kernel
def apply_lowpass(
    signal: Array[float32],
    output: Array[float32],
    alpha: float32
):
    \"\"\"
    Простой низкочастотный фильтр первого порядка.
    y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    \"\"\"
    for i in range(len(signal)):
        if i == 0:
            output[i] = signal[i]
        else:
            output[i] = alpha * signal[i] + (1.0 - alpha) * output[i - 1]

@kernel
def apply_gain(signal: Array[float32], gain: float32):
    \"\"\"Применяет усиление к сигналу.\"\"\"
    for i in range(len(signal)):
        signal[i] = signal[i] * gain

@kernel
def mix_signals(a: Array[float32], b: Array[float32], out: Array[float32], mix: float32):
    \"\"\"Смешивает два сигнала: out = a * (1-mix) + b * mix\"\"\"
    for i in range(len(a)):
        out[i] = a[i] * (1.0 - mix) + b[i] * mix

@kernel
def apply_distortion(signal: Array[float32], threshold: float32):
    \"\"\"Применяет жёсткий клиппинг (distortion).\"\"\"
    for i in range(len(signal)):
        if signal[i] > threshold:
            signal[i] = threshold
        if signal[i] < -threshold:
            signal[i] = -threshold
""")

program.show_cuda()
'''


# ============================================================================
# Run examples
# ============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/claude/pycuda_compiler')
    
    from compiler import CUDAProgram
    
    print("=" * 70)
    print("Example 1: Simple kernel")
    print("=" * 70)
    
    program = CUDAProgram("""
@kernel
def process_data(data: Array[float32]):
    for i in range(len(data)):
        if data[i] > 0:
            data[i] = sqrt(data[i])
        else:
            data[i] = data[i] * data[i]
""")
    program.show_cuda()
    
    print("\n" + "=" * 70)
    print("Example 2: Classes and methods")
    print("=" * 70)
    
    program2 = CUDAProgram("""
class Particle:
    x: float32
    y: float32
    vx: float32
    vy: float32

@kernel
def update_particles(particles: Array[Particle], dt: float32):
    for i in range(len(particles)):
        particles[i].x += particles[i].vx * dt
        particles[i].y += particles[i].vy * dt
""")
    program2.show_cuda()
    
    print("\n" + "=" * 70)
    print("Example 3: Complex logic")
    print("=" * 70)
    
    program3 = CUDAProgram("""
@kernel
def classify(values: Array[float32], labels: Array[int32]):
    for i in range(len(values)):
        v = values[i]
        
        if v < 0:
            labels[i] = -1
        elif v < 0.5:
            labels[i] = 0
        elif v < 1.0:
            labels[i] = 1
        else:
            labels[i] = 2
""")
    program3.show_cuda()
    
    print("\n" + "=" * 70)
    print("Example 4: While loop")
    print("=" * 70)
    
    program4 = CUDAProgram("""
@kernel
def newton_sqrt(values: Array[float32], iterations: int32):
    for i in range(len(values)):
        x = values[i]
        if x > 0:
            guess = x / 2.0
            count = 0
            
            while count < iterations:
                guess = (guess + x / guess) / 2.0
                count += 1
            
            values[i] = guess
        else:
            values[i] = 0.0
""")
    program4.show_cuda()
    
    print("\n" + "=" * 70)
    print("Example 5: Math functions")
    print("=" * 70)
    
    program5 = CUDAProgram("""
@kernel
def compute_wave(x: Array[float32], freq: float32, phase: float32):
    for i in range(len(x)):
        t = x[i]
        x[i] = sin(2.0 * 3.14159 * freq * t + phase) * exp(-t * 0.1)
""")
    program5.show_cuda()
    
    print("\n" + "=" * 70)
    print("All examples compiled successfully!")
    print("=" * 70)
