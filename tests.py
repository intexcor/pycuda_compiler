#!/usr/bin/env python3
"""
PyCUDA Compiler - Tests

Тесты для компилятора Python → CUDA.
Запуск: python3 tests.py
"""

import sys
sys.path.insert(0, '/home/claude/pycuda_compiler')

from compiler import CUDAProgram, cuda_compile, Array, float32
from parser import PythonParser
from type_inference import TypeInference
from codegen import CUDACodeGen

def test_simple_kernel():
    """Тест простого kernel."""
    print("Test: simple_kernel... ", end="")
    
    program = CUDAProgram("""
@kernel
def process(data: Array[float32]):
    for i in range(len(data)):
        data[i] = data[i] * 2.0
""")
    
    code = program.cuda_code
    assert '__global__' in code
    assert 'process' in code
    assert 'THREAD_ID' in code
    assert 'GRID_STRIDE' in code
    print("OK")


def test_conditionals():
    """Тест условных операторов."""
    print("Test: conditionals... ", end="")
    
    program = CUDAProgram("""
@kernel
def classify(x: Array[float32], y: Array[int32]):
    for i in range(len(x)):
        if x[i] < 0:
            y[i] = -1
        elif x[i] < 1:
            y[i] = 0
        else:
            y[i] = 1
""")
    
    code = program.cuda_code
    assert 'if' in code
    assert 'else' in code
    print("OK")


def test_while_loop():
    """Тест while цикла."""
    print("Test: while_loop... ", end="")
    
    program = CUDAProgram("""
@kernel
def iterate(x: Array[float32], max_iter: int32):
    for i in range(len(x)):
        count = 0
        while count < max_iter:
            x[i] = x[i] + 1.0
            count += 1
""")
    
    code = program.cuda_code
    assert 'while' in code
    print("OK")


def test_math_functions():
    """Тест математических функций."""
    print("Test: math_functions... ", end="")
    
    program = CUDAProgram("""
@kernel
def compute(x: Array[float32]):
    for i in range(len(x)):
        x[i] = sqrt(sin(x[i]) * cos(x[i])) + exp(log(abs(x[i]) + 1.0))
""")
    
    code = program.cuda_code
    assert 'sqrtf' in code
    assert 'sinf' in code
    assert 'cosf' in code
    assert 'expf' in code
    assert 'logf' in code
    assert 'fabsf' in code
    print("OK")


def test_struct():
    """Тест структур (классов)."""
    print("Test: struct... ", end="")
    
    program = CUDAProgram("""
class Point:
    x: float32
    y: float32

@kernel
def move_points(points: Array[Point], dx: float32, dy: float32):
    for i in range(len(points)):
        points[i].x += dx
        points[i].y += dy
""")
    
    code = program.cuda_code
    assert 'struct Point' in code
    assert 'float x;' in code
    assert 'float y;' in code
    assert 'points[i].x' in code
    print("OK")


def test_struct_with_methods():
    """Тест структур с методами."""
    print("Test: struct_with_methods... ", end="")
    
    program = CUDAProgram("""
class Particle:
    x: float32
    y: float32
    vx: float32
    vy: float32
    
    def update(self, dt: float32):
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def speed(self) -> float32:
        return sqrt(self.vx * self.vx + self.vy * self.vy)

@kernel
def simulate(particles: Array[Particle], dt: float32):
    for i in range(len(particles)):
        particles[i].x += particles[i].vx * dt
""")
    
    code = program.cuda_code
    assert 'struct Particle' in code
    assert '__device__' in code
    assert 'Particle_update' in code
    assert 'Particle_speed' in code
    print("OK")


def test_multiple_arrays():
    """Тест с несколькими массивами."""
    print("Test: multiple_arrays... ", end="")
    
    program = CUDAProgram("""
@kernel
def add_vectors(a: Array[float32], b: Array[float32], c: Array[float32]):
    for i in range(len(a)):
        c[i] = a[i] + b[i]
""")
    
    code = program.cuda_code
    assert 'float* a' in code
    assert 'float* b' in code
    assert 'float* c' in code
    assert '_size_a' in code
    assert '_size_b' in code
    print("OK")


def test_ternary():
    """Тест тернарного оператора."""
    print("Test: ternary... ", end="")
    
    program = CUDAProgram("""
@kernel
def abs_val(x: Array[float32]):
    for i in range(len(x)):
        x[i] = x[i] if x[i] > 0 else -x[i]
""")
    
    code = program.cuda_code
    assert '?' in code
    assert ':' in code
    print("OK")


def test_comparison_chain():
    """Тест цепочки сравнений."""
    print("Test: comparison_chain... ", end="")
    
    program = CUDAProgram("""
@kernel
def in_range(x: Array[float32], flags: Array[int32]):
    for i in range(len(x)):
        if 0 < x[i] < 1:
            flags[i] = 1
        else:
            flags[i] = 0
""")
    
    code = program.cuda_code
    assert '&&' in code  # цепочка превращается в AND
    print("OK")


def test_bitwise_ops():
    """Тест битовых операций."""
    print("Test: bitwise_ops... ", end="")
    
    program = CUDAProgram("""
@kernel
def bitwise(a: Array[int32], b: Array[int32], c: Array[int32]):
    for i in range(len(a)):
        c[i] = (a[i] & b[i]) | (a[i] ^ b[i])
""")
    
    code = program.cuda_code
    assert '&' in code
    assert '|' in code
    assert '^' in code
    print("OK")


def test_power_operator():
    """Тест оператора возведения в степень."""
    print("Test: power_operator... ", end="")
    
    program = CUDAProgram("""
@kernel
def power(x: Array[float32], exp: float32):
    for i in range(len(x)):
        x[i] = x[i] ** exp
""")
    
    code = program.cuda_code
    assert 'powf' in code
    print("OK")


def test_floor_division():
    """Тест целочисленного деления."""
    print("Test: floor_division... ", end="")
    
    program = CUDAProgram("""
@kernel
def floor_div(x: Array[int32], divisor: int32):
    for i in range(len(x)):
        x[i] = x[i] // divisor
""")
    
    code = program.cuda_code
    assert '(int)' in code
    print("OK")


def test_nested_loops():
    """Тест вложенных циклов."""
    print("Test: nested_loops... ", end="")
    
    program = CUDAProgram("""
@kernel
def matrix_op(data: Array[float32], n: int32, m: int32):
    for i in range(len(data)):
        sum_val: float32 = 0.0
        for j in range(m):
            sum_val += j
        data[i] = sum_val
""")
    
    code = program.cuda_code
    assert 'for (int i = THREAD_ID' in code
    assert 'for (int j = 0' in code
    print("OK")


def test_type_inference():
    """Тест вывода типов."""
    print("Test: type_inference... ", end="")
    
    program = CUDAProgram("""
@kernel
def infer_types(x: Array[float32]):
    for i in range(len(x)):
        a = x[i]       # float from array
        b = 1          # int
        c = 1.0        # float
        d = a + b      # float (promotion)
        e = b + 1      # int
        f = c * d      # float
        x[i] = f
""")
    
    code = program.cuda_code
    # Должны быть правильные типы
    assert 'float' in code
    assert 'int' in code
    print("OK")


def test_boolean_ops():
    """Тест логических операций."""
    print("Test: boolean_ops... ", end="")
    
    program = CUDAProgram("""
@kernel
def logic(x: Array[float32], flags: Array[int32]):
    for i in range(len(x)):
        if x[i] > 0 and x[i] < 1:
            flags[i] = 1
        elif x[i] <= 0 or x[i] >= 1:
            flags[i] = 0
""")
    
    code = program.cuda_code
    assert '&&' in code
    assert '||' in code
    print("OK")


def test_unary_ops():
    """Тест унарных операций."""
    print("Test: unary_ops... ", end="")
    
    program = CUDAProgram("""
@kernel
def unary(x: Array[float32], flags: Array[int32]):
    for i in range(len(x)):
        x[i] = -x[i]
        if not (x[i] > 0):
            flags[i] = 1
""")
    
    code = program.cuda_code
    assert '(-' in code
    assert '!' in code
    print("OK")


def test_aug_assign():
    """Тест составного присваивания."""
    print("Test: aug_assign... ", end="")
    
    program = CUDAProgram("""
@kernel
def aug(x: Array[float32], factor: float32):
    for i in range(len(x)):
        x[i] += 1.0
        x[i] -= 0.5
        x[i] *= factor
        x[i] /= 2.0
""")
    
    code = program.cuda_code
    assert '+=' in code
    assert '-=' in code
    assert '*=' in code
    assert '/=' in code
    print("OK")


def test_break_continue():
    """Тест break и continue."""
    print("Test: break_continue... ", end="")
    
    program = CUDAProgram("""
@kernel
def early_exit(x: Array[float32]):
    for i in range(len(x)):
        count = 0
        while count < 100:
            if x[i] > 10:
                break
            if count % 2 == 0:
                count += 1
                continue
            x[i] += 1.0
            count += 1
""")
    
    code = program.cuda_code
    assert 'break;' in code
    assert 'continue;' in code
    print("OK")


def test_multiple_kernels():
    """Тест нескольких kernels."""
    print("Test: multiple_kernels... ", end="")
    
    program = CUDAProgram("""
@kernel
def kernel1(x: Array[float32]):
    for i in range(len(x)):
        x[i] = x[i] * 2.0

@kernel
def kernel2(x: Array[float32], y: Array[float32]):
    for i in range(len(x)):
        y[i] = x[i] + 1.0
""")
    
    code = program.cuda_code
    assert 'void kernel1' in code
    assert 'void kernel2' in code
    print("OK")


def test_device_function():
    """Тест device функций."""
    print("Test: device_function... ", end="")
    
    program = CUDAProgram("""
def helper(x: float32) -> float32:
    return x * x + 1.0

@kernel
def use_helper(data: Array[float32]):
    for i in range(len(data)):
        data[i] = helper(data[i])
""")
    
    code = program.cuda_code
    assert '__device__' in code
    assert 'helper(' in code
    print("OK")


def run_all_tests():
    """Запускает все тесты."""
    print("=" * 60)
    print("Running PyCUDA Compiler Tests")
    print("=" * 60)
    
    tests = [
        test_simple_kernel,
        test_conditionals,
        test_while_loop,
        test_math_functions,
        test_struct,
        test_struct_with_methods,
        test_multiple_arrays,
        test_ternary,
        test_comparison_chain,
        test_bitwise_ops,
        test_power_operator,
        test_floor_division,
        test_nested_loops,
        test_type_inference,
        test_boolean_ops,
        test_unary_ops,
        test_aug_assign,
        test_break_continue,
        test_multiple_kernels,
        test_device_function,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAILED: {repr(e)}")
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
