import cupy as cp
import os

# 1. Загружаем скомпилированный PTX модуль
# Указываем путь к файлу kernel.ptx
ptx_file = os.path.join(os.path.dirname(__file__), 'kernel.ptx')
module = cp.RawModule(path=ptx_file)

# 2. Получаем функцию ядра по имени
# Имя функции такое же, как в Python коде (или в output.cu)
kernel_name = 'square_root_kernel'
kernel = module.get_function(kernel_name)

# 3. Подготавливаем данные
N = 4
data = cp.array([1.0, 4.0, 9.0, 16.0], dtype=cp.float32)
result = cp.zeros_like(data)

# 4. Запускаем ядро
# Аргументы: data pointer, size_data, result pointer, size_result
# Обратите внимание: args должны быть ctypes pointers или numpy/cupy scalars
args = (
    data,               # float* data
    cp.int32(data.size),# int _size_data
    result,             # float* result
    cp.int32(result.size) # int _size_result
)

# Grid/Block расчет
block_size = 256
grid_size = (N + block_size - 1) // block_size

print(f"Running kernel '{kernel_name}' with grid={grid_size}, block={block_size}...")
kernel((grid_size,), (block_size,), args)

# 5. Результат
print("Input:", data)
print("Result:", result)
