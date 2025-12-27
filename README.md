# PyCUDA Compiler

> [!CAUTION]
> **PROPRIETARY / CONFIDENTIAL**. This software is a Research Preview.
> **Commercial use, redistribution, or derivation is STRICTLY PROHIBITED** without a written license.
> All Rights Reserved. (c) 2025 Ivan Shivalov.

Компилятор Python → CUDA, позволяющий писать GPU-код на чистом Python.

## Особенности

- ✅ **Автоматическая параллелизация** — циклы `for` превращаются в параллельные CUDA grid-stride loops
- ✅ **Классы → структуры** — Python классы компилируются в CUDA structs
- ✅ **Методы → device функции** — методы классов становятся `__device__` функциями
- ✅ **Вывод типов** — автоматическое определение типов переменных
- ✅ **Автоматический grid/block** — не нужно вручную указывать размеры
- ✅ **Рекурсия** — поддержка рекурсивных функций
- ✅ **Коллекции** — List, Dict, Set с ограничениями

## Установка

```bash
pip install cupy-cuda12x numpy
cp -r pycuda_compiler /your/project/
```

## Быстрый старт

```python
from pycuda_compiler import cuda_compile, Array, float32

@cuda_compile
def process(data: Array[float32]):
    for i in range(len(data)):
        if data[i] > 0:
            data[i] = sqrt(data[i])
        else:
            data[i] = data[i] * data[i]

data = process.array([4.0, -2.0, 9.0, -3.0])
process(data)
print(process.to_numpy(data))  # [2.0, 4.0, 3.0, 9.0]
```

## Расширенные возможности

### Рекурсия ✅

```python
def factorial(n: int32) -> int32:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int32) -> int32:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

@kernel
def compute(input: Array[int32], output: Array[int32]):
    for i in range(len(input)):
        output[i] = factorial(input[i])
```

### Списки (List) ✅

```python
@kernel
def filter_positive(data: Array[float32], result: List[float32]):
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i])
```

Методы: `append()`, `pop()`, `clear()`, `len()`, индексация `[]`

### Словари (Dict) ✅

```python
@kernel
def histogram(data: Array[int32], counts: Dict[int32, int32]):
    for i in range(len(data)):
        key = data[i]
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
```

Реализация: hash-таблица с open addressing (capacity=256 по умолчанию)

### Множества (Set) ✅

```python
@kernel
def unique_values(data: Array[int32], unique: Set[int32]):
    for i in range(len(data)):
        unique.add(data[i])
```

### Строки ✅

```python
name: str = "hello"  # char[64]
length = len(name)
```

### Кортежи ✅

```python
point: Tuple[float32, float32, float32] = (1.0, 2.0, 3.0)
```

### Optional ✅

```python
def find(arr: Array[float32], target: float32) -> Optional[int32]:
    for i in range(len(arr)):
        if arr[i] == target:
            return Some(i)
    return None
```

## Поддерживаемые конструкции

### Типы
```python
float32, float64, int32, int64, bool
Array[T], List[T], Dict[K,V], Set[T]
Optional[T], Tuple[T1, T2, ...]
```

### Операторы
```python
+, -, *, /, //, %, **          # арифметика
==, !=, <, <=, >, >=           # сравнения
and, or, not                    # логика
&, |, ^, ~, <<, >>             # битовые
+=, -=, *=, /=                 # составные
```

### Управление
```python
if/elif/else
for i in range(n)
while condition
break, continue
return
x if cond else y               # тернарный
```

### Математика
```python
sqrt, sin, cos, tan, exp, log, log2, log10
asin, acos, atan, atan2, sinh, cosh, tanh
pow, floor, ceil, abs, min, max
```

## Архитектура

```
Python → Parser → IR → Type Inference → CodeGen → CUDA C++ → nvrtc → GPU
```

| Файл | Описание |
|------|----------|
| `types_ir.py` | Типы и IR |
| `parser.py` | Python AST → IR |
| `type_inference.py` | Вывод типов |
| `codegen.py` | IR → CUDA |
| `extended_features.py` | List, Dict, Set, строки |
| `compiler.py` | API |
| `runtime.py` | GPU runtime |

## Ограничения

❌ **Не поддерживается:**
- Динамическое выделение памяти
- Исключения
- Импорты внутри kernel
- Глубокая рекурсия (лимит стека GPU ~1000)
- Строки произвольной длины

⚠️ **Ограничения коллекций:**
- List: фиксированный max_size (1024)
- Dict/Set: фиксированный capacity (256)
- Нет атомарных операций (race conditions в параллельном коде)

## Лицензия

## Лицензия

**PROPRIETARY SOURCE AVAILABLE**. See [LICENSE](LICENSE) for details.
Commercial use prohibited.
