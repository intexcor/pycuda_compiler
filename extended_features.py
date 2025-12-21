"""
PyCUDA Compiler - Extended Features

Расширения: рекурсия, замыкания, строки, словари.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import ast

from types_ir import (
    CUDAType, TypeKind, IRNode, IRNodeKind,
    FLOAT32, INT32, BOOL, VOID,
    IRFunctionDef, IRCall, IRVar, IRConst, IRStructDef,
    IRAssign, IRReturn, IRIf, IRFor, IRWhile,
    BinaryOp, CompareOp
)


# ============================================================================
# STRING SUPPORT
# ============================================================================

@dataclass
class CUDAString:
    """Строка фиксированной длины для CUDA."""
    max_length: int = 64
    
    def to_cuda_type(self) -> str:
        return f'char[{self.max_length}]'


@dataclass
class IRStringConst(IRNode):
    """Строковая константа."""
    kind: IRNodeKind = IRNodeKind.CONST
    value: str = ''
    max_length: int = 64


class StringSupport:
    """Генерация кода для строк."""
    
    @staticmethod
    def generate_string_helpers() -> str:
        """Генерирует вспомогательные функции для строк."""
        return '''
// String helpers
__device__ int cuda_strlen(const char* s) {
    int len = 0;
    while (s[len] != '\\0' && len < 256) len++;
    return len;
}

__device__ void cuda_strcpy(char* dst, const char* src, int max_len) {
    int i = 0;
    while (src[i] != '\\0' && i < max_len - 1) {
        dst[i] = src[i];
        i++;
    }
    dst[i] = '\\0';
}

__device__ int cuda_strcmp(const char* a, const char* b) {
    int i = 0;
    while (a[i] != '\\0' && b[i] != '\\0') {
        if (a[i] != b[i]) return a[i] - b[i];
        i++;
    }
    return a[i] - b[i];
}

__device__ void cuda_strcat(char* dst, const char* src, int max_len) {
    int dst_len = cuda_strlen(dst);
    int i = 0;
    while (src[i] != '\\0' && dst_len + i < max_len - 1) {
        dst[dst_len + i] = src[i];
        i++;
    }
    dst[dst_len + i] = '\\0';
}
'''


# ============================================================================
# DICTIONARY SUPPORT (Static Hash Table)
# ============================================================================

@dataclass 
class CUDADict:
    """Статический словарь для CUDA."""
    key_type: CUDAType = field(default_factory=lambda: INT32)
    value_type: CUDAType = field(default_factory=lambda: FLOAT32)
    capacity: int = 256
    
    def to_struct(self, name: str) -> str:
        key_cuda = 'int' if self.key_type == INT32 else 'float'
        val_cuda = 'float' if self.value_type == FLOAT32 else 'int'
        
        return f'''
struct {name} {{
    {key_cuda} keys[{self.capacity}];
    {val_cuda} values[{self.capacity}];
    bool occupied[{self.capacity}];
    int size;
}};
'''


class DictSupport:
    """Генерация кода для словарей."""
    
    @staticmethod
    def generate_dict_helpers(name: str, key_type: str = 'int', val_type: str = 'float', capacity: int = 256) -> str:
        """Генерирует структуру и функции для словаря."""
        return f'''
// Dictionary: {name}
struct {name} {{
    {key_type} keys[{capacity}];
    {val_type} values[{capacity}];
    bool occupied[{capacity}];
    int size;
}};

__device__ void {name}_init({name}* d) {{
    d->size = 0;
    for (int i = 0; i < {capacity}; i++) {{
        d->occupied[i] = false;
    }}
}}

__device__ unsigned int {name}_hash({key_type} key) {{
    // Simple hash function
    unsigned int h = (unsigned int)key;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = (h >> 16) ^ h;
    return h % {capacity};
}}

__device__ bool {name}_set({name}* d, {key_type} key, {val_type} value) {{
    unsigned int idx = {name}_hash(key);
    int attempts = 0;
    
    while (attempts < {capacity}) {{
        if (!d->occupied[idx] || d->keys[idx] == key) {{
            if (!d->occupied[idx]) {{
                d->size++;
            }}
            d->keys[idx] = key;
            d->values[idx] = value;
            d->occupied[idx] = true;
            return true;
        }}
        idx = (idx + 1) % {capacity};
        attempts++;
    }}
    return false;  // Full
}}

__device__ bool {name}_get({name}* d, {key_type} key, {val_type}* value) {{
    unsigned int idx = {name}_hash(key);
    int attempts = 0;
    
    while (attempts < {capacity}) {{
        if (!d->occupied[idx]) {{
            return false;  // Not found
        }}
        if (d->keys[idx] == key) {{
            *value = d->values[idx];
            return true;
        }}
        idx = (idx + 1) % {capacity};
        attempts++;
    }}
    return false;
}}

__device__ bool {name}_contains({name}* d, {key_type} key) {{
    {val_type} dummy;
    return {name}_get(d, key, &dummy);
}}

__device__ bool {name}_remove({name}* d, {key_type} key) {{
    unsigned int idx = {name}_hash(key);
    int attempts = 0;
    
    while (attempts < {capacity}) {{
        if (!d->occupied[idx]) {{
            return false;
        }}
        if (d->keys[idx] == key) {{
            d->occupied[idx] = false;
            d->size--;
            return true;
        }}
        idx = (idx + 1) % {capacity};
        attempts++;
    }}
    return false;
}}
'''


# ============================================================================
# CLOSURE SUPPORT
# ============================================================================

@dataclass
class CUDAClosure:
    """Замыкание для CUDA."""
    name: str = ''
    captured_vars: List[Tuple[str, CUDAType]] = field(default_factory=list)
    params: List[Tuple[str, CUDAType]] = field(default_factory=list)
    return_type: CUDAType = field(default_factory=lambda: VOID)
    body: List[IRNode] = field(default_factory=list)


class ClosureSupport:
    """Поддержка замыканий."""
    
    @staticmethod
    def transform_closure(closure: CUDAClosure) -> Tuple[str, str]:
        """
        Превращает замыкание в структуру + функцию.
        
        Возвращает: (struct_code, function_code)
        """
        struct_name = f'{closure.name}_closure'
        
        # Структура для захваченных переменных
        struct_lines = [f'struct {struct_name} {{']
        for var_name, var_type in closure.captured_vars:
            cuda_type = var_type.cuda_name if var_type else 'float'
            struct_lines.append(f'    {cuda_type} {var_name};')
        struct_lines.append('};')
        struct_code = '\n'.join(struct_lines)
        
        # Функция
        ret_type = closure.return_type.cuda_name if closure.return_type else 'void'
        
        params = [f'{struct_name}* __closure']
        for param_name, param_type in closure.params:
            cuda_type = param_type.cuda_name if param_type else 'float'
            params.append(f'{cuda_type} {param_name}')
        
        func_lines = [f'__device__ {ret_type} {closure.name}({", ".join(params)}) {{']
        
        # Разворачиваем захваченные переменные
        for var_name, _ in closure.captured_vars:
            func_lines.append(f'    auto {var_name} = __closure->{var_name};')
        
        func_lines.append('    // body here')
        func_lines.append('}')
        func_code = '\n'.join(func_lines)
        
        return struct_code, func_code


# ============================================================================
# RECURSION SUPPORT  
# ============================================================================

class RecursionSupport:
    """Поддержка рекурсии."""
    
    @staticmethod
    def add_stack_check(func_name: str, max_depth: int = 100) -> str:
        """Добавляет проверку глубины рекурсии."""
        return f'''
__device__ int __{func_name}_depth = 0;

#define CHECK_RECURSION_{func_name.upper()}() \\
    if (++__{func_name}_depth > {max_depth}) {{ \\
        __{func_name}_depth--; \\
        return; \\
    }}

#define END_RECURSION_{func_name.upper()}() \\
    __{func_name}_depth--;
'''
    
    @staticmethod
    def generate_recursive_function_example() -> str:
        """Пример рекурсивной функции."""
        return '''
// Recursive factorial
__device__ int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// Recursive fibonacci  
__device__ int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Recursive binary search
__device__ int binary_search(float* arr, float target, int low, int high) {
    if (low > high) return -1;
    
    int mid = (low + high) / 2;
    
    if (arr[mid] == target) return mid;
    if (arr[mid] > target) return binary_search(arr, target, low, mid - 1);
    return binary_search(arr, target, mid + 1, high);
}
'''


# ============================================================================
# LIST SUPPORT (Dynamic Array with pre-allocated buffer)
# ============================================================================

class ListSupport:
    """Поддержка списков (динамические массивы с фиксированной максимальной длиной)."""
    
    @staticmethod
    def generate_list_type(name: str, elem_type: str = 'float', max_size: int = 1024) -> str:
        return f'''
// Dynamic list: {name}
struct {name} {{
    {elem_type} data[{max_size}];
    int size;
    int capacity;
}};

__device__ void {name}_init({name}* list) {{
    list->size = 0;
    list->capacity = {max_size};
}}

__device__ bool {name}_append({name}* list, {elem_type} value) {{
    if (list->size >= list->capacity) return false;
    list->data[list->size++] = value;
    return true;
}}

__device__ {elem_type} {name}_get({name}* list, int index) {{
    if (index < 0 || index >= list->size) return ({elem_type})0;
    return list->data[index];
}}

__device__ bool {name}_set({name}* list, int index, {elem_type} value) {{
    if (index < 0 || index >= list->size) return false;
    list->data[index] = value;
    return true;
}}

__device__ {elem_type} {name}_pop({name}* list) {{
    if (list->size == 0) return ({elem_type})0;
    return list->data[--list->size];
}}

__device__ void {name}_clear({name}* list) {{
    list->size = 0;
}}

__device__ int {name}_len({name}* list) {{
    return list->size;
}}

__device__ bool {name}_contains({name}* list, {elem_type} value) {{
    for (int i = 0; i < list->size; i++) {{
        if (list->data[i] == value) return true;
    }}
    return false;
}}
'''


# ============================================================================
# SET SUPPORT
# ============================================================================

class SetSupport:
    """Поддержка множеств."""
    
    @staticmethod
    def generate_set_type(name: str, elem_type: str = 'int', capacity: int = 256) -> str:
        return f'''
// Set: {name}
struct {name} {{
    {elem_type} data[{capacity}];
    bool occupied[{capacity}];
    int size;
}};

__device__ void {name}_init({name}* s) {{
    s->size = 0;
    for (int i = 0; i < {capacity}; i++) {{
        s->occupied[i] = false;
    }}
}}

__device__ unsigned int {name}_hash({elem_type} value) {{
    unsigned int h = (unsigned int)value;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    return (h >> 16) ^ h;
}}

__device__ bool {name}_add({name}* s, {elem_type} value) {{
    unsigned int idx = {name}_hash(value) % {capacity};
    int attempts = 0;
    
    while (attempts < {capacity}) {{
        if (!s->occupied[idx]) {{
            s->data[idx] = value;
            s->occupied[idx] = true;
            s->size++;
            return true;
        }}
        if (s->data[idx] == value) {{
            return false;  // Already exists
        }}
        idx = (idx + 1) % {capacity};
        attempts++;
    }}
    return false;  // Full
}}

__device__ bool {name}_contains({name}* s, {elem_type} value) {{
    unsigned int idx = {name}_hash(value) % {capacity};
    int attempts = 0;
    
    while (attempts < {capacity}) {{
        if (!s->occupied[idx]) return false;
        if (s->data[idx] == value) return true;
        idx = (idx + 1) % {capacity};
        attempts++;
    }}
    return false;
}}

__device__ bool {name}_remove({name}* s, {elem_type} value) {{
    unsigned int idx = {name}_hash(value) % {capacity};
    int attempts = 0;
    
    while (attempts < {capacity}) {{
        if (!s->occupied[idx]) return false;
        if (s->data[idx] == value) {{
            s->occupied[idx] = false;
            s->size--;
            return true;
        }}
        idx = (idx + 1) % {capacity};
        attempts++;
    }}
    return false;
}}

__device__ int {name}_len({name}* s) {{
    return s->size;
}}
'''


# ============================================================================
# TUPLE SUPPORT
# ============================================================================

class TupleSupport:
    """Поддержка кортежей."""
    
    @staticmethod
    def generate_tuple_type(name: str, types: List[str]) -> str:
        """Генерирует структуру для кортежа."""
        lines = [f'struct {name} {{']
        for i, t in enumerate(types):
            lines.append(f'    {t} _{i};')
        lines.append('};')
        
        # Конструктор
        params = ', '.join(f'{t} v{i}' for i, t in enumerate(types))
        assigns = '\n'.join(f'    t._{i} = v{i};' for i in range(len(types)))
        
        lines.append(f'''
__device__ {name} {name}_make({params}) {{
    {name} t;
{assigns}
    return t;
}}''')
        
        return '\n'.join(lines)


# ============================================================================
# OPTIONAL/MAYBE SUPPORT
# ============================================================================

class OptionalSupport:
    """Поддержка Optional типов."""
    
    @staticmethod
    def generate_optional_type(name: str, inner_type: str) -> str:
        return f'''
struct {name} {{
    {inner_type} value;
    bool has_value;
}};

__device__ {name} {name}_some({inner_type} v) {{
    {name} opt;
    opt.value = v;
    opt.has_value = true;
    return opt;
}}

__device__ {name} {name}_none() {{
    {name} opt;
    opt.has_value = false;
    return opt;
}}

__device__ bool {name}_is_some({name}* opt) {{
    return opt->has_value;
}}

__device__ {inner_type} {name}_unwrap({name}* opt) {{
    return opt->value;
}}

__device__ {inner_type} {name}_unwrap_or({name}* opt, {inner_type} default_val) {{
    return opt->has_value ? opt->value : default_val;
}}
'''
