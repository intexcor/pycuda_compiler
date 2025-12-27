"""
PyCUDA Compiler - Extended Parser

Расширенный парсер с поддержкой:
- Рекурсии
- Замыканий (lambda)
- Строк
- Словарей
- Множеств
- Списков (append, pop)
"""

from __future__ import annotations
import ast
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

from types_ir import (
    CUDAType, TypeKind, 
    FLOAT32,
    IRNode
)

from extended_features import (
    StringSupport, DictSupport, ListSupport, 
    SetSupport, TupleSupport, OptionalSupport,
    RecursionSupport
)


# ============================================================================
# Extended Types
# ============================================================================

# Строки
STRING = CUDAType(TypeKind.STRUCT, 'string', 'char*', 8)

# Новые типы для коллекций
def make_list_type(elem_type: CUDAType, max_size: int = 1024) -> CUDAType:
    return CUDAType(
        kind=TypeKind.STRUCT,
        name=f'List[{elem_type.name}]',
        cuda_name=f'List_{elem_type.name}',
        element_type=elem_type
    )

def make_dict_type(key_type: CUDAType, val_type: CUDAType, capacity: int = 256) -> CUDAType:
    return CUDAType(
        kind=TypeKind.STRUCT,
        name=f'Dict[{key_type.name}, {val_type.name}]',
        cuda_name=f'Dict_{key_type.name}_{val_type.name}'
    )

def make_set_type(elem_type: CUDAType, capacity: int = 256) -> CUDAType:
    return CUDAType(
        kind=TypeKind.STRUCT,
        name=f'Set[{elem_type.name}]',
        cuda_name=f'Set_{elem_type.name}',
        element_type=elem_type
    )

def make_optional_type(inner_type: CUDAType) -> CUDAType:
    return CUDAType(
        kind=TypeKind.STRUCT,
        name=f'Optional[{inner_type.name}]',
        cuda_name=f'Optional_{inner_type.name}',
        element_type=inner_type
    )


# ============================================================================
# Extended IR Nodes
# ============================================================================

@dataclass
class IRLambda(IRNode):
    """Lambda выражение."""
    params: List[Tuple[str, CUDAType]] = field(default_factory=list)
    body: IRNode = None
    captured_vars: List[str] = field(default_factory=list)


@dataclass
class IRListComprehension(IRNode):
    """List comprehension."""
    element_expr: IRNode = None
    var_name: str = ''
    iterable: IRNode = None
    condition: Optional[IRNode] = None


@dataclass
class IRDictLiteral(IRNode):
    """Словарь."""
    keys: List[IRNode] = field(default_factory=list)
    values: List[IRNode] = field(default_factory=list)


@dataclass
class IRSetLiteral(IRNode):
    """Множество."""
    elements: List[IRNode] = field(default_factory=list)


@dataclass
class IRStringLiteral(IRNode):
    """Строковый литерал."""
    value: str = ''


# ============================================================================
# Extended Code Generator
# ============================================================================

class ExtendedCodeGen:
    """Расширенный генератор кода."""
    
    def __init__(self):
        self.required_helpers: Set[str] = set()
        self.generated_types: Dict[str, str] = {}
    
    def generate_helpers(self) -> str:
        """Генерирует все необходимые вспомогательные функции."""
        code = []
        
        if 'string' in self.required_helpers:
            code.append(StringSupport.generate_string_helpers())
        
        if 'recursion' in self.required_helpers:
            code.append(RecursionSupport.generate_recursive_function_example())
        
        for type_name, type_code in self.generated_types.items():
            code.append(type_code)
        
        return '\n\n'.join(code)
    
    def ensure_list_type(self, elem_type: str, max_size: int = 1024) -> str:
        """Гарантирует, что тип списка сгенерирован."""
        type_name = f'List_{elem_type}'
        if type_name not in self.generated_types:
            self.generated_types[type_name] = ListSupport.generate_list_type(
                type_name, elem_type, max_size
            )
        return type_name
    
    def ensure_dict_type(self, key_type: str, val_type: str, capacity: int = 256) -> str:
        """Гарантирует, что тип словаря сгенерирован."""
        type_name = f'Dict_{key_type}_{val_type}'
        if type_name not in self.generated_types:
            self.generated_types[type_name] = DictSupport.generate_dict_helpers(
                type_name, key_type, val_type, capacity
            )
        return type_name
    
    def ensure_set_type(self, elem_type: str, capacity: int = 256) -> str:
        """Гарантирует, что тип множества сгенерирован."""
        type_name = f'Set_{elem_type}'
        if type_name not in self.generated_types:
            self.generated_types[type_name] = SetSupport.generate_set_type(
                type_name, elem_type, capacity
            )
        return type_name
    
    def ensure_optional_type(self, inner_type: str) -> str:
        """Гарантирует, что Optional тип сгенерирован."""
        type_name = f'Optional_{inner_type}'
        if type_name not in self.generated_types:
            self.generated_types[type_name] = OptionalSupport.generate_optional_type(
                type_name, inner_type
            )
        return type_name
    
    def ensure_tuple_type(self, types: List[str]) -> str:
        """Гарантирует, что тип кортежа сгенерирован."""
        type_name = 'Tuple_' + '_'.join(types)
        if type_name not in self.generated_types:
            self.generated_types[type_name] = TupleSupport.generate_tuple_type(
                type_name, types
            )
        return type_name
    
    # ========== Генерация кода для новых конструкций ==========
    
    def gen_lambda(self, lambda_node: IRLambda, closure_id: int) -> Tuple[str, str]:
        """
        Генерирует код для lambda.
        Возвращает: (struct_code, использование)
        """
        closure_name = f'__closure_{closure_id}'
        
        # Структура для захваченных переменных
        struct_lines = [f'struct {closure_name}_t {{']
        for var in lambda_node.captured_vars:
            struct_lines.append(f'    float {var};  // captured')
        struct_lines.append('};')
        
        # Функция
        params = [f'{closure_name}_t* __ctx']
        for pname, ptype in lambda_node.params:
            cuda_type = ptype.cuda_name if ptype else 'float'
            params.append(f'{cuda_type} {pname}')
        
        func_lines = [
            f'__device__ float {closure_name}_call({", ".join(params)}) {{'
        ]
        # Разворачиваем captured
        for var in lambda_node.captured_vars:
            func_lines.append(f'    float {var} = __ctx->{var};')
        
        # Body (пока упрощённо)
        func_lines.append('    // lambda body')
        func_lines.append('    return 0.0f;')
        func_lines.append('}')
        
        struct_code = '\n'.join(struct_lines)
        func_code = '\n'.join(func_lines)
        
        return struct_code + '\n' + func_code, closure_name
    
    def gen_list_comprehension(self, comp: IRListComprehension) -> str:
        """
        Генерирует код для list comprehension.
        [expr for x in iterable if cond] → цикл
        """
        lines = []
        lines.append('{')
        lines.append('    int __idx = 0;')
        lines.append(f'    for (int {comp.var_name} = 0; {comp.var_name} < __size; {comp.var_name}++) {{')
        
        if comp.condition:
            lines.append('        if (/* condition */) {')
            lines.append('            __result[__idx++] = /* expr */;')
            lines.append('        }')
        else:
            lines.append('        __result[__idx++] = /* expr */;')
        
        lines.append('    }')
        lines.append('}')
        
        return '\n'.join(lines)
    
    def gen_dict_operations(self, dict_type: str) -> Dict[str, str]:
        """Возвращает маппинг Python операций → CUDA вызовы."""
        return {
            '__setitem__': f'{dict_type}_set',
            '__getitem__': f'{dict_type}_get',
            '__contains__': f'{dict_type}_contains',
            '__delitem__': f'{dict_type}_remove',
            'get': f'{dict_type}_get',
            'keys': f'{dict_type}_keys',
            'values': f'{dict_type}_values',
        }
    
    def gen_list_operations(self, list_type: str) -> Dict[str, str]:
        """Возвращает маппинг Python операций → CUDA вызовы."""
        return {
            'append': f'{list_type}_append',
            'pop': f'{list_type}_pop',
            '__getitem__': f'{list_type}_get',
            '__setitem__': f'{list_type}_set',
            '__len__': f'{list_type}_len',
            'clear': f'{list_type}_clear',
            '__contains__': f'{list_type}_contains',
        }
    
    def gen_set_operations(self, set_type: str) -> Dict[str, str]:
        """Возвращает маппинг Python операций → CUDA вызовы."""
        return {
            'add': f'{set_type}_add',
            'remove': f'{set_type}_remove',
            '__contains__': f'{set_type}_contains',
            '__len__': f'{set_type}_len',
        }


# ============================================================================
# Extended Parser Mixin
# ============================================================================

class ExtendedParserMixin:
    """Миксин для расширенного парсинга."""
    
    def __init__(self):
        self.lambdas: List[IRLambda] = []
        self.lambda_counter = 0
        self.extended_codegen = ExtendedCodeGen()
    
    def parse_lambda(self, node: ast.Lambda) -> IRLambda:
        """Парсит lambda выражение."""
        params = []
        for arg in node.args.args:
            params.append((arg.arg, FLOAT32))  # default type
        
        # Находим captured переменные
        captured = self._find_captured_vars(node.body, set(p[0] for p in params))
        
        lambda_node = IRLambda(
            params=params,
            captured_vars=list(captured)
        )
        
        self.lambdas.append(lambda_node)
        self.lambda_counter += 1
        
        return lambda_node
    
    def _find_captured_vars(self, node: ast.AST, local_vars: Set[str]) -> Set[str]:
        """Находит переменные, захваченные замыканием."""
        captured = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id not in local_vars and child.id not in dir(__builtins__):
                    captured.add(child.id)
        
        return captured
    
    def parse_dict_literal(self, node: ast.Dict) -> IRDictLiteral:
        """Парсит словарь."""
        keys = [self.visit_expr(k) for k in node.keys if k is not None]
        values = [self.visit_expr(v) for v in node.values]
        
        return IRDictLiteral(keys=keys, values=values)
    
    def parse_set_literal(self, node: ast.Set) -> IRSetLiteral:
        """Парсит множество."""
        elements = [self.visit_expr(e) for e in node.elts]
        return IRSetLiteral(elements=elements)
    
    def parse_list_comprehension(self, node: ast.ListComp) -> IRListComprehension:
        """Парсит list comprehension."""
        # Поддерживаем только простые comprehensions
        if len(node.generators) != 1:
            raise NotImplementedError("Only single generator comprehensions supported")
        
        gen = node.generators[0]
        
        return IRListComprehension(
            element_expr=self.visit_expr(node.elt),
            var_name=gen.target.id if isinstance(gen.target, ast.Name) else '_',
            iterable=self.visit_expr(gen.iter),
            condition=self.visit_expr(gen.ifs[0]) if gen.ifs else None
        )
    
    def parse_string_literal(self, node: ast.Constant) -> IRStringLiteral:
        """Парсит строковый литерал."""
        if isinstance(node.value, str):
            return IRStringLiteral(value=node.value)
        raise ValueError("Not a string literal")


# ============================================================================
# Full Example: Extended Compiler Usage
# ============================================================================

EXTENDED_EXAMPLE = '''
# Example with extended features

from pycuda_compiler import CUDAProgram, cuda_compile

# 1. Recursion
program = CUDAProgram("""
def factorial(n: int32) -> int32:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int32) -> int32:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

@kernel
def compute_factorials(input: Array[int32], output: Array[int32]):
    for i in range(len(input)):
        output[i] = factorial(input[i])
""")

# 2. Lists (dynamic arrays)
program2 = CUDAProgram("""
@kernel
def filter_positive(data: Array[float32], result: List[float32]):
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i])
""")

# 3. Dictionaries
program3 = CUDAProgram("""
@kernel  
def count_occurrences(data: Array[int32], counts: Dict[int32, int32]):
    for i in range(len(data)):
        key = data[i]
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
""")

# 4. Sets
program4 = CUDAProgram("""
@kernel
def find_unique(data: Array[int32], unique: Set[int32]):
    for i in range(len(data)):
        unique.add(data[i])
""")
'''


# ============================================================================
# Code Generation Integration
# ============================================================================

def generate_extended_cuda_code(
    module,  # IRModule
    features: Set[str] = None
) -> str:
    """
    Генерирует CUDA код с поддержкой расширенных возможностей.
    
    features: {'list', 'dict', 'set', 'string', 'recursion', 'lambda'}
    """
    if features is None:
        features = set()
    

    lines = []
    lines.append('// Generated by PyCUDA Compiler (Extended)')
    lines.append('#include <cuda_runtime.h>')
    lines.append('#include <math.h>')
    lines.append('')
    
    # Вспомогательные функции
    if 'string' in features:
        lines.append(StringSupport.generate_string_helpers())
        lines.append('')
    
    # Типы коллекций
    if 'list' in features:
        lines.append(ListSupport.generate_list_type('FloatList', 'float', 1024))
        lines.append(ListSupport.generate_list_type('IntList', 'int', 1024))
        lines.append('')
    
    if 'dict' in features:
        lines.append(DictSupport.generate_dict_helpers('IntFloatDict', 'int', 'float', 256))
        lines.append(DictSupport.generate_dict_helpers('IntIntDict', 'int', 'int', 256))
        lines.append('')
    
    if 'set' in features:
        lines.append(SetSupport.generate_set_type('IntSet', 'int', 256))
        lines.append(SetSupport.generate_set_type('FloatSet', 'float', 256))
        lines.append('')
    
    # Thread helpers
    lines.append('#define THREAD_ID (blockIdx.x * blockDim.x + threadIdx.x)')
    lines.append('#define GRID_STRIDE (blockDim.x * gridDim.x)')
    lines.append('')
    
    return '\n'.join(lines)
