"""
PyCUDA Compiler - Type System and Intermediate Representation

Система типов и промежуточное представление для компиляции Python → CUDA.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum, auto


# ============================================================================
# TYPE SYSTEM
# ============================================================================

class TypeKind(Enum):
    """Виды типов."""
    VOID = auto()
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    ARRAY = auto()
    STRUCT = auto()
    POINTER = auto()
    FUNCTION = auto()
    TUPLE = auto()
    GENERIC = auto()  # для вывода типов
    LIST = auto()     # Dynamic List
    OPTIONAL = auto() # Optional[T]
    STRING = auto()   # Fixed size string
    DICT = auto()     # Fixed Hash Map
    SET = auto()      # Fixed Hash Set


@dataclass
class CUDAType:
    """Базовый тип CUDA."""
    kind: TypeKind
    name: str
    cuda_name: str
    size: int = 0
    
    # Для массивов и указателей
    element_type: Optional[CUDAType] = None
    
    # Для структур
    fields: Dict[str, CUDAType] = field(default_factory=dict)
    methods: Dict[str, FunctionType] = field(default_factory=dict)
    
    # Для функций
    param_types: List[CUDAType] = field(default_factory=list)
    return_type: Optional[CUDAType] = None
    
    # Для кортежей
    element_types: List[CUDAType] = field(default_factory=list)
    
    # Для списков (capacity fix)
    capacity: int = 1024
    
    def __hash__(self):
        return hash((self.kind, self.name))
    
    def __eq__(self, other):
        if not isinstance(other, CUDAType):
            return False
        return self.kind == other.kind and self.name == other.name
    
    def is_numeric(self) -> bool:
        return self.kind in (TypeKind.INT, TypeKind.FLOAT, TypeKind.BOOL)
    
    def is_array(self) -> bool:
        return self.kind == TypeKind.ARRAY
    
    def is_list(self) -> bool:
        return self.kind == TypeKind.LIST
    
    def is_pointer(self) -> bool:
        return self.kind == TypeKind.POINTER
    
    def is_struct(self) -> bool:
        return self.kind == TypeKind.STRUCT

    def is_dict(self) -> bool:
        return self.kind == TypeKind.DICT
    
    def pointer_to(self) -> CUDAType:
        """Создаёт указатель на этот тип."""
        return CUDAType(
            kind=TypeKind.POINTER,
            name=f'{self.name}*',
            cuda_name=f'{self.cuda_name}*',
            size=8,
            element_type=self
        )
    
    def array_of(self, size: int = -1) -> CUDAType:
        """Создаёт массив этого типа."""
        return CUDAType(
            kind=TypeKind.ARRAY,
            name=f'Array[{self.name}]',
            cuda_name=f'{self.cuda_name}*',
            size=size * self.size if size > 0 else 0,
            element_type=self
        )

    def list_of(self, capacity: int = 1024) -> CUDAType:
        """Создаёт список этого типа."""
        return CUDAType(
            kind=TypeKind.LIST,
            name=f'List[{self.name}]',
            cuda_name=f'List_{self.name}', # Will be generated struct name
            element_type=self,
            capacity=capacity
        )

    def optional_of(self) -> CUDAType:
        """Создаёт Optional[T] тип."""
        return CUDAType(
            kind=TypeKind.OPTIONAL,
            name=f'Optional[{self.name}]',
            cuda_name=f'Optional_{self.name}',
            element_type=self
        )

    @staticmethod
    def tuple_of(types: List[CUDAType]) -> CUDAType:
        """Создаёт кортеж типов."""
        ptypes = [t for t in types]
        name = f"Tuple[{', '.join(t.name for t in ptypes)}]"
        # Generate safe C++ name (Tuple_float_int)
        cuda_name = f"Tuple_{'_'.join(t.cuda_name.replace('*', 'Ptr').replace(' ', '') for t in ptypes)}"
        return CUDAType(
            kind=TypeKind.TUPLE,
            name=name,
            cuda_name=cuda_name,
            element_types=ptypes
        )

    def dict_of(self, value_type: CUDAType, capacity: int = 256) -> CUDAType:
        """Создаёт словарь Dict[Key, Value]."""
        # self is Key type
        name = f"Dict[{self.name}, {value_type.name}]"
        cuda_name = f"Dict_{self.cuda_name}_{value_type.cuda_name}"
        return CUDAType(
            kind=TypeKind.DICT,
            name=name,
            cuda_name=cuda_name,
            element_types=[self, value_type],
            capacity=capacity
        )

    def set_of(self, capacity: int = 256) -> CUDAType:
        """Создаёт множество Set[self]."""
        name = f"Set[{self.name}]"
        cuda_name = f"Set_{self.cuda_name}"
        return CUDAType(
            kind=TypeKind.SET,
            name=name,
            cuda_name=cuda_name,
            element_type=self,
            capacity=capacity
        )

    def is_tensor(self) -> bool:
        return self.kind == TypeKind.ARRAY and self.size < 0  # Convention: negative size (like -rank) or specific flag?
        # Better: checking name or adding a new Kind. 
        # For now, let's reuse ARRAY but add 'rank' attribute to CUDAType
        
    def tensor_of(self, rank: int) -> CUDAType:
        """Создаёт тензор этого типа."""
        t = CUDAType(
            kind=TypeKind.ARRAY,
            name=f'Tensor[{self.name}, {rank}]',
            cuda_name=f'{self.cuda_name}*',
            element_type=self,
            size=-rank # abuse size to store negative rank
        )
        return t
        
    @property
    def rank(self) -> int:
        if self.kind == TypeKind.ARRAY and self.size < 0:
            return -self.size
        return 0


@dataclass
class FunctionType:
    """Тип функции."""
    name: str
    params: List[Tuple[str, CUDAType]]
    return_type: CUDAType
    is_kernel: bool = False
    is_device: bool = True
    is_method: bool = False
    parent_class: Optional[str] = None


# Базовые типы
VOID = CUDAType(TypeKind.VOID, 'void', 'void', 0)
BOOL = CUDAType(TypeKind.BOOL, 'bool', 'bool', 1)
INT8 = CUDAType(TypeKind.INT, 'int8', 'char', 1)
INT16 = CUDAType(TypeKind.INT, 'int16', 'short', 2)
INT32 = CUDAType(TypeKind.INT, 'int32', 'int', 4)
INT64 = CUDAType(TypeKind.INT, 'int64', 'long long', 8)
UINT8 = CUDAType(TypeKind.INT, 'uint8', 'unsigned char', 1)
UINT16 = CUDAType(TypeKind.INT, 'uint16', 'unsigned short', 2)
UINT32 = CUDAType(TypeKind.INT, 'uint32', 'unsigned int', 4)
UINT64 = CUDAType(TypeKind.INT, 'uint64', 'unsigned long long', 8)
FLOAT16 = CUDAType(TypeKind.FLOAT, 'float16', '__half', 2)
FLOAT32 = CUDAType(TypeKind.FLOAT, 'float32', 'float', 4)
FLOAT64 = CUDAType(TypeKind.FLOAT, 'float64', 'double', 8)
STRING = CUDAType(TypeKind.STRING, 'str', 'String', 64) # Fixed 64 chars

# Алиасы
INT = INT32
FLOAT = FLOAT32
DOUBLE = FLOAT64

# Словарь для поиска типов по имени
BUILTIN_TYPES: Dict[str, CUDAType] = {
    'void': VOID,
    'bool': BOOL,
    'int8': INT8, 'int16': INT16, 'int32': INT32, 'int64': INT64,
    'uint8': UINT8, 'uint16': UINT16, 'uint32': UINT32, 'uint64': UINT64,
    'float16': FLOAT16, 'float32': FLOAT32, 'float64': FLOAT64,
    'int': INT32, 'float': FLOAT32, 'double': FLOAT64,
    'str': STRING,
    # Python aliases
    'True': BOOL, 'False': BOOL,
}


# ============================================================================
# INTERMEDIATE REPRESENTATION (IR)
# ============================================================================

class IRNodeKind(Enum):
    """Виды IR-узлов."""
    # Модуль
    MODULE = auto()
    
    # Определения
    STRUCT_DEF = auto()
    FUNCTION_DEF = auto()
    VAR_DECL = auto()
    
    # Statements
    ASSIGN = auto()
    AUG_ASSIGN = auto()
    IF = auto()
    FOR = auto()
    WHILE = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    EXPR_STMT = auto()
    BLOCK = auto()
    PARALLEL_FOR = auto()
    
    # Expressions
    CONST = auto()
    VAR = auto()
    BINOP = auto()
    UNARYOP = auto()
    COMPARE = auto()
    CALL = auto()
    METHOD_CALL = auto()
    INDEX = auto()
    ATTR = auto()
    TERNARY = auto()
    CAST = auto()
    ARRAY_INIT = auto()
    STRUCT_INIT = auto()
    TENSOR_SLICE = auto()
    TUPLE_INIT = auto()
    OPTIONAL_INIT = auto()


class BinaryOp(Enum):
    """Бинарные операторы."""
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    MOD = '%'
    POW = '**'
    FLOORDIV = '//'
    LSHIFT = '<<'
    RSHIFT = '>>'
    BITOR = '|'
    BITXOR = '^'
    BITAND = '&'
    AND = '&&'
    OR = '||'


class UnaryOp(Enum):
    """Унарные операторы."""
    NEG = '-'
    POS = '+'
    NOT = '!'
    BITNOT = '~'


class CompareOp(Enum):
    """Операторы сравнения."""
    EQ = '=='
    NE = '!='
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='
    IN = 'in'
    NOT_IN = 'not in'


@dataclass
class IRNode:
    """Базовый IR-узел."""
    kind: IRNodeKind
    type: Optional[CUDAType] = None
    
    # Позиция в исходном коде для отладки
    lineno: int = 0
    col_offset: int = 0


@dataclass
class IRModule(IRNode):
    """Модуль (весь файл)."""
    kind: IRNodeKind = IRNodeKind.MODULE
    structs: List[IRStructDef] = field(default_factory=list)
    functions: List[IRFunctionDef] = field(default_factory=list)
    globals: List[IRVarDecl] = field(default_factory=list)


@dataclass
class IRStructDef(IRNode):
    """Определение структуры (класса)."""
    kind: IRNodeKind = IRNodeKind.STRUCT_DEF
    name: str = ''
    fields: List[Tuple[str, CUDAType]] = field(default_factory=list)
    methods: List[IRFunctionDef] = field(default_factory=list)


@dataclass
class IRFunctionDef(IRNode):
    """Определение функции."""
    kind: IRNodeKind = IRNodeKind.FUNCTION_DEF
    name: str = ''
    params: List[Tuple[str, CUDAType]] = field(default_factory=list)
    return_type: CUDAType = field(default_factory=lambda: VOID)
    body: List[IRNode] = field(default_factory=list)
    is_kernel: bool = False
    is_device: bool = True
    is_method: bool = False
    parent_class: Optional[str] = None
    local_vars: Dict[str, CUDAType] = field(default_factory=dict)


@dataclass 
class IRVarDecl(IRNode):
    """Объявление переменной."""
    kind: IRNodeKind = IRNodeKind.VAR_DECL
    name: str = ''
    var_type: CUDAType = field(default_factory=lambda: FLOAT32)
    init_value: Optional[IRNode] = None


@dataclass
class IRAssign(IRNode):
    """Присваивание."""
    kind: IRNodeKind = IRNodeKind.ASSIGN
    target: IRNode = None
    value: IRNode = None


@dataclass
class IRAugAssign(IRNode):
    """Составное присваивание (+=, -= и т.д.)."""
    kind: IRNodeKind = IRNodeKind.AUG_ASSIGN
    target: IRNode = None
    op: BinaryOp = BinaryOp.ADD
    value: IRNode = None


@dataclass
class IRIf(IRNode):
    """Условный оператор."""
    kind: IRNodeKind = IRNodeKind.IF
    condition: IRNode = None
    then_body: List[IRNode] = field(default_factory=list)
    else_body: List[IRNode] = field(default_factory=list)


@dataclass
class IRFor(IRNode):
    """Цикл for."""
    kind: IRNodeKind = IRNodeKind.FOR
    var_name: str = ''
    start: IRNode = None
    end: IRNode = None
    step: IRNode = None
    body: List[IRNode] = field(default_factory=list)
    is_parallel: bool = False


@dataclass
class IRParallelFor(IRNode):
    """Параллельный цикл (маппится на CUDA grid)."""
    kind: IRNodeKind = IRNodeKind.PARALLEL_FOR
    var_name: str = ''
    size_expr: IRNode = None
    body: List[IRNode] = field(default_factory=list)


@dataclass
class IRWhile(IRNode):
    """Цикл while."""
    kind: IRNodeKind = IRNodeKind.WHILE
    condition: IRNode = None
    body: List[IRNode] = field(default_factory=list)


@dataclass
class IRReturn(IRNode):
    """Возврат из функции."""
    kind: IRNodeKind = IRNodeKind.RETURN
    value: Optional[IRNode] = None


@dataclass
class IRBreak(IRNode):
    """Break."""
    kind: IRNodeKind = IRNodeKind.BREAK


@dataclass
class IRContinue(IRNode):
    """Continue."""
    kind: IRNodeKind = IRNodeKind.CONTINUE


@dataclass
class IRExprStmt(IRNode):
    """Expression statement."""
    kind: IRNodeKind = IRNodeKind.EXPR_STMT
    expr: IRNode = None


@dataclass
class IRBlock(IRNode):
    """Блок statements."""
    kind: IRNodeKind = IRNodeKind.BLOCK
    statements: List[IRNode] = field(default_factory=list)


@dataclass
class IRConst(IRNode):
    """Константа."""
    kind: IRNodeKind = IRNodeKind.CONST
    value: Any = None


@dataclass
class IRVar(IRNode):
    """Переменная."""
    kind: IRNodeKind = IRNodeKind.VAR
    name: str = ''


@dataclass
class IRBinOp(IRNode):
    """Бинарная операция."""
    kind: IRNodeKind = IRNodeKind.BINOP
    op: BinaryOp = BinaryOp.ADD
    left: IRNode = None
    right: IRNode = None


@dataclass
class IRUnaryOp(IRNode):
    """Унарная операция."""
    kind: IRNodeKind = IRNodeKind.UNARYOP
    op: UnaryOp = UnaryOp.NEG
    operand: IRNode = None


@dataclass
class IRCompare(IRNode):
    """Сравнение."""
    kind: IRNodeKind = IRNodeKind.COMPARE
    op: CompareOp = CompareOp.EQ
    left: IRNode = None
    right: IRNode = None


@dataclass
class IRCall(IRNode):
    """Вызов функции."""
    kind: IRNodeKind = IRNodeKind.CALL
    func_name: str = ''
    args: List[IRNode] = field(default_factory=list)


@dataclass
class IRMethodCall(IRNode):
    """Вызов метода."""
    kind: IRNodeKind = IRNodeKind.METHOD_CALL
    obj: IRNode = None
    method_name: str = ''
    args: List[IRNode] = field(default_factory=list)


@dataclass
class IRIndex(IRNode):
    """Индексация массива."""
    kind: IRNodeKind = IRNodeKind.INDEX
    obj: IRNode = None
    index: IRNode = None


@dataclass
class IRAttr(IRNode):
    """Доступ к атрибуту."""
    kind: IRNodeKind = IRNodeKind.ATTR
    obj: IRNode = None
    attr_name: str = ''


@dataclass
class IRTernary(IRNode):
    """Тернарный оператор (a if cond else b)."""
    kind: IRNodeKind = IRNodeKind.TERNARY
    condition: IRNode = None
    then_value: IRNode = None
    else_value: IRNode = None


@dataclass
class IRCast(IRNode):
    """Приведение типа."""
    kind: IRNodeKind = IRNodeKind.CAST
    expr: IRNode = None
    target_type: CUDAType = None


@dataclass
class IRArrayInit(IRNode):
    """Инициализация массива."""
    kind: IRNodeKind = IRNodeKind.ARRAY_INIT
    elements: List[IRNode] = field(default_factory=list)


@dataclass
class IRTupleInit(IRNode):
    """Инициализация кортежа."""
    kind: IRNodeKind = IRNodeKind.TUPLE_INIT
    elements: List[IRNode] = field(default_factory=list)


@dataclass
class IROptionalInit(IRNode):
    """Инициализация Optional."""
    kind: IRNodeKind = IRNodeKind.OPTIONAL_INIT
    value: Optional[IRNode] = None # None for empty/null, or expr
    val_type: Optional[CUDAType] = None # Explicit type if known (e.g. for None)


@dataclass
class IRStructInit(IRNode):
    """Инициализация структуры."""
    kind: IRNodeKind = IRNodeKind.STRUCT_INIT
    struct_name: str = ''
    field_values: Dict[str, IRNode] = field(default_factory=dict)


@dataclass
class IRTensorSlice(IRNode):
    """Срез тензора (возвращает под-тензор или элемент)."""
    kind: IRNodeKind = IRNodeKind.TENSOR_SLICE
    obj: IRNode = None
    slices: List[IRNode] = field(default_factory=list)
    # Metadata for aliasing
    base_name: Optional[str] = None
    dim_offset: int = 0
