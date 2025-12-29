"""
PyCUDA Compiler - Type Inference

Выводит типы переменных и выражений автоматически.
"""

from __future__ import annotations
from typing import Dict, Optional
from types_ir import (
    CUDAType, TypeKind,
    VOID, BOOL, INT32, FLOAT32, IRNode, IRModule, IRStructDef, IRFunctionDef, IRVarDecl,
    IRAssign, IRAugAssign, IRIf, IRFor, IRParallelFor, IRWhile,
    IRReturn, IRExprStmt, IRBlock,
    IRConst, IRVar, IRBinOp, IRUnaryOp, IRCompare,
    IRCall, IRMethodCall, IRIndex, IRAttr, IRTernary, IRCast,
    IRArrayInit, IRTupleInit, IROptionalInit, IRStructInit, IRTensorSlice, IRArrayAlloc, IRDelete, BinaryOp, UnaryOp
)


class TypeInferenceError(Exception):
    """Ошибка вывода типов."""
    pass


class TypeInference:
    """Выводит типы для всего IR."""
    
    # Сигнатуры встроенных функций
    BUILTIN_FUNCS = {
        'sqrt': (FLOAT32, [FLOAT32]),
        'sin': (FLOAT32, [FLOAT32]),
        'cos': (FLOAT32, [FLOAT32]),
        'tan': (FLOAT32, [FLOAT32]),
        'exp': (FLOAT32, [FLOAT32]),
        'log': (FLOAT32, [FLOAT32]),
        'log2': (FLOAT32, [FLOAT32]),
        'log10': (FLOAT32, [FLOAT32]),
        'floor': (FLOAT32, [FLOAT32]),
        'ceil': (FLOAT32, [FLOAT32]),
        'abs': (FLOAT32, [FLOAT32]),
        'fabs': (FLOAT32, [FLOAT32]),
        'pow': (FLOAT32, [FLOAT32, FLOAT32]),
        'min': (FLOAT32, [FLOAT32, FLOAT32]),
        'max': (FLOAT32, [FLOAT32, FLOAT32]),
        'asin': (FLOAT32, [FLOAT32]),
        'acos': (FLOAT32, [FLOAT32]),
        'atan': (FLOAT32, [FLOAT32]),
        'atan2': (FLOAT32, [FLOAT32, FLOAT32]),
        'sinh': (FLOAT32, [FLOAT32]),
        'cosh': (FLOAT32, [FLOAT32]),
        'tanh': (FLOAT32, [FLOAT32]),
        'rsqrt': (FLOAT32, [FLOAT32]),
        'fma': (FLOAT32, [FLOAT32, FLOAT32, FLOAT32]),
        # Integer versions
        'abs_int': (INT32, [INT32]),
        'min_int': (INT32, [INT32, INT32]),
        'max_int': (INT32, [INT32, INT32]),
    }
    
    def __init__(self):
        self.structs: Dict[str, IRStructDef] = {}
        self.functions: Dict[str, IRFunctionDef] = {}
        self.current_function: Optional[IRFunctionDef] = None
        self.var_types: Dict[str, CUDAType] = {}
        self.changed: bool = False
    
    def infer(self, module: IRModule) -> IRModule:
        """Выводит типы для всего модуля."""
        # Собираем информацию о структурах
        for struct in module.structs:
            self.structs[struct.name] = struct
        
        # Собираем сигнатуры функций
        for func in module.functions:
            self.functions[func.name] = func
        
        # Итеративно выводим типы до сходимости
        for _ in range(10):  # максимум 10 итераций
            self.changed = False
            
            for struct in module.structs:
                self.infer_struct(struct)
            
            for func in module.functions:
                self.infer_function(func)
            
            if not self.changed:
                break
        
        return module
    
    def infer_struct(self, struct: IRStructDef):
        """Выводит типы для структуры."""
        # Создаём тип структуры
        struct_type = CUDAType(
            kind=TypeKind.STRUCT,
            name=struct.name,
            cuda_name=struct.name,
            fields={name: t for name, t in struct.fields}
        )
        
        struct.type = struct_type
        
        # Методы
        for method in struct.methods:
            self.infer_function(method)
    
    def infer_function(self, func: IRFunctionDef):
        """Выводит типы для функции."""
        self.current_function = func
        self.var_types = {}
        
        # Регистрируем параметры
        for param_name, param_type in func.params:
            self.var_types[param_name] = param_type
        
        # Регистрируем уже известные локальные переменные
        for var_name, var_type in func.local_vars.items():
            if var_name not in self.var_types:
                self.var_types[var_name] = var_type
        
        # self для методов
        if func.is_method and func.parent_class:
            if func.parent_class in self.structs:
                struct = self.structs[func.parent_class]
                self_type = CUDAType(
                    kind=TypeKind.POINTER,
                    name=f'{func.parent_class}*',
                    cuda_name=f'{func.parent_class}*',
                    element_type=struct.type
                )
                self.var_types['self'] = self_type
        
        # Выводим типы в теле
        for stmt in func.body:
            self.infer_stmt(stmt)
        
        # Обновляем локальные переменные
        func.local_vars.update(self.var_types)
        
        self.current_function = None
    
    def infer_stmt(self, stmt: IRNode):
        """Выводит типы для statement."""
        if isinstance(stmt, IRAssign):
            self.infer_assign(stmt)
        elif isinstance(stmt, IRAugAssign):
            self.infer_aug_assign(stmt)
        elif isinstance(stmt, IRVarDecl):
            self.infer_var_decl(stmt)
        elif isinstance(stmt, IRIf):
            self.infer_if(stmt)
        elif isinstance(stmt, (IRFor, IRParallelFor)):
            self.infer_for(stmt)
        elif isinstance(stmt, IRWhile):
            self.infer_while(stmt)
        elif isinstance(stmt, IRReturn):
            self.infer_return(stmt)
        elif isinstance(stmt, IRExprStmt):
            self.infer_expr(stmt.expr)
        elif isinstance(stmt, IRBlock):
            for s in stmt.statements:
                self.infer_stmt(s)
        elif isinstance(stmt, IRDelete):
            self.infer_expr(stmt.target)
    
    def infer_assign(self, stmt: IRAssign):
        """Выводит тип для присваивания."""
        value_type = self.infer_expr(stmt.value)
        
        if isinstance(stmt.target, IRVar):
            var_name = stmt.target.name
            
            if var_name in self.var_types:
                # Уже известен тип
                current_type = self.var_types[var_name]
                if current_type == FLOAT32 and value_type and value_type != FLOAT32:
                    # Уточняем тип (был float по умолчанию)
                    self.var_types[var_name] = value_type
                    stmt.target.type = value_type
                    self.changed = True
                elif current_type.kind == TypeKind.OPTIONAL and value_type and value_type.kind == TypeKind.OPTIONAL and value_type.element_type == VOID:
                     # Assigning None (Optional[Void]) to Optional[T] is valid
                     # Propagate type to the value node so codegen knows what to generate
                     stmt.value.type = current_type
                     pass
                else:
                    stmt.target.type = current_type
            else:
                # Новая переменная - присваиваем тип от значения
                self.var_types[var_name] = value_type or FLOAT32
                stmt.target.type = self.var_types[var_name]
                self.changed = True
        else:
            self.infer_expr(stmt.target)
    
    def infer_aug_assign(self, stmt: IRAugAssign):
        """Выводит тип для +=."""
        self.infer_expr(stmt.target)
        self.infer_expr(stmt.value)
    
    def infer_var_decl(self, stmt: IRVarDecl):
        """Выводит тип для объявления переменной."""
        if stmt.name:
            self.var_types[stmt.name] = stmt.var_type
        if stmt.init_value:
            self.infer_expr(stmt.init_value)
    
    def infer_if(self, stmt: IRIf):
        """Выводит типы для if."""
        self.infer_expr(stmt.condition)
        for s in stmt.then_body:
            self.infer_stmt(s)
        for s in stmt.else_body:
            self.infer_stmt(s)
    
    def infer_for(self, stmt):
        """Выводит типы для for."""
        if hasattr(stmt, 'var_name'):
            self.var_types[stmt.var_name] = INT32
        
        if hasattr(stmt, 'start') and stmt.start:
            self.infer_expr(stmt.start)
        if hasattr(stmt, 'end') and stmt.end:
            self.infer_expr(stmt.end)
        if hasattr(stmt, 'step') and stmt.step:
            self.infer_expr(stmt.step)
        if hasattr(stmt, 'size_expr') and stmt.size_expr:
            self.infer_expr(stmt.size_expr)
        
        for s in stmt.body:
            self.infer_stmt(s)
    
    def infer_while(self, stmt: IRWhile):
        """Выводит типы для while."""
        self.infer_expr(stmt.condition)
        for s in stmt.body:
            self.infer_stmt(s)
    
    def infer_return(self, stmt: IRReturn):
        """Выводит тип для return."""
        if stmt.value:
            ret_type = self.infer_expr(stmt.value)
            # Проверяем совместимость с объявленным типом возврата
            if self.current_function and self.current_function.return_type == VOID:
                if ret_type and ret_type != VOID:
                    # Allow inferred return type update
                    self.current_function.return_type = ret_type
                    self.changed = True
            elif self.current_function and self.current_function.return_type.kind == TypeKind.OPTIONAL:
                 # Check if returning None (Optional[Void])
                 if ret_type and ret_type.kind == TypeKind.OPTIONAL and ret_type.element_type == VOID:
                      # Propagate return type to the return value node
                      stmt.value.type = self.current_function.return_type
                      pass # returning None is fine
                 elif ret_type != self.current_function.return_type:
                      pass # Mismatch? Or maybe generic Optional logic?
    
    def infer_expr(self, expr: IRNode) -> Optional[CUDAType]:
        """Выводит тип выражения."""
        if expr is None:
            return None
        
        if isinstance(expr, IRConst):
            return expr.type
        
        elif isinstance(expr, IRVar):
            return self.infer_var(expr)
        
        elif isinstance(expr, IRBinOp):
            return self.infer_binop(expr)
        
        elif isinstance(expr, IRUnaryOp):
            return self.infer_unaryop(expr)
        
        elif isinstance(expr, IRCompare):
            self.infer_expr(expr.left)
            self.infer_expr(expr.right)
            expr.type = BOOL
            return BOOL
        
        elif isinstance(expr, IRCall):
            return self.infer_call(expr)
        
        elif isinstance(expr, IRMethodCall):
            return self.infer_method_call(expr)
        
        elif isinstance(expr, IRIndex):
            return self.infer_index(expr)
        
        elif isinstance(expr, IRAttr):
            return self.infer_attr(expr)
        
        elif isinstance(expr, IRTernary):
            return self.infer_ternary(expr)
        
        elif isinstance(expr, IRCast):
            expr.type = expr.target_type
            return expr.target_type
        
        if isinstance(expr, IRArrayInit):
            return self.infer_array_init(expr)
        
        elif isinstance(expr, IRTupleInit):
            return self.infer_tuple_init(expr)
            
        elif isinstance(expr, IROptionalInit):
            self.infer_optional_init(expr)
        elif isinstance(expr, IRStructInit):
            self.infer_struct_init(expr)
        elif isinstance(expr, IRArrayAlloc):
            self.infer_array_alloc(expr)
        
        elif isinstance(expr, IRTensorSlice):
            return self.infer_tensor_slice(expr)
        
        return expr.type

    def infer_tensor_slice(self, expr: IRTensorSlice) -> Optional[CUDAType]:
        """Выводит тип среза тензора."""
        # Тип уже должен быть установлен парсером, но мы можем проверить
        if expr.type:
            return expr.type
        
        # Если нет, выводим из объекта
        obj_type = self.infer_expr(expr.obj)
        if obj_type and obj_type.is_tensor():
             # Re-calculate rank logic if needed, but Parser usually does this better 
             # because it has access to AST structure (indices count).
             # Start with simple propagation if parser set it.
             pass
        
        return expr.type
    
    def infer_var(self, expr: IRVar) -> Optional[CUDAType]:
        """Выводит тип переменной."""
        if expr.name in self.var_types:
            expr.type = self.var_types[expr.name]
        return expr.type
    
    def infer_binop(self, expr: IRBinOp) -> CUDAType:
        """Выводит тип бинарной операции."""
        left_type = self.infer_expr(expr.left)
        right_type = self.infer_expr(expr.right)
        
        # Логические операции → bool
        if expr.op in (BinaryOp.AND, BinaryOp.OR):
            expr.type = BOOL
            return BOOL
        
        # Битовые операции → int
        if expr.op in (BinaryOp.LSHIFT, BinaryOp.RSHIFT, BinaryOp.BITOR, 
                      BinaryOp.BITXOR, BinaryOp.BITAND):
            expr.type = INT32
            return INT32
        
        # float + anything → float
        if left_type and left_type.kind == TypeKind.FLOAT:
            expr.type = left_type
            return left_type
        if right_type and right_type.kind == TypeKind.FLOAT:
            expr.type = right_type
            return right_type
        
        # int / anything → float (деление)
        if expr.op == BinaryOp.DIV:
            expr.type = FLOAT32
            return FLOAT32
        
        # int + int → int
        if left_type and left_type.kind == TypeKind.INT:
            expr.type = left_type
            return left_type
        
        # По умолчанию float
        if expr.type is None:
            expr.type = FLOAT32
        return expr.type
    
    def infer_unaryop(self, expr: IRUnaryOp) -> CUDAType:
        """Выводит тип унарной операции."""
        operand_type = self.infer_expr(expr.operand)
        
        if expr.op == UnaryOp.NOT:
            expr.type = BOOL
        elif expr.op == UnaryOp.BITNOT:
            expr.type = INT32
        else:
            expr.type = operand_type
        
        return expr.type
    
    def infer_call(self, expr: IRCall) -> Optional[CUDAType]:
        """Выводит тип вызова функции."""
        # Выводим типы аргументов
        for arg in expr.args:
            self.infer_expr(arg)
        
        func_name = expr.func_name
        
        # Встроенные функции
        if func_name in self.BUILTIN_FUNCS:
            ret_type, _ = self.BUILTIN_FUNCS[func_name]
            expr.type = ret_type
            return ret_type
        
        # len → int
        if func_name == 'len':
            expr.type = INT32
            return INT32
        
        # Приведение типов
        if func_name == 'int':
            expr.type = INT32
            return INT32
        if func_name == 'float':
            expr.type = FLOAT32
            return FLOAT32
        if func_name == 'bool':
            expr.type = BOOL
            return BOOL
        
        # Пользовательская функция
        if func_name in self.functions:
            func = self.functions[func_name]
            expr.type = func.return_type
            return func.return_type
        
        # Конструктор структуры
        if func_name in self.structs:
            struct = self.structs[func_name]
            expr.type = struct.type
            return struct.type
        
        return expr.type
    
    def infer_method_call(self, expr: IRMethodCall) -> Optional[CUDAType]:
        """Выводит тип вызова метода."""
        obj_type = self.infer_expr(expr.obj)
        
        for arg in expr.args:
            self.infer_expr(arg)
        
        # Ищем метод в структуре
        if obj_type and obj_type.is_struct():
            struct_name = obj_type.name
            if struct_name in self.structs:
                struct = self.structs[struct_name]
                for method in struct.methods:
                    if method.name == expr.method_name:
                        expr.type = method.return_type
                        return method.return_type
        
        # Для указателей на структуры
        if obj_type and obj_type.is_pointer() and obj_type.element_type:
            elem_type = obj_type.element_type
            if elem_type.is_struct():
                struct_name = elem_type.name
                if struct_name in self.structs:
                    struct = self.structs[struct_name]
                    for method in struct.methods:
                        if method.name == expr.method_name:
                            expr.type = method.return_type
                            return method.return_type
        
        return expr.type
    
    def infer_index(self, expr: IRIndex) -> Optional[CUDAType]:
        """Выводит тип индексации."""
        obj_type = self.infer_expr(expr.obj)
        self.infer_expr(expr.index)
        
        if obj_type:
            if obj_type.element_type:
                expr.type = obj_type.element_type
            elif obj_type.kind == TypeKind.TUPLE:
                # Tuple indexing: tuple[const]
                if isinstance(expr.index, IRConst) and isinstance(expr.index.value, int):
                    idx = expr.index.value
                    if 0 <= idx < len(obj_type.element_types):
                        expr.type = obj_type.element_types[idx]
            # Handle variable index for homogeneous tuple? Not supporting that yet as field access is static.
        
        return expr.type
    
    def infer_attr(self, expr: IRAttr) -> Optional[CUDAType]:
        """Выводит тип атрибута."""
        obj_type = self.infer_expr(expr.obj)
        
        # Для структуры
        if obj_type and obj_type.is_struct():
            if expr.attr_name in obj_type.fields:
                expr.type = obj_type.fields[expr.attr_name]
                return expr.type
        
        # Для указателя на структуру
        if obj_type and obj_type.is_pointer() and obj_type.element_type:
            elem_type = obj_type.element_type
            if elem_type.is_struct() and expr.attr_name in elem_type.fields:
                expr.type = elem_type.fields[expr.attr_name]
                return expr.type
        
        # self.field в методах
        if isinstance(expr.obj, IRVar) and expr.obj.name == 'self':
            if self.current_function and self.current_function.parent_class:
                struct_name = self.current_function.parent_class
                if struct_name in self.structs:
                    struct = self.structs[struct_name]
                    for field_name, field_type in struct.fields:
                        if field_name == expr.attr_name:
                            expr.type = field_type
                            return field_type
        
        return expr.type
    
    def infer_ternary(self, expr: IRTernary) -> Optional[CUDAType]:
        """Выводит тип тернарного оператора."""
        self.infer_expr(expr.condition)
        then_type = self.infer_expr(expr.then_value)
        else_type = self.infer_expr(expr.else_value)
        
        # Выбираем более широкий тип
        if then_type and else_type:
            if then_type.kind == TypeKind.FLOAT or else_type.kind == TypeKind.FLOAT:
                expr.type = FLOAT32
            else:
                expr.type = then_type
        else:
            expr.type = then_type or else_type
        
        return expr.type
    
    def infer_array_init(self, expr: IRArrayInit) -> Optional[CUDAType]:
        """Выводит тип инициализации массива."""
        elem_type = FLOAT32
        if expr.elements:
            elem_type = self.infer_expr(expr.elements[0]) or FLOAT32
        
        expr.type = elem_type.array_of(len(expr.elements))
        return expr.type

    def infer_array_alloc(self, expr: IRArrayAlloc) -> Optional[CUDAType]:
        """Выводит тип выделения массива (malloc/new)."""
        self.infer_expr(expr.size)
        if expr.values:
            for v in expr.values:
                 self.infer_expr(v)
        
        if expr.element_type:
            # Return pointer to element type (as new[] returns pointer)
            expr.type = expr.element_type.pointer_to()
        else:
            expr.type = FLOAT32.pointer_to()
            
        return expr.type

    def infer_tuple_init(self, expr: IRTupleInit) -> Optional[CUDAType]:
        """Выводит тип инициализации кортежа."""
        types = []
        for e in expr.elements:
            t = self.infer_expr(e) or FLOAT32
            types.append(t)
        
        expr.type = CUDAType.tuple_of(types)
        return expr.type

    def infer_optional_init(self, expr: IROptionalInit) -> Optional[CUDAType]:
        """Выводит тип инициализации Optional."""
        if expr.value is None:
            # None -> Optional[Void] (generic None)
            expr.type = VOID.optional_of()
            return expr.type
            
        val_type = self.infer_expr(expr.value)
        if val_type:
            expr.type = val_type.optional_of()
        return expr.type
    
    def infer_struct_init(self, expr: IRStructInit) -> Optional[CUDAType]:
        """Выводит тип инициализации структуры."""
        if expr.struct_name in self.structs:
            struct = self.structs[expr.struct_name]
            expr.type = struct.type
        return expr.type
