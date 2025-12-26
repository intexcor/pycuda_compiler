"""
PyCUDA Compiler - Python AST Parser

Парсит Python AST и создаёт IR (промежуточное представление).
"""

from __future__ import annotations
import ast
from typing import Dict, List, Optional, Tuple, Any, Set
from types_ir import (
    CUDAType, TypeKind, FunctionType,
    BUILTIN_TYPES, VOID, BOOL, INT32, INT64, FLOAT32, FLOAT64,
    IRNode, IRModule, IRStructDef, IRFunctionDef, IRVarDecl,
    IRAssign, IRAugAssign, IRIf, IRFor, IRParallelFor, IRWhile,
    IRReturn, IRBreak, IRContinue, IRExprStmt, IRBlock,
    IRConst, IRVar, IRBinOp, IRUnaryOp, IRCompare,
    IRCall, IRMethodCall, IRIndex, IRAttr, IRTernary, IRCast,
    IRArrayInit, IRStructInit, IRTensorSlice,
    BinaryOp, UnaryOp, CompareOp
)


class ParseError(Exception):
    """Ошибка парсинга."""
    def __init__(self, message: str, node: ast.AST = None):
        self.message = message
        self.node = node
        self.lineno = getattr(node, 'lineno', 0) if node else 0
        self.col_offset = getattr(node, 'col_offset', 0) if node else 0
        super().__init__(f"Line {self.lineno}: {message}")


class PythonParser:
    """Парсер Python AST → IR."""
    
    # Маппинг Python операторов в наши
    BINOP_MAP = {
        ast.Add: BinaryOp.ADD,
        ast.Sub: BinaryOp.SUB,
        ast.Mult: BinaryOp.MUL,
        ast.Div: BinaryOp.DIV,
        ast.Mod: BinaryOp.MOD,
        ast.Pow: BinaryOp.POW,
        ast.FloorDiv: BinaryOp.FLOORDIV,
        ast.LShift: BinaryOp.LSHIFT,
        ast.RShift: BinaryOp.RSHIFT,
        ast.BitOr: BinaryOp.BITOR,
        ast.BitXor: BinaryOp.BITXOR,
        ast.BitAnd: BinaryOp.BITAND,
    }
    
    UNARYOP_MAP = {
        ast.USub: UnaryOp.NEG,
        ast.UAdd: UnaryOp.POS,
        ast.Not: UnaryOp.NOT,
        ast.Invert: UnaryOp.BITNOT,
    }
    
    CMPOP_MAP = {
        ast.Eq: CompareOp.EQ,
        ast.NotEq: CompareOp.NE,
        ast.Lt: CompareOp.LT,
        ast.LtE: CompareOp.LE,
        ast.Gt: CompareOp.GT,
        ast.GtE: CompareOp.GE,
    }
    
    def __init__(self):
        self.structs: Dict[str, IRStructDef] = {}
        self.functions: Dict[str, IRFunctionDef] = {}
        self.current_function: Optional[IRFunctionDef] = None
        self.current_class: Optional[str] = None
        self.scope_stack: List[Dict[str, CUDAType]] = [{}]
        self.tensor_aliases: Dict[str, Tuple[str, int]] = {} # local_name -> (base_param_name, dim_offset)
        self.loop_depth: int = 0
        
    def parse(self, source: str) -> IRModule:
        """Парсит исходный код Python."""
        tree = ast.parse(source)
        return self.visit_module(tree)
    
    def push_scope(self):
        """Добавляет новую область видимости."""
        self.scope_stack.append({})
    
    def pop_scope(self):
        """Удаляет текущую область видимости."""
        self.scope_stack.pop()
    
    def define_var(self, name: str, var_type: CUDAType):
        """Определяет переменную в текущей области видимости."""
        self.scope_stack[-1][name] = var_type
        if self.current_function:
            self.current_function.local_vars[name] = var_type
    
    def lookup_var(self, name: str) -> Optional[CUDAType]:
        """Ищет переменную во всех областях видимости."""
        for scope in reversed(self.scope_stack):
            if name in scope:
                return scope[name]
        return None
    
    def visit_module(self, node: ast.Module) -> IRModule:
        """Обрабатывает модуль."""
        module = IRModule()
        
        for item in node.body:
            if isinstance(item, ast.ClassDef):
                struct = self.visit_class(item)
                module.structs.append(struct)
                self.structs[struct.name] = struct
            elif isinstance(item, ast.FunctionDef):
                func = self.visit_function(item)
                module.functions.append(func)
                self.functions[func.name] = func
            elif isinstance(item, ast.Assign):
                # Глобальная переменная
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        var_decl = IRVarDecl(
                            name=target.id,
                            init_value=self.visit_expr(item.value),
                            lineno=item.lineno
                        )
                        module.globals.append(var_decl)
            elif isinstance(item, ast.AnnAssign):
                # Глобальная переменная с аннотацией типа
                if isinstance(item.target, ast.Name):
                    var_type = self.parse_type_annotation(item.annotation)
                    var_decl = IRVarDecl(
                        name=item.target.id,
                        var_type=var_type,
                        init_value=self.visit_expr(item.value) if item.value else None,
                        lineno=item.lineno
                    )
                    module.globals.append(var_decl)
        
        return module
    
    def visit_class(self, node: ast.ClassDef) -> IRStructDef:
        """Обрабатывает класс → структуру."""
        self.current_class = node.name
        struct = IRStructDef(
            name=node.name,
            lineno=node.lineno
        )
        
        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                # Поле класса: x: float32
                if isinstance(item.target, ast.Name):
                    field_name = item.target.id
                    field_type = self.parse_type_annotation(item.annotation)
                    struct.fields.append((field_name, field_type))
            elif isinstance(item, ast.FunctionDef):
                # Метод
                method = self.visit_function(item, is_method=True)
                method.parent_class = node.name
                struct.methods.append(method)
        
        self.current_class = None
        return struct
    
    def visit_function(self, node: ast.FunctionDef, is_method: bool = False) -> IRFunctionDef:
        """Обрабатывает функцию."""
        # Проверяем декораторы
        is_kernel = False
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == 'kernel':
                is_kernel = True
        
        func = IRFunctionDef(
            name=node.name,
            is_kernel=is_kernel,
            is_device=not is_kernel,
            is_method=is_method,
            parent_class=self.current_class,
            lineno=node.lineno
        )
        
        self.current_function = func
        self.push_scope()
        
        # Параметры
        for arg in node.args.args:
            if is_method and arg.arg == 'self':
                continue
            
            arg_type = FLOAT32  # тип по умолчанию
            if arg.annotation:
                arg_type = self.parse_type_annotation(arg.annotation)
            
            func.params.append((arg.arg, arg_type))
            self.define_var(arg.arg, arg_type)
        
        # Возвращаемый тип
        if node.returns:
            func.return_type = self.parse_type_annotation(node.returns)
        else:
            func.return_type = VOID
        
        # Тело функции
        for stmt in node.body:
            ir_stmt = self.visit_stmt(stmt)
            if ir_stmt:
                func.body.append(ir_stmt)
        
        self.pop_scope()
        self.current_function = None
        
        return func
    
    def visit_stmt(self, node: ast.stmt) -> Optional[IRNode]:
        """Обрабатывает statement."""
        if isinstance(node, ast.Assign):
            return self.visit_assign(node)
        elif isinstance(node, ast.AnnAssign):
            return self.visit_ann_assign(node)
        elif isinstance(node, ast.AugAssign):
            return self.visit_aug_assign(node)
        elif isinstance(node, ast.If):
            return self.visit_if(node)
        elif isinstance(node, ast.For):
            return self.visit_for(node)
        elif isinstance(node, ast.While):
            return self.visit_while(node)
        elif isinstance(node, ast.Return):
            return self.visit_return(node)
        elif isinstance(node, ast.Break):
            return IRBreak(lineno=node.lineno)
        elif isinstance(node, ast.Continue):
            return IRContinue(lineno=node.lineno)
        elif isinstance(node, ast.Expr):
            return IRExprStmt(expr=self.visit_expr(node.value), lineno=node.lineno)
        elif isinstance(node, ast.Pass):
            return None
        else:
            raise ParseError(f"Unsupported statement: {type(node).__name__}", node)
    
    def visit_assign(self, node: ast.Assign) -> IRNode:
        """Обрабатывает присваивание."""
        value_ir = self.visit_expr(node.value)
        
        # Для простоты берём первую цель
        target = node.targets[0]
        target_ir = self.visit_expr(target)
        
        # Если это новая переменная - регистрируем
        if isinstance(target, ast.Name):
            if not self.lookup_var(target.id):
                var_type = value_ir.type if value_ir.type else FLOAT32
                self.define_var(target.id, var_type)
            
            # Tensor Alias Tracking
            if isinstance(value_ir, IRTensorSlice):
                # Calculate new offset based on consumed dimensions
                consumed = len(value_ir.slices)
                new_offset = value_ir.dim_offset + consumed
                self.tensor_aliases[target.id] = (value_ir.base_name, new_offset)
                
                # Also update the variable type to the slice type
                if self.current_function:
                    self.current_function.local_vars[target.id] = value_ir.type
        
        return IRAssign(target=target_ir, value=value_ir, lineno=node.lineno)
    
    def visit_ann_assign(self, node: ast.AnnAssign) -> IRNode:
        """Обрабатывает присваивание с аннотацией типа."""
        var_type = self.parse_type_annotation(node.annotation)
        
        if isinstance(node.target, ast.Name):
            self.define_var(node.target.id, var_type)
        
        target_ir = self.visit_expr(node.target)
        target_ir.type = var_type
        
        if node.value:
            value_ir = self.visit_expr(node.value)
            return IRAssign(target=target_ir, value=value_ir, lineno=node.lineno)
        else:
            return IRVarDecl(
                name=node.target.id if isinstance(node.target, ast.Name) else '',
                var_type=var_type,
                lineno=node.lineno
            )
    
    def visit_aug_assign(self, node: ast.AugAssign) -> IRNode:
        """Обрабатывает += и подобные."""
        target_ir = self.visit_expr(node.target)
        value_ir = self.visit_expr(node.value)
        op = self.BINOP_MAP.get(type(node.op), BinaryOp.ADD)
        
        return IRAugAssign(
            target=target_ir,
            op=op,
            value=value_ir,
            lineno=node.lineno
        )
    
    def visit_if(self, node: ast.If) -> IRNode:
        """Обрабатывает if."""
        condition = self.visit_expr(node.test)
        
        then_body = []
        for stmt in node.body:
            ir_stmt = self.visit_stmt(stmt)
            if ir_stmt:
                then_body.append(ir_stmt)
        
        else_body = []
        for stmt in node.orelse:
            ir_stmt = self.visit_stmt(stmt)
            if ir_stmt:
                else_body.append(ir_stmt)
        
        return IRIf(
            condition=condition,
            then_body=then_body,
            else_body=else_body,
            lineno=node.lineno
        )
    
    def visit_for(self, node: ast.For) -> IRNode:
        """Обрабатывает for."""
        if not isinstance(node.target, ast.Name):
            raise ParseError("Only simple for loop variable supported", node)
        
        var_name = node.target.id
        self.define_var(var_name, INT32)
        
        # Auto-Parallelization Logic
        should_parallelize = False
        if self.current_function and self.current_function.is_kernel and self.loop_depth == 0:
            should_parallelize = True

        self.loop_depth += 1
        try:
            # Проверяем range()
            if isinstance(node.iter, ast.Call):
                if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                    # Check if parallelization is requested
                    loop_node = self.visit_range_for(node, var_name, node.iter)
                    if should_parallelize and isinstance(loop_node, IRFor):
                         loop_node.is_parallel = True
                    return loop_node
                
                elif isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'len':
                    # for i in range(len(arr)) → parallel for (Existing explicit logic)
                    return self.visit_parallel_for(node, var_name, node.iter)
            
            # for x in arr → обычный цикл
            raise ParseError("Only range() loops are supported", node)
        finally:
            self.loop_depth -= 1
    
    def visit_range_for(self, node: ast.For, var_name: str, range_call: ast.Call) -> IRNode:
        """Обрабатывает for i in range(...)."""
        args = range_call.args
        
        if len(args) == 1:
            start = IRConst(value=0, type=INT32)
            end = self.visit_expr(args[0])
            step = IRConst(value=1, type=INT32)
        elif len(args) == 2:
            start = self.visit_expr(args[0])
            end = self.visit_expr(args[1])
            step = IRConst(value=1, type=INT32)
        elif len(args) == 3:
            start = self.visit_expr(args[0])
            end = self.visit_expr(args[1])
            step = self.visit_expr(args[2])
        else:
            raise ParseError("range() takes 1-3 arguments", range_call)
        
        # Тело цикла
        body = []
        self.push_scope()
        for stmt in node.body:
            ir_stmt = self.visit_stmt(stmt)
            if ir_stmt:
                body.append(ir_stmt)
        self.pop_scope()
        
        # Если это kernel и range(len(...)) — делаем параллельный
        if (self.current_function and self.current_function.is_kernel and
            isinstance(range_call.args[0], ast.Call)):
            inner = range_call.args[0]
            if isinstance(inner.func, ast.Name) and inner.func.id == 'len':
                size_expr = self.visit_expr(inner.args[0])
                return IRParallelFor(
                    var_name=var_name,
                    size_expr=size_expr,
                    body=body,
                    lineno=node.lineno
                )
        
        return IRFor(
            var_name=var_name,
            start=start,
            end=end,
            step=step,
            body=body,
            lineno=node.lineno
        )
    
    def visit_parallel_for(self, node: ast.For, var_name: str, len_call: ast.Call) -> IRNode:
        """Обрабатывает параллельный for."""
        size_expr = self.visit_expr(len_call.args[0])
        
        body = []
        self.push_scope()
        for stmt in node.body:
            ir_stmt = self.visit_stmt(stmt)
            if ir_stmt:
                body.append(ir_stmt)
        self.pop_scope()
        
        return IRParallelFor(
            var_name=var_name,
            size_expr=size_expr,
            body=body,
            lineno=node.lineno
        )
    
    def visit_while(self, node: ast.While) -> IRNode:
        """Обрабатывает while."""
        condition = self.visit_expr(node.test)
        
        body = []
        self.push_scope()
        self.loop_depth += 1
        try:
            for stmt in node.body:
                ir_stmt = self.visit_stmt(stmt)
                if ir_stmt:
                    body.append(ir_stmt)
        finally:
            self.loop_depth -= 1
        self.pop_scope()
        
        return IRWhile(
            condition=condition,
            body=body,
            lineno=node.lineno
        )
    
    def visit_return(self, node: ast.Return) -> IRNode:
        """Обрабатывает return."""
        value = None
        if node.value:
            value = self.visit_expr(node.value)
        
        return IRReturn(value=value, lineno=node.lineno)
    
    def visit_expr(self, node: ast.expr) -> IRNode:
        """Обрабатывает выражение."""
        if isinstance(node, ast.Constant):
            return self.visit_constant(node)
        elif isinstance(node, ast.Num):
            # Python 3.7 compatibility
            return self.visit_num(node)
        elif isinstance(node, ast.Name):
            return self.visit_name(node)
        elif isinstance(node, ast.BinOp):
            return self.visit_binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self.visit_unaryop(node)
        elif isinstance(node, ast.Compare):
            return self.visit_compare(node)
        elif isinstance(node, ast.BoolOp):
            return self.visit_boolop(node)
        elif isinstance(node, ast.Call):
            return self.visit_call(node)
        elif isinstance(node, ast.Subscript):
            return self.visit_subscript(node)
        elif isinstance(node, ast.Attribute):
            return self.visit_attribute(node)
        elif isinstance(node, ast.IfExp):
            return self.visit_ifexp(node)
        elif isinstance(node, ast.List):
            return self.visit_list(node)
        elif isinstance(node, ast.Tuple):
            return self.visit_tuple(node)
        elif isinstance(node, ast.NameConstant):
            # Python 3.7 compatibility
            return self.visit_name_constant(node)
        else:
            raise ParseError(f"Unsupported expression: {type(node).__name__}", node)
    
    def visit_constant(self, node: ast.Constant) -> IRNode:
        """Обрабатывает константу."""
        value = node.value
        if isinstance(value, bool):
            return IRConst(value=value, type=BOOL, lineno=node.lineno)
        elif isinstance(value, int):
            return IRConst(value=value, type=INT32, lineno=node.lineno)
        elif isinstance(value, float):
            return IRConst(value=value, type=FLOAT32, lineno=node.lineno)
        else:
            raise ParseError(f"Unsupported constant type: {type(value)}", node)
    
    def visit_num(self, node: ast.Num) -> IRNode:
        """Обрабатывает число (Python 3.7)."""
        value = node.n
        if isinstance(value, int):
            return IRConst(value=value, type=INT32, lineno=node.lineno)
        elif isinstance(value, float):
            return IRConst(value=value, type=FLOAT32, lineno=node.lineno)
        else:
            raise ParseError(f"Unsupported number type: {type(value)}", node)
    
    def visit_name_constant(self, node: ast.NameConstant) -> IRNode:
        """Обрабатывает True/False/None (Python 3.7)."""
        if node.value is True or node.value is False:
            return IRConst(value=node.value, type=BOOL, lineno=node.lineno)
        raise ParseError(f"Unsupported constant: {node.value}", node)
    
    def visit_name(self, node: ast.Name) -> IRNode:
        """Обрабатывает имя."""
        var_type = self.lookup_var(node.id)
        return IRVar(name=node.id, type=var_type, lineno=node.lineno)
    
    def visit_binop(self, node: ast.BinOp) -> IRNode:
        """Обрабатывает бинарную операцию."""
        left = self.visit_expr(node.left)
        right = self.visit_expr(node.right)
        op = self.BINOP_MAP.get(type(node.op), BinaryOp.ADD)
        
        # Вывод типа результата
        result_type = self.infer_binop_type(left.type, right.type, op)
        
        return IRBinOp(
            op=op,
            left=left,
            right=right,
            type=result_type,
            lineno=node.lineno
        )
    
    def visit_unaryop(self, node: ast.UnaryOp) -> IRNode:
        """Обрабатывает унарную операцию."""
        operand = self.visit_expr(node.operand)
        op = self.UNARYOP_MAP.get(type(node.op), UnaryOp.NEG)
        
        return IRUnaryOp(
            op=op,
            operand=operand,
            type=operand.type,
            lineno=node.lineno
        )
    
    def visit_compare(self, node: ast.Compare) -> IRNode:
        """Обрабатывает сравнение."""
        left = self.visit_expr(node.left)
        
        # Обрабатываем цепочки сравнений: a < b < c → (a < b) && (b < c)
        if len(node.ops) == 1:
            right = self.visit_expr(node.comparators[0])
            op = self.CMPOP_MAP.get(type(node.ops[0]), CompareOp.EQ)
            return IRCompare(
                op=op,
                left=left,
                right=right,
                type=BOOL,
                lineno=node.lineno
            )
        else:
            # Цепочка сравнений
            result = None
            prev = left
            for i, (op_node, comp) in enumerate(zip(node.ops, node.comparators)):
                right = self.visit_expr(comp)
                op = self.CMPOP_MAP.get(type(op_node), CompareOp.EQ)
                cmp = IRCompare(op=op, left=prev, right=right, type=BOOL)
                
                if result is None:
                    result = cmp
                else:
                    result = IRBinOp(
                        op=BinaryOp.AND,
                        left=result,
                        right=cmp,
                        type=BOOL
                    )
                prev = right
            
            return result
    
    def visit_boolop(self, node: ast.BoolOp) -> IRNode:
        """Обрабатывает and/or."""
        op = BinaryOp.AND if isinstance(node.op, ast.And) else BinaryOp.OR
        
        result = self.visit_expr(node.values[0])
        for value in node.values[1:]:
            right = self.visit_expr(value)
            result = IRBinOp(op=op, left=result, right=right, type=BOOL)
        
        return result
    
    def visit_call(self, node: ast.Call) -> IRNode:
        """Обрабатывает вызов функции."""
        args = [self.visit_expr(arg) for arg in node.args]
        
        # Вызов метода: obj.method()
        if isinstance(node.func, ast.Attribute):
            obj = self.visit_expr(node.func.value)
            method_name = node.func.attr
            return IRMethodCall(
                obj=obj,
                method_name=method_name,
                args=args,
                lineno=node.lineno
            )
        
        # Обычный вызов функции
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Встроенные функции
            if func_name in ('len', 'range', 'int', 'float', 'bool'):
                return IRCall(func_name=func_name, args=args, lineno=node.lineno)
            
            # Математические функции
            if func_name in ('sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'log2', 'log10',
                           'floor', 'ceil', 'abs', 'pow', 'min', 'max', 'fabs',
                           'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh'):
                return IRCall(func_name=func_name, args=args, type=FLOAT32, lineno=node.lineno)
            
            # Конструктор структуры
            if func_name in self.structs:
                return IRStructInit(
                    struct_name=func_name,
                    lineno=node.lineno
                )
            
            # Пользовательская функция
            return IRCall(func_name=func_name, args=args, lineno=node.lineno)
        
        raise ParseError(f"Unsupported call: {ast.dump(node.func)}", node)
    
    def visit_subscript(self, node: ast.Subscript) -> IRNode:
        """Обрабатывает индексацию."""
        obj = self.visit_expr(node.value)
        
        # Python 3.9+ использует node.slice напрямую
        if isinstance(node.slice, ast.Index):
            # Python 3.7-3.8
            index = self.visit_expr(node.slice.value)
        else:
            index = self.visit_expr(node.slice)
        
        # Определяем тип элемента
        elem_type = None
        if obj.type and obj.type.element_type:
            elem_type = obj.type.element_type
        
        # Tensor Slicing Logic
        if obj.type and obj.type.is_tensor():
            # Determine base name and offset
            base_name = ''
            dim_offset = 0
            
            if isinstance(obj, IRVar):
                if obj.name in self.tensor_aliases:
                    base_name, dim_offset = self.tensor_aliases[obj.name]
                else:
                    base_name = obj.name
            elif isinstance(obj, IRTensorSlice):
                base_name = obj.base_name
                dim_offset = obj.dim_offset + len(obj.slices)
            
            # Helper to normalize indices
            indices = []
            if isinstance(node.slice, ast.Index): # Py3.7/3.8
                slice_val = node.slice.value
                if isinstance(slice_val, ast.Tuple):
                     for e in slice_val.elts:
                         indices.append(self.visit_expr(e))
                else:
                    indices.append(self.visit_expr(slice_val))
            elif isinstance(node.slice, ast.Tuple): # Py3.9+
                for e in node.slice.elts:
                    indices.append(self.visit_expr(e))
            else: # Single index
                indices.append(self.visit_expr(node.slice))
            
            # Determine result type
            # obj.type.rank is the rank of the current object (e.g. 2 for path[i]).
            # result rank = obj.type.rank - len(indices)
            current_rank = obj.type.rank
            res_rank = current_rank - len(indices)
            
            res_type = None
            if res_rank <= 0:
                res_type = obj.type.element_type # Scalar
            else:
                # Create new tensor with same element type but different rank
                res_type = obj.type.element_type.tensor_of(res_rank)
            
            return IRTensorSlice(
                obj=obj,
                slices=indices,
                type=res_type,
                base_name=base_name,
                dim_offset=dim_offset,
                lineno=node.lineno
            )

        if isinstance(obj, IRAttr) and obj.attr_name == 'shape':
            # Handle arr.shape[i]
            target_obj = obj.obj
            base_name = ''
            dim_offset = 0
            
            if isinstance(target_obj, IRVar):
                if target_obj.name in self.tensor_aliases:
                    base_name, dim_offset = self.tensor_aliases[target_obj.name]
                else:
                    base_name = target_obj.name
            
            # Extract constant index
            idx_val = -1
            if isinstance(node.slice, ast.Index):
                 if isinstance(node.slice.value, (ast.Num, ast.Constant)):
                     val = node.slice.value.n if isinstance(node.slice.value, ast.Num) else node.slice.value.value
                     if isinstance(val, int):
                         idx_val = val
            elif isinstance(node.slice, (ast.Num, ast.Constant)):
                 val = node.slice.n if isinstance(node.slice, ast.Num) else node.slice.value
                 if isinstance(val, int):
                     idx_val = val
            
            if idx_val >= 0:
                # Return variable referencing the shape argument
                real_dim = dim_offset + idx_val
                return IRVar(name=f"_shape_{base_name}_{real_dim}", type=INT32, lineno=node.lineno)

        return IRIndex(
            obj=obj,
            index=index,
            type=elem_type,
            lineno=node.lineno
        )
    
    def visit_attribute(self, node: ast.Attribute) -> IRNode:
        """Обрабатывает доступ к атрибуту."""
        obj = self.visit_expr(node.value)
        
        # Определяем тип атрибута
        attr_type = None
        if obj.type and obj.type.is_struct():
            for field_name, field_type in obj.type.fields.items():
                if field_name == node.attr:
                    attr_type = field_type
                    break
        
        return IRAttr(
            obj=obj,
            attr_name=node.attr,
            type=attr_type,
            lineno=node.lineno
        )
    
    def visit_ifexp(self, node: ast.IfExp) -> IRNode:
        """Обрабатывает тернарный оператор."""
        condition = self.visit_expr(node.test)
        then_value = self.visit_expr(node.body)
        else_value = self.visit_expr(node.orelse)
        
        return IRTernary(
            condition=condition,
            then_value=then_value,
            else_value=else_value,
            type=then_value.type,
            lineno=node.lineno
        )
    
    def visit_list(self, node: ast.List) -> IRNode:
        """Обрабатывает список."""
        elements = [self.visit_expr(e) for e in node.elts]
        
        elem_type = elements[0].type if elements else FLOAT32
        array_type = elem_type.array_of(len(elements))
        
        return IRArrayInit(
            elements=elements,
            type=array_type,
            lineno=node.lineno
        )
    
    def visit_tuple(self, node: ast.Tuple) -> IRNode:
        """Обрабатывает кортеж (как массив)."""
        list_node = ast.List(elts=node.elts, ctx=node.ctx)
        list_node.lineno = getattr(node, 'lineno', 0)
        list_node.col_offset = getattr(node, 'col_offset', 0)
        return self.visit_list(list_node)
    
    def parse_type_annotation(self, node: ast.expr) -> CUDAType:
        """Парсит аннотацию типа."""
        if isinstance(node, ast.Name):
            type_name = node.id
            if type_name in BUILTIN_TYPES:
                return BUILTIN_TYPES[type_name]
            if type_name in self.structs:
                struct = self.structs[type_name]
                return CUDAType(
                    kind=TypeKind.STRUCT,
                    name=type_name,
                    cuda_name=type_name,
                    fields={name: t for name, t in struct.fields}
                )
            # Неизвестный тип - предполагаем структуру
            return CUDAType(
                kind=TypeKind.STRUCT,
                name=type_name,
                cuda_name=type_name
            )
        
        elif isinstance(node, ast.Subscript):
            # Array[float32], List[int], Tensor[float32, 3]
            if isinstance(node.value, ast.Name):
                container = node.value.id
                
                # Extract args
                args = []
                if isinstance(node.slice, ast.Index):
                    slice_val = node.slice.value
                    if isinstance(slice_val, ast.Tuple):
                        args = slice_val.elts
                    else:
                        args = [slice_val]
                elif isinstance(node.slice, ast.Tuple):
                    args = node.slice.elts
                else:
                    args = [node.slice]
                
                if container == 'Tensor' and len(args) >= 2:
                    # Tensor[type, rank]
                    elem_type = self.parse_type_annotation(args[0])
                    # Parse Rank
                    rank = 1
                    rank_node = args[1]
                    if isinstance(rank_node, (ast.Num, ast.Constant)):
                         val = rank_node.n if isinstance(rank_node, ast.Num) else rank_node.value
                         if isinstance(val, int):
                             rank = val
                    return elem_type.tensor_of(rank)
                
                if container in ('Array', 'List', 'Tensor'):
                    # Array[type] or Tensor[type] (default rank 1)
                    elem_type = self.parse_type_annotation(args[0])
                    return elem_type.array_of()
        
        return FLOAT32
    
    def infer_binop_type(self, left: CUDAType, right: CUDAType, op: BinaryOp) -> CUDAType:
        """Выводит тип результата бинарной операции."""
        if left is None:
            return right or FLOAT32
        if right is None:
            return left or FLOAT32
        
        # float + int → float
        if left.kind == TypeKind.FLOAT or right.kind == TypeKind.FLOAT:
            return FLOAT32
        
        # int + int → int
        if left.kind == TypeKind.INT and right.kind == TypeKind.INT:
            return INT32
        
        # Pointer arithmetic
        # Array + int -> Array (effectively pointer)
        if (left.is_array() or left.is_pointer()) and right.kind == TypeKind.INT:
            return left
        
        return FLOAT32
