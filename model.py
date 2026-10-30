"""
Compilers for Machine Learning: Build a Tensor Compiler from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - UOp
import sys
import math
from enum import Enum, auto
import numpy as np

sys.setrecursionlimit(100000)
INF = float("inf")


class DType:
    def __init__(self, name, c_name, np_type):
        self.name, self.c_name, self.np = name, c_name, np_type
    def __repr__(self): return "dtypes." + self.name


class dtypes:
    bool = DType("bool", "int", np.int32)
    int32 = DType("int32", "int", np.int32)
    float32 = DType("float32", "float", np.float32)


class Ops(Enum):
    CONST = auto(); PARAM = auto(); BUFFER = auto()
    ADD = auto(); MUL = auto(); MAX = auto(); CMPLT = auto(); AND = auto(); IDIV = auto(); MOD = auto()
    RECIP = auto(); EXP2 = auto(); LOG2 = auto(); SQRT = auto(); CAST = auto(); WHERE = auto()
    RESHAPE = auto(); EXPAND = auto(); PERMUTE = auto(); FLIP = auto(); PAD = auto(); SHRINK = auto(); REDUCE_AXIS = auto()
    RANGE = auto(); SPECIAL = auto(); INDEX = auto(); LOAD = auto(); STORE = auto(); DEFINE_ACC = auto(); SINK = auto()


BINARY = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPLT, Ops.AND, Ops.IDIV, Ops.MOD}
UNARY = {Ops.RECIP, Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.CAST}
ALU = BINARY | UNARY | {Ops.WHERE}
MOVEMENT = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.FLIP, Ops.PAD, Ops.SHRINK}


def _hashable(v):
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(val)) for k, val in v.items()))
    return v


class UOp:
    __slots__ = ("op", "dtype", "src", "arg")
    _cache = {}

    def __new__(cls, op, dtype=None, src=(), arg=None):
        src = tuple(src)
        key = (op, dtype, src, _hashable(arg))
        cached = cls._cache.get(key)
        if cached is not None:
            return cached
        obj = object.__new__(cls)
        cls._cache[key] = obj
        return obj

    def __init__(self, op, dtype=None, src=(), arg=None):
        if getattr(self, "op", None) is not None:
            return
        self.op = op
        self.dtype = dtype
        self.src = tuple(src)
        self.arg = arg

    @staticmethod
    def const(dtype, v):
        if dtype is dtypes.bool:
            v = bool(v)
        elif dtype is dtypes.int32:
            v = int(v)
        elif dtype is dtypes.float32:
            v = float(v)
        return UOp(Ops.CONST, dtype, (), v)

    @staticmethod
    def range(n, i):
        return UOp(Ops.RANGE, dtypes.int32, (UOp.const(dtypes.int32, n),), i)

    def alu(self, op, *src):
        wrapped = [self]
        for s in src:
            wrapped.append(s if isinstance(s, UOp) else UOp.const(self.dtype, s))
        out_dtype = dtypes.bool if op in (Ops.CMPLT, Ops.AND) else self.dtype
        return UOp(op, out_dtype, tuple(wrapped))

    def __add__(self, other):
        return self.alu(Ops.ADD, other)
    def __radd__(self, other):
        return UOp.const(self.dtype, other).alu(Ops.ADD, self)
    def __mul__(self, other):
        return self.alu(Ops.MUL, other)
    def __rmul__(self, other):
        return UOp.const(self.dtype, other).alu(Ops.MUL, self)
    def __floordiv__(self, other):
        return self.alu(Ops.IDIV, other)
    def __mod__(self, other):
        return self.alu(Ops.MOD, other)
    def __lt__(self, other):
        return self.alu(Ops.CMPLT, other)
    def __and__(self, other):
        return self.alu(Ops.AND, other)
    def __neg__(self):
        if self.op is Ops.CONST:
            return UOp.const(self.dtype, -self.arg)
        return self * -1
    def __sub__(self, other):
        return self + (-other)

    def maximum(self, other):
        return self.alu(Ops.MAX, other)
    def recip(self):
        return self.alu(Ops.RECIP)
    def exp2(self):
        return self.alu(Ops.EXP2)
    def log2(self):
        return self.alu(Ops.LOG2)
    def sqrt(self):
        return self.alu(Ops.SQRT)
    def cast(self, dtype):
        return UOp(Ops.CAST, dtype, (self,))

    def where(self, a, b):
        if isinstance(a, UOp):
            base = a.dtype
        elif isinstance(b, UOp):
            base = b.dtype
        else:
            base = dtypes.float32
        if not isinstance(a, UOp):
            a = UOp.const(base, a)
        if not isinstance(b, UOp):
            b = UOp.const(base, b)
        return UOp(Ops.WHERE, a.dtype, (self, a, b))

    def toposort(self):
        order = []
        seen = set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            if node in seen:
                continue
            seen.add(node)
            stack.append((node, True))
            for s in reversed(node.src):
                if s not in seen:
                    stack.append((s, False))
        return order

    def __repr__(self):
        src_repr = "(" + ", ".join(repr(s) for s in self.src) + ")"
        if self.arg is not None:
            return f"UOp({self.op.name}, {self.dtype}, {src_repr}, arg={self.arg})"
        return f"UOp({self.op.name}, {self.dtype}, {src_repr})"

# Step 2 - bounds (not yet solved)
# TODO: implement

# Step 3 - UPat (not yet solved)
# TODO: implement

# Step 4 - graph_rewrite (not yet solved)
# TODO: implement

# Step 5 - exec_alu (not yet solved)
# TODO: implement

# Step 6 - fold_div (not yet solved)
# TODO: implement

# Step 7 - make_symbolic (not yet solved)
# TODO: implement

# Step 8 - render_sink (not yet solved)
# TODO: implement

# Step 9 - eval_sink (not yet solved)
# TODO: implement

# Step 10 - compile_c (not yet solved)
# TODO: implement

# Step 11 - shape_of (not yet solved)
# TODO: implement

# Step 12 - Tensor (not yet solved)
# TODO: implement

# Step 13 - eval_tensor (not yet solved)
# TODO: implement

# Step 14 - conv2d (not yet solved)
# TODO: implement

# Step 15 - Kernel (not yet solved)
# TODO: implement

# Step 16 - Lowerer (not yet solved)
# TODO: implement

# Step 17 - ReduceLowerer (not yet solved)
# TODO: implement

# Step 18 - run_kernel_np (not yet solved)
# TODO: implement

# Step 19 - render_kernel (not yet solved)
# TODO: implement

# Step 20 - schedule (not yet solved)
# TODO: implement

# Step 21 - Compiled (not yet solved)
# TODO: implement

# Step 22 - split_range (not yet solved)
# TODO: implement

# Step 23 - unroll_output (not yet solved)
# TODO: implement

# Step 24 - unroll_reduce (not yet solved)
# TODO: implement

# Step 25 - optimize_gemm (not yet solved)
# TODO: implement

# Step 26 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 27 - to_gpu (not yet solved)
# TODO: implement

# Step 28 - render_cuda (not yet solved)
# TODO: implement

# Step 29 - grad_alu (not yet solved)
# TODO: implement

# Step 30 - grad_movement (not yet solved)
# TODO: implement

# Step 31 - backward (not yet solved)
# TODO: implement

# Step 32 - gpt_forward (not yet solved)
# TODO: implement

# Step 33 - build_train_program (not yet solved)
# TODO: implement

# Step 34 - train_gpt (not yet solved)
# TODO: implement

