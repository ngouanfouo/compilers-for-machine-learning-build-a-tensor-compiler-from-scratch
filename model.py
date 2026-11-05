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

# Step 2 - bounds
import functools
import math

@functools.lru_cache(maxsize=None)
def bounds(u):
    # Floats (and anything not int/bool) are unknown.
    if u.dtype not in (dtypes.int32, dtypes.bool):
        return (-INF, INF)

    op = u.op

    if op is Ops.CONST:
        v = u.arg
        if isinstance(v, bool):
            v = int(v)
        return (v, v)

    if op in (Ops.RANGE, Ops.SPECIAL):
        n = u.src[0].arg
        return (0, n - 1)

    if op is Ops.ADD:
        a = bounds(u.src[0])
        b = bounds(u.src[1])
        return (a[0] + b[0], a[1] + b[1])

    if op is Ops.MUL:
        a = bounds(u.src[0])
        b = bounds(u.src[1])
        candidates = []
        for x in a:
            for y in b:
                if (x == 0 and math.isinf(y)) or (y == 0 and math.isinf(x)):
                    continue
                candidates.append(x * y)
        if not candidates:
            return (-INF, INF)
        return (min(candidates), max(candidates))

    if op is Ops.MAX:
        a = bounds(u.src[0])
        b = bounds(u.src[1])
        return (max(a[0], b[0]), max(a[1], b[1]))

    if op is Ops.IDIV:
        a = bounds(u.src[0])
        c = u.src[1].arg
        if c > 0:
            return (math.floor(a[0] / c), math.floor(a[1] / c))
        return (-INF, INF)

    if op is Ops.MOD:
        a = bounds(u.src[0])
        c = u.src[1].arg
        if c > 0:
            if 0 <= a[0] and a[1] < c:
                return a
            return (0, c - 1)
        return (-INF, INF)

    if op is Ops.CMPLT:
        a = bounds(u.src[0])
        b = bounds(u.src[1])
        if a[1] < b[0]:
            return (1, 1)
        if a[0] >= b[1]:
            return (0, 0)
        return (0, 1)

    if op is Ops.AND:
        a = bounds(u.src[0])
        b = bounds(u.src[1])
        if a[0] == 1 and b[0] == 1:
            return (1, 1)
        if a[1] == 0 or b[1] == 0:
            return (0, 0)
        return (0, 1)

    if op is Ops.WHERE:
        cond = bounds(u.src[0])
        a = bounds(u.src[1])
        b = bounds(u.src[2])
        if cond == (1, 1):
            return a
        if cond == (0, 0):
            return b
        return (min(a[0], b[0]), max(a[1], b[1]))

    if op is Ops.CAST:
        return bounds(u.src[0])

    return (-INF, INF)

# Step 3 - UPat
_UNSET = object()


class UPat:
    def __init__(self, op=None, dtype=None, src=None, arg=_UNSET, name=None):
        self.op = op
        self.dtype = dtype
        self.src = src
        self.arg = arg
        self.name = name

    @staticmethod
    def var(name=None, dtype=None):
        return UPat(dtype=dtype, name=name)

    @staticmethod
    def cvar(name=None, dtype=None):
        return UPat(Ops.CONST, dtype=dtype, name=name)

    def match(self, u, store):
        if self.op is not None:
            if isinstance(self.op, (set, frozenset)):
                if u.op not in self.op:
                    return False
            elif u.op != self.op:
                return False

        if self.dtype is not None and u.dtype != self.dtype:
            return False

        if self.arg is not _UNSET and u.arg != self.arg:
            return False

        if self.src is not None:
            u_src = u.src if u.src is not None else ()
            if len(self.src) != len(u_src):
                return False
            for p, s in zip(self.src, u_src):
                if not p.match(s, store):
                    return False

        if self.name is not None:
            if self.name in store:
                if store[self.name] is not u:
                    return False
            else:
                store[self.name] = u

        return True


class PatternMatcher:
    def __init__(self, patterns):
        self.patterns = list(patterns)

    def __add__(self, other):
        return PatternMatcher(self.patterns + list(other.patterns))

    def rewrite(self, u):
        for pat, fn in self.patterns:
            store = {}
            if pat.match(u, store):
                result = fn(**store)
                if result is not None and result is not u:
                    return result
        return None

# Step 4 - graph_rewrite
import functools


def graph_rewrite(root, pm):
    @functools.cache
    def rw(u):
        # 1) rewrite sources first (bottom-up)
        old_src = tuple(u.src) if u.src is not None else ()
        srcs = tuple(rw(s) for s in old_src)

        # 2) rebuild with new sources only when something actually changed.
        #    The UOp constructor hash-conses, so this preserves canonical identity.
        ret = UOp(u.op, u.dtype, srcs, u.arg) if srcs != old_src else u

        # 3) fixed-point loop at the current node
        for _ in range(1000):
            rewritten = pm.rewrite(ret)
            if rewritten is None:
                break
            # the replacement's sources may be unprocessed
            ret = rw(rewritten) if rewritten.src else rewritten
            # stop when the rule returned a node that rw maps to itself
            if ret is rewritten:
                break
        return ret

    return rw(root)

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

