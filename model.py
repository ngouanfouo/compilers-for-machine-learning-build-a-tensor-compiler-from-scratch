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

# Step 5 - exec_alu
import math


def exec_alu(op, dtype, vals):
    if op == Ops.ADD:
        result = vals[0] + vals[1]
    elif op == Ops.MUL:
        result = vals[0] * vals[1]
    elif op == Ops.MAX:
        result = max(vals[0], vals[1])
    elif op == Ops.CMPLT:
        result = vals[0] < vals[1]
    elif op == Ops.AND:
        result = bool(vals[0]) and bool(vals[1])
    elif op == Ops.IDIV:
        # Python floor division; 0 when dividing by zero
        result = vals[0] // vals[1] if vals[1] != 0 else 0
    elif op == Ops.MOD:
        # Python modulo; 0 for a zero divisor
        result = vals[0] % vals[1] if vals[1] != 0 else 0
    elif op == Ops.RECIP:
        x = vals[0]
        result = math.copysign(INF, x) if x == 0 else 1.0 / x
    elif op == Ops.EXP2:
        x = vals[0]
        result = INF if x >= 128 else float(2 ** x)
    elif op == Ops.LOG2:
        x = vals[0]
        if x == 0:
            result = -INF
        elif x < 0:
            result = float("nan")
        else:
            result = math.log2(x)
    elif op == Ops.SQRT:
        x = vals[0]
        result = float("nan") if x < 0 else math.sqrt(x)
    elif op == Ops.CAST:
        result = vals[0]
    elif op == Ops.WHERE:
        result = vals[1] if vals[0] else vals[2]
    else:
        return None
    return UOp.const(dtype, result)


def basic_rules():
    return PatternMatcher([
        # 1) constant fold unary ALU with a constant source
        (UPat(ALU, src=(UPat.cvar('c'),), name='u'),
         lambda u, c: exec_alu(u.op, u.dtype, [c.arg])),

        # 2) constant fold binary ALU with two constant sources
        (UPat(ALU, src=(UPat.cvar('a'), UPat.cvar('b')), name='u'),
         lambda u, a, b: exec_alu(u.op, u.dtype, [a.arg, b.arg])),

        # 3) WHERE with constant condition picks a branch
        (UPat(Ops.WHERE, src=(UPat.cvar('c'), UPat.var('a'), UPat.var('b'))),
         lambda c, a, b: a if c.arg else b),

        # 4) WHERE with identical branches
        (UPat(Ops.WHERE, src=(UPat.var('c'), UPat.var('x'), UPat.var('x'))),
         lambda c, x: x),

        # 5) commutative canonicalization: constant to the right
        (UPat({Ops.ADD, Ops.MUL, Ops.MAX, Ops.AND},
              src=(UPat.cvar('c'), UPat.var('x')), name='u'),
         lambda u, c, x: None if x.op == Ops.CONST else UOp(u.op, u.dtype, (x, c), u.arg)),

        # 6) x + 0 -> x
        (UPat(Ops.ADD, src=(UPat.var('x'), UPat.cvar('c'))),
         lambda x, c: x if c.arg == 0 else None),

        # 7) x * 1 -> x  and  x * 0 -> 0
        (UPat(Ops.MUL, src=(UPat.var('x'), UPat.cvar('c'))),
         lambda x, c: x if c.arg == 1 else (UOp.const(x.dtype, 0) if c.arg == 0 else None)),

        # 8) x // 1 -> x
        (UPat(Ops.IDIV, src=(UPat.var('x'), UPat.cvar('c'))),
         lambda x, c: x if c.arg == 1 else None),

        # 9) x % 1 -> 0
        (UPat(Ops.MOD, src=(UPat.var('x'), UPat.cvar('c'))),
         lambda x, c: UOp.const(x.dtype, 0) if c.arg == 1 else None),

        # 10) x & True -> x  and  x & False -> False
        (UPat(Ops.AND, src=(UPat.var('x'), UPat.cvar('c'))),
         lambda x, c: x if c.arg else UOp.const(x.dtype, False)),

        # 11) max(x, x) -> x
        (UPat(Ops.MAX, src=(UPat.var('x'), UPat.var('x'))),
         lambda x: x),

        # 12) (x + c1) + c2 -> x + (c1 + c2)
        (UPat(Ops.ADD, src=(UPat(Ops.ADD, src=(UPat.var('x'), UPat.cvar('c1'))), UPat.cvar('c2'))),
         lambda x, c1, c2: x + UOp.const(x.dtype, c1.arg + c2.arg)),

        # 13) (x * c1) * c2 -> x * (c1 * c2)
        (UPat(Ops.MUL, src=(UPat(Ops.MUL, src=(UPat.var('x'), UPat.cvar('c1'))), UPat.cvar('c2'))),
         lambda x, c1, c2: x * UOp.const(x.dtype, c1.arg * c2.arg)),

        # 14) (x + c) + y -> (x + y) + c  (y not constant)
        (UPat(Ops.ADD, src=(UPat(Ops.ADD, src=(UPat.var('x'), UPat.cvar('c'))), UPat.var('y'))),
         lambda x, c, y: None if y.op == Ops.CONST else (x + y) + c),

        # 15) x + (y + c) -> (x + y) + c  (y not constant)
        (UPat(Ops.ADD, src=(UPat.var('x'), UPat(Ops.ADD, src=(UPat.var('y'), UPat.cvar('c'))))),
         lambda x, y, c: None if y.op == Ops.CONST else (x + y) + c),

        # 16) int32 distribution: (a + b) * c -> a*c + b*c
        (UPat(Ops.MUL, dtype=dtypes.int32,
              src=(UPat(Ops.ADD, src=(UPat.var('a'), UPat.var('b'))), UPat.cvar('c'))),
         lambda a, b, c: (a * c) + (b * c)),
    ])

# Step 6 - fold_div
def _terms(u):
    if u.op == Ops.ADD:
        return _terms(u.src[0]) + _terms(u.src[1])
    return [u]


def _factor(t):
    if t.op == Ops.CONST:
        return (None, t.arg)
    if t.op == Ops.MUL:
        if t.src[1].op == Ops.CONST:
            return (t.src[0], t.src[1].arg)
        if t.src[0].op == Ops.CONST:
            return (t.src[1], t.src[0].arg)
    return (t, 1)


def _sum(terms, dtype):
    if not terms:
        return UOp.const(dtype, 0)
    result = terms[0]
    for t in terms[1:]:
        result = result + t
    return result


def fold_div(x, c):
    if c.op != Ops.CONST or c.arg <= 0:
        return None
    dtype = x.dtype
    cval = c.arg
    terms = _terms(x)

    quotients = []
    R = []
    for t in terms:
        base, k = _factor(t)
        if k % cval == 0:
            q = k // cval
            if base is None:
                quotients.append(UOp.const(dtype, q))
            elif q == 1:
                quotients.append(base)
            else:
                quotients.append(base * UOp.const(dtype, q))
        else:
            R.append(t)

    # No divisible terms, single term y * k with k > 1 and c % k == 0
    if not quotients and len(terms) == 1:
        base, k = _factor(terms[0])
        if base is not None and k > 1 and cval % k == 0:
            return base // UOp.const(dtype, cval // k)

    R_uop = _sum(R, dtype)
    vmin, vmax = bounds(R_uop)
    if vmin >= 0 and vmax < cval:
        return _sum(quotients, dtype)
    if quotients:
        return _sum(quotients, dtype) + (R_uop // c)
    return None


def fold_mod(x, c):
    if c.op != Ops.CONST or c.arg <= 0:
        return None
    dtype = x.dtype
    cval = c.arg
    terms = _terms(x)

    R = []
    dropped = False
    for t in terms:
        _, k = _factor(t)
        if k % cval == 0:
            dropped = True
        else:
            R.append(t)

    R_uop = _sum(R, dtype)
    vmin, vmax = bounds(R_uop)
    if vmin >= 0 and vmax < cval:
        return R_uop
    if dropped:
        return R_uop % c
    return None


def fold_cmplt(x, y):
    vmin_x, vmax_x = bounds(x)
    vmin_y, vmax_y = bounds(y)
    if vmax_x < vmin_y:
        return UOp.const(dtypes.bool, True)
    if vmin_x >= vmax_y:
        return UOp.const(dtypes.bool, False)
    return None


def fold_max(x, y):
    vmin_x, vmax_x = bounds(x)
    vmin_y, vmax_y = bounds(y)
    if vmin_x >= vmax_y:
        return x
    if vmin_y >= vmax_x:
        return y
    return None

# Step 7 - make_symbolic
def make_symbolic():
    return basic_rules() + PatternMatcher([
        # fold_div on IDIV of an int32 expression by a constant
        (UPat(Ops.IDIV, src=(UPat.var('x', dtype=dtypes.int32), UPat.cvar('c'))),
         lambda x, c: fold_div(x, c)),

        # fold_mod on MOD likewise
        (UPat(Ops.MOD, src=(UPat.var('x', dtype=dtypes.int32), UPat.cvar('c'))),
         lambda x, c: fold_mod(x, c)),

        # fold_cmplt on CMPLT whose left operand is int32
        (UPat(Ops.CMPLT, src=(UPat.var('x', dtype=dtypes.int32), UPat.var('y'))),
         lambda x, y: fold_cmplt(x, y)),

        # fold_max on MAX with an int32 left operand
        (UPat(Ops.MAX, src=(UPat.var('x', dtype=dtypes.int32), UPat.var('y'))),
         lambda x, y: fold_max(x, y)),
    ])


symbolic = make_symbolic()


def simplify(u):
    return graph_rewrite(u, symbolic)

# Step 8 - render_sink
import math
import numpy as np


def _is_float(dt):
    for name in ("float32", "float16", "float64", "bfloat16"):
        f = getattr(dtypes, name, None)
        if f is not None and dt == f:
            return True
    return False


def _ctype(dt):
    return "float" if _is_float(dt) else "int"


def param(name, dtype, i):
    return UOp(Ops.PARAM, dtype, (), (name, i))


def load(p, idx):
    return UOp(Ops.LOAD, p.dtype,
               (UOp(Ops.INDEX, p.dtype, (p, UOp.const(dtypes.int32, idx))),))


def store(p, idx, val):
    return UOp(Ops.STORE, None,
               (UOp(Ops.INDEX, p.dtype, (p, UOp.const(dtypes.int32, idx))), val))


def sink(*stores):
    return UOp(Ops.SINK, None, stores)


def c_literal(dtype, v):
    if _is_float(dtype):
        fv = float(np.float32(v))
        if math.isnan(fv):
            return "NAN"
        if math.isinf(fv):
            return "(-INFINITY)" if fv < 0 else "INFINITY"
        return f"{fv!r}f"
    return str(int(v))


class CRenderer:
    def __init__(self):
        self.lines, self.scopes, self.n = [], [{}], 0

    def push(self):
        self.scopes.append({})

    def pop(self):
        self.scopes.pop()

    def lookup(self, u):
        for scope in reversed(self.scopes):
            if u in scope:
                return scope[u]
        return None

    def emit(self, line):
        self.lines.append("  " * len(self.scopes) + line)

    def new_var(self, u, expr):
        name = f"v{self.n}"
        self.n += 1
        self.emit(f"{_ctype(u.dtype)} {name} = {expr};")
        self.scopes[-1][u] = name
        return name

    def expr(self, u):
        if (name := self.lookup(u)) is not None:
            return name
        op = u.op
        if op == Ops.CONST:
            return c_literal(u.dtype, u.arg)
        if op == Ops.RANGE:
            return f"r{u.arg}"
        if op == Ops.SPECIAL:
            return u.arg[1]
        if op == Ops.DEFINE_ACC:
            return f"acc{u.arg}"
        if op == Ops.LOAD:
            index = u.src[0]
            p, idx = index.src[0], index.src[1]
            return self.new_var(u, f"data{p.arg[1]}[{self.expr(idx)}]")
        if op == Ops.WHERE:
            c, a, b = u.src
            cc, aa, bb = self.expr(c), self.expr(a), self.expr(b)
            return self.new_var(u, f"({cc} ? {aa} : {bb})")
        if op == Ops.CAST:
            x = self.expr(u.src[0])
            return self.new_var(u, f"({_ctype(u.dtype)})({x})")
        if op == Ops.RECIP:
            x = self.expr(u.src[0])
            return self.new_var(u, f"(1.0f/{x})")
        if op == Ops.EXP2:
            x = self.expr(u.src[0])
            return self.new_var(u, f"exp2f({x})")
        if op == Ops.LOG2:
            x = self.expr(u.src[0])
            return self.new_var(u, f"log2f({x})")
        if op == Ops.SQRT:
            x = self.expr(u.src[0])
            return self.new_var(u, f"sqrtf({x})")
        if op == Ops.MAX:
            a, b = self.expr(u.src[0]), self.expr(u.src[1])
            if _is_float(u.dtype):
                return self.new_var(u, f"fmaxf({a}, {b})")
            return self.new_var(u, f"({a} > {b} ? {a} : {b})")
        if op in (Ops.IDIV, Ops.MOD):
            a, b = self.expr(u.src[0]), self.expr(u.src[1])
            vmin, _ = bounds(u.src[0])
            if vmin >= 0:
                sym = "/" if op == Ops.IDIV else "%"
                return self.new_var(u, f"({a} {sym} {b})")
            fn = "fdiv" if op == Ops.IDIV else "fmod_floor"
            return self.new_var(u, f"{fn}({a}, {b})")
        if op == Ops.ADD:
            a, b = self.expr(u.src[0]), self.expr(u.src[1])
            return self.new_var(u, f"({a} + {b})")
        if op == Ops.MUL:
            a, b = self.expr(u.src[0]), self.expr(u.src[1])
            return self.new_var(u, f"({a} * {b})")
        if op == Ops.CMPLT:
            a, b = self.expr(u.src[0]), self.expr(u.src[1])
            return self.new_var(u, f"({a} < {b})")
        if op == Ops.AND:
            a, b = self.expr(u.src[0]), self.expr(u.src[1])
            return self.new_var(u, f"({a} && {b})")
        raise NotImplementedError(f"expr: {op}")


C_HEADER = """#include <math.h>
static inline int fdiv(int a, int b) {
  int q = a / b, r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}
static inline int fmod_floor(int a, int b) {
  int r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}
"""


def params_of(u):
    seen = {}

    def walk(x):
        if x.op == Ops.PARAM:
            seen[x.arg[1]] = x
        for s in x.src:
            walk(s)

    walk(u)
    return [seen[k] for k in sorted(seen)]


def signature(name, ps, written):
    args = []
    for p in ps:
        pos = p.arg[1]
        const = "" if pos in written else "const "
        args.append(f"{const}{_ctype(p.dtype)}* restrict data{pos}")
    return f"void {name}({', '.join(args)})"


def render_sink(s, name="kernel"):
    ps = params_of(s)
    written = {st.src[0].src[0].arg[1] for st in s.src}
    sig = signature(name, ps, written)
    r = CRenderer()
    r.push()
    for st in s.src:
        index, val = st.src
        p, idx = index.src
        idx_s = r.expr(idx)
        rhs = r.expr(val)
        r.emit(f"data{p.arg[1]}[{idx_s}] = {rhs};")
    r.pop()
    return sig + " {\n" + "\n".join(r.lines) + "\n}\n"

# Step 9 - eval_sink
import numpy as np


def eval_sink(s, bufs):
    memo = {}

    def ev(u):
        if u in memo:
            return memo[u]
        op = u.op
        if op == Ops.CONST:
            r = u.arg
        elif op == Ops.INDEX:
            r = int(ev(u.src[1]))
        elif op == Ops.LOAD:
            index = u.src[0]              # INDEX
            p = index.src[0]              # PARAM
            i = int(ev(index.src[1]))
            r = bufs[p.arg[0]].reshape(-1)[i].item()
        elif op in ALU:
            vals = [ev(x) for x in u.src]
            r = exec_alu(op, u.dtype, vals).arg
        else:
            raise NotImplementedError(f"eval_sink: {op}")
        memo[u] = r
        return r

    for st in s.src:                       # STORE nodes, in order
        index, val = st.src                # INDEX, value
        p = index.src[0]                   # PARAM
        i = int(ev(index.src[1]))
        bufs[p.arg[0]].reshape(-1)[i] = ev(val)
    return bufs

# Step 10 - compile_c
import os
import shutil
import subprocess
import tempfile
import hashlib
import ctypes
import numpy as np

_CC = None
_LIBS = {}


def compiler():
    global _CC
    if _CC is not None:
        return _CC
    cc = os.environ.get("CC")
    if cc:
        _CC = cc
    else:
        for cand in ("clang", "gcc", "cc"):
            path = shutil.which(cand)
            if path is not None:
                _CC = path
                break
    if _CC is None:
        raise RuntimeError("no C compiler found (set CC or install clang/gcc/cc)")
    return _CC


def compile_c(src):
    full = C_HEADER + src
    key = hashlib.sha1(full.encode()).hexdigest()
    if key in _LIBS:
        return _LIBS[key]

    tmpdir = tempfile.mkdtemp(prefix="kern")
    c_path = os.path.join(tmpdir, "kernel.c")
    so_path = os.path.join(tmpdir, "kernel.so")
    with open(c_path, "w") as f:
        f.write(full)

    cmd = [compiler(), "-O3", "-march=native", "-shared", "-fPIC", "-w",
           c_path, "-o", so_path, "-lm"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())

    lib = ctypes.CDLL(so_path)
    _LIBS[key] = lib
    return lib


def call_kernel(lib, name, bufs):
    fn = getattr(lib, name)
    args = [ctypes.c_void_p(arr.ctypes.data) for arr in bufs]
    fn(*args)


def run_sink(s, bufs, name="kernel"):
    ps = params_of(s)                        # sorted by pointer position
    arrays = [bufs[p.arg[0]] for p in ps]    # in parameter order
    src = render_sink(s, name)
    lib = compile_c(src)
    call_kernel(lib, name, arrays)
    return bufs

# Step 11 - shape_of
import functools
import math


def buffer(name, shape, dtype=dtypes.float32):
    return UOp(Ops.BUFFER, dtype, (), (name, tuple(int(s) for s in shape)))


def strides_of(shape):
    strides = []
    acc = 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= s
    return tuple(reversed(strides))


@functools.lru_cache(maxsize=None)
def shape_of(u):
    op = u.op
    if op == Ops.BUFFER:
        return u.arg[1]
    if op == Ops.CONST:
        return ()
    if op in (Ops.RESHAPE, Ops.EXPAND):
        return tuple(u.arg)
    if op == Ops.PERMUTE:
        src = shape_of(u.src[0])
        return tuple(src[i] for i in u.arg)
    if op == Ops.FLIP:
        return shape_of(u.src[0])
    if op == Ops.PAD:
        src = shape_of(u.src[0])
        return tuple(s + lo + hi for s, (lo, hi) in zip(src, u.arg))
    if op == Ops.SHRINK:
        src = shape_of(u.src[0])
        return tuple(e - b for s, (b, e) in zip(src, u.arg))
    if op == Ops.REDUCE_AXIS:
        src = shape_of(u.src[0])
        _, axes = u.arg
        return tuple(1 if i in axes else s for i, s in enumerate(src))
    if op == Ops.WHERE:
        return shape_of(u.src[1])
    if op in ALU:
        return shape_of(u.src[0])
    raise NotImplementedError(f"shape_of: {op}")

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

