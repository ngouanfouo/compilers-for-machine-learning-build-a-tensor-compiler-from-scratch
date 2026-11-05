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

# Step 12 - Tensor
import math


class Tensor:
    def __init__(self, uop):
        self.uop = uop

    @staticmethod
    def input(name, shape, dtype=dtypes.float32):
        return Tensor(buffer(name, shape, dtype))

    @staticmethod
    def const(v, dtype=dtypes.float32):
        return Tensor(UOp.const(dtype, v))

    @property
    def shape(self):
        return shape_of(self.uop)

    @property
    def dtype(self):
        return self.uop.dtype

    def __repr__(self):
        return f"Tensor(shape={self.shape}, {self.dtype})"

    # ---------------- movement ops ----------------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        if -1 in shape:
            idx = shape.index(-1)
            known = 1
            for i, s in enumerate(shape):
                if i != idx:
                    known *= s
            total = 1
            for s in self.shape:
                total *= s
            shape = shape[:idx] + (total // known,) + shape[idx + 1:]
        prod = 1
        for s in shape:
            prod *= s
        prod0 = 1
        for s in self.shape:
            prod0 *= s
        assert prod == prod0, f"reshape {self.shape} -> {shape}"
        if shape == self.shape:
            return self
        # consecutive reshapes collapse to the original source
        if self.uop.op == Ops.RESHAPE:
            src = self.uop.src[0]
            if shape == shape_of(src):
                return Tensor(src)
        else:
            src = self.uop
        return Tensor(UOp(Ops.RESHAPE, self.dtype, (src,), shape))

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        assert len(shape) == len(self.shape), f"expand {self.shape} -> {shape}"
        for a, b in zip(self.shape, shape):
            assert a == b or a == 1, f"expand {self.shape} -> {shape}"
        if shape == self.shape:
            return self
        return Tensor(UOp(Ops.EXPAND, self.dtype, (self.uop,), shape))

    def permute(self, *perm):
        if len(perm) == 1 and isinstance(perm[0], (tuple, list)):
            perm = tuple(perm[0])
        else:
            perm = tuple(perm)
        ndim = len(self.shape)
        perm = tuple(p % ndim for p in perm)
        if perm == tuple(range(ndim)):
            return self
        return Tensor(UOp(Ops.PERMUTE, self.dtype, (self.uop,), perm))

    def flip(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes)
        ndim = len(self.shape)
        axes = tuple(sorted({a % ndim for a in axes}))
        if not axes:
            return self
        return Tensor(UOp(Ops.FLIP, self.dtype, (self.uop,), axes))

    def pad(self, pads):
        pads = tuple((int(lo), int(hi)) for lo, hi in pads)
        if all(lo == 0 and hi == 0 for lo, hi in pads):
            return self
        return Tensor(UOp(Ops.PAD, self.dtype, (self.uop,), pads))

    def shrink(self, ranges):
        ranges = tuple((int(b), int(e)) for b, e in ranges)
        for (b, e), s in zip(ranges, self.shape):
            assert 0 <= b <= e <= s, f"shrink {ranges} on {self.shape}"
        if all(b == 0 and e == s for (b, e), s in zip(ranges, self.shape)):
            return self
        return Tensor(UOp(Ops.SHRINK, self.dtype, (self.uop,), ranges))

    def transpose(self, a=-2, b=-1):
        ndim = len(self.shape)
        a %= ndim
        b %= ndim
        perm = list(range(ndim))
        perm[a], perm[b] = perm[b], perm[a]
        return self.permute(*perm)

    # ---------------- broadcasting ----------------
    def _expand_to(self, target):
        s = self.shape
        if len(s) < len(target):
            new = (1,) * (len(target) - len(s)) + s
            t = self.reshape(new)
        else:
            t = self
        if t.shape == target:
            return t
        return t.expand(target)

    def _bcast(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)
        shapes = [self.shape, other.shape]
        rank = max(len(s) for s in shapes)
        padded = [(1,) * (rank - len(s)) + s for s in shapes]
        out = tuple(max(c) for c in zip(*padded))
        return self._expand_to(out), other._expand_to(out)

    def _alu(self, op, *others):
        # number-wrap dtype
        if op == Ops.WHERE:
            wrap_dtype = dtypes.float32
            for t in others:
                if isinstance(t, Tensor):
                    wrap_dtype = t.dtype
                    break
        else:
            wrap_dtype = self.dtype
        tensors = []
        for o in (self,) + others:
            if isinstance(o, Tensor):
                tensors.append(o)
            else:
                tensors.append(Tensor.const(o, wrap_dtype))
        rank = max(len(t.shape) for t in tensors)
        padded = [(1,) * (rank - len(t.shape)) + t.shape for t in tensors]
        out = tuple(max(c) for c in zip(*padded))
        srcs = tuple(t._expand_to(out).uop for t in tensors)
        if op in (Ops.CMPLT, Ops.AND):
            dt = dtypes.bool
        elif op == Ops.WHERE:
            dt = tensors[1].dtype
        else:
            dt = self.dtype
        return Tensor(UOp(op, dt, srcs))

    # ---------------- arithmetic / elementwise ----------------
    def __add__(self, other):
        return self._alu(Ops.ADD, other)

    def __radd__(self, other):
        return self._alu(Ops.ADD, other)

    def __mul__(self, other):
        return self._alu(Ops.MUL, other)

    def __rmul__(self, other):
        return self._alu(Ops.MUL, other)

    def __neg__(self):
        return self._alu(Ops.MUL, -1)

    def __sub__(self, other):
        return self._alu(Ops.ADD, -other)

    def __rsub__(self, other):
        return (-self) + other

    def __floordiv__(self, other):
        return self._alu(Ops.IDIV, other)

    def __mod__(self, other):
        return self._alu(Ops.MOD, other)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)
        return self * other.recip()

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)
        return other * self.recip()

    def __lt__(self, other):
        return self._alu(Ops.CMPLT, other)

    def __gt__(self, other):
        if isinstance(other, Tensor):
            return other._alu(Ops.CMPLT, self)
        return Tensor.const(other, self.dtype)._alu(Ops.CMPLT, self)

    def __and__(self, other):
        return self._alu(Ops.AND, other)

    def maximum(self, other):
        return self._alu(Ops.MAX, other)

    def where(self, a, b):
        return self._alu(Ops.WHERE, a, b)

    def recip(self):
        return Tensor(UOp(Ops.RECIP, self.dtype, (self.uop,)))

    def exp2(self):
        return Tensor(UOp(Ops.EXP2, self.dtype, (self.uop,)))

    def log2(self):
        return Tensor(UOp(Ops.LOG2, self.dtype, (self.uop,)))

    def sqrt(self):
        return Tensor(UOp(Ops.SQRT, self.dtype, (self.uop,)))

    def exp(self):
        return (self * math.log2(math.e)).exp2()

    def log(self):
        return self.log2() * math.log(2)

    def relu(self):
        return self.maximum(0)

    def square(self):
        return self * self

    def cast(self, dtype):
        return Tensor(UOp(Ops.CAST, dtype, (self.uop,)))

    # ---------------- reductions ----------------
    def _reduce(self, op, axis, keepdim):
        ndim = len(self.shape)
        if axis is None:
            axes = tuple(range(ndim))
        else:
            if isinstance(axis, int):
                axis = (axis,)
            axes = tuple(sorted({a % ndim for a in axis}))
        reduced_shape = tuple(1 if i in axes else s for i, s in enumerate(self.shape))
        r = Tensor(UOp(Ops.REDUCE_AXIS, self.dtype, (self.uop,), (op, axes)))
        if keepdim:
            return r
        newshape = tuple(s for i, s in enumerate(self.shape) if i not in axes)
        if newshape == reduced_shape:
            return r
        return r.reshape(newshape)

    def sum(self, axis=None, keepdim=False):
        return self._reduce(Ops.ADD, axis, keepdim)

    def max(self, axis=None, keepdim=False):
        return self._reduce(Ops.MAX, axis, keepdim)

    def mean(self, axis=None, keepdim=False):
        s = self.sum(axis, keepdim)
        ndim = len(self.shape)
        if axis is None:
            axes = tuple(range(ndim))
        else:
            if isinstance(axis, int):
                axis = (axis,)
            axes = tuple({a % ndim for a in axis})
        count = 1
        for a in axes:
            count *= self.shape[a]
        return s * (1.0 / count)

    # ---------------- composites ----------------
    def matmul(self, w):
        M, K = self.shape[-2], self.shape[-1]
        N = w.shape[-1]
        a = self.reshape(self.shape[:-2] + (M, 1, K))
        wt = w.transpose()  # (..., N, K)
        b = wt.reshape(wt.shape[:-2] + (1, N, K))
        return (a * b).sum(-1)

    def __matmul__(self, w):
        return self.matmul(w)

    def softmax(self, axis=-1):
        m = self.max(axis, keepdim=True)
        e = (self - m).exp()
        s = e.sum(axis, keepdim=True)
        return e / s

    def logsoftmax(self, axis=-1):
        m = self.max(axis, keepdim=True)
        z = self - m
        return z - z.exp().sum(axis, keepdim=True).log()

    def layernorm(self, eps=1e-5):
        mu = self.mean(-1, keepdim=True)
        d = self - mu
        var = (d * d).mean(-1, keepdim=True)
        return d / (var + eps).sqrt()

# Step 13 - eval_tensor
import numpy as np


def _npdtype(dt):
    return getattr(dt, "np", dt)


def _alu_np(op, s):
    if op == Ops.ADD:
        return s[0] + s[1]
    if op == Ops.MUL:
        return s[0] * s[1]
    if op == Ops.MAX:
        return np.maximum(s[0], s[1])
    if op == Ops.CMPLT:
        return s[0] < s[1]
    if op == Ops.AND:
        return np.logical_and(s[0], s[1])
    if op == Ops.IDIV:
        return np.floor_divide(s[0], s[1])
    if op == Ops.MOD:
        return np.mod(s[0], s[1])
    if op == Ops.RECIP:
        return np.reciprocal(s[0])
    if op == Ops.EXP2:
        return np.exp2(s[0])
    if op == Ops.LOG2:
        return np.log2(s[0])
    if op == Ops.SQRT:
        return np.sqrt(s[0])
    if op == Ops.CAST:
        return s[0]
    if op == Ops.WHERE:
        return np.where(s[0], s[1], s[2])
    raise NotImplementedError(f"alu: {op}")


def eval_tensor(u, bufs, cache=None):
    if cache is None:
        cache = {}
    if u in cache:
        return cache[u]

    op = u.op
    if op == Ops.BUFFER:
        name, shape = u.arg
        arr = np.asarray(bufs[name]).astype(_npdtype(u.dtype)).reshape(shape)
    elif op == Ops.CONST:
        arr = np.array(u.arg, dtype=_npdtype(u.dtype))
    elif op == Ops.RESHAPE:
        arr = eval_tensor(u.src[0], bufs, cache).reshape(tuple(u.arg))
    elif op == Ops.EXPAND:
        arr = np.broadcast_to(eval_tensor(u.src[0], bufs, cache), tuple(u.arg))
    elif op == Ops.PERMUTE:
        arr = np.transpose(eval_tensor(u.src[0], bufs, cache), u.arg)
    elif op == Ops.FLIP:
        arr = eval_tensor(u.src[0], bufs, cache)
        for a in u.arg:
            arr = np.flip(arr, a)
    elif op == Ops.PAD:
        arr = np.pad(eval_tensor(u.src[0], bufs, cache), u.arg)
    elif op == Ops.SHRINK:
        src = eval_tensor(u.src[0], bufs, cache)
        arr = src[tuple(slice(b, e) for b, e in u.arg)]
    elif op == Ops.REDUCE_AXIS:
        rop, axes = u.arg
        src = eval_tensor(u.src[0], bufs, cache)
        fn = np.sum if rop == Ops.ADD else np.max
        arr = fn(src, axis=axes, keepdims=True)
    elif op in ALU:
        with np.errstate(all="ignore"):
            vals = [eval_tensor(s, bufs, cache) for s in u.src]
            arr = _alu_np(op, vals)
    else:
        raise NotImplementedError(f"eval_tensor: {op}")

    arr = np.asarray(arr).astype(_npdtype(u.dtype))
    cache[u] = arr
    return arr

# Step 14 - conv2d
def conv2d(x, w, pad=0):
    N, Cin, H, W = x.shape
    Cout, Cin2, kh, kw = w.shape
    assert Cin == Cin2
    Ho = H + 2 * pad - kh + 1
    Wo = W + 2 * pad - kw + 1

    # pad the two spatial axes (identity when pad == 0, so no PAD node)
    xp = x.pad(((0, 0), (0, 0), (pad, pad), (pad, pad)))

    total = None
    for dh in range(kh):
        for dw in range(kw):
            # shifted window of the padded input -> (N, 1, Cin, Ho, Wo) -> (N, Cout, Cin, Ho, Wo)
            win = xp.shrink(((0, N), (0, Cin), (dh, dh + Ho), (dw, dw + Wo)))
            win = win.reshape(N, 1, Cin, Ho, Wo)
            win = win.expand(N, Cout, Cin, Ho, Wo)

            # the matching weight tap -> (1, Cout, Cin, 1, 1) -> (N, Cout, Cin, Ho, Wo)
            wt = w.shrink(((0, Cout), (0, Cin), (dh, dh + 1), (dw, dw + 1)))
            wt = wt.reshape(1, Cout, Cin, 1, 1)
            wt = wt.expand(N, Cout, Cin, Ho, Wo)

            prod = win * wt
            total = prod if total is None else total + prod

    # single reduce over the input channels: (N, Cout, Cin, Ho, Wo) -> (N, Cout, Ho, Wo)
    return total.sum(2)


Tensor.conv2d = conv2d

# Step 15 - Kernel
def load(p, idx):
    if not isinstance(idx, UOp):
        idx = UOp.const(dtypes.int32, idx)
    return UOp(Ops.LOAD, p.dtype, (UOp(Ops.INDEX, p.dtype, (p, idx)),))


class Reduce:
    def __init__(self, ranges, accs, body):
        self.ranges, self.accs, self.body = tuple(ranges), [list(a) for a in accs], list(body)

    def __repr__(self):
        return f"Reduce(ranges={[r.arg for r in self.ranges]}, accs={len(self.accs)}, body={self.body})"


class Kernel:
    def __init__(self, name, params, out_ranges, body, stores):
        self.name, self.params, self.out_ranges, self.body, self.stores = name, list(params), list(out_ranges), list(body), list(stores)

    def map_exprs(self, f):
        def map_reduce(red):
            new_accs = [[acc, f(init), f(update)] for acc, init, update in red.accs]
            new_body = [map_reduce(b) for b in red.body]
            return Reduce(red.ranges, new_accs, new_body)

        new_body = [map_reduce(b) for b in self.body]
        new_stores = [(f(idx), f(val)) for idx, val in self.stores]
        return Kernel(self.name, self.params, self.out_ranges, new_body, new_stores)

    def all_exprs(self):
        exprs = []

        def collect(red):
            for acc, init, update in red.accs:
                exprs.append(init)
                exprs.append(update)
            for b in red.body:
                collect(b)

        for b in self.body:
            collect(b)
        for idx, val in self.stores:
            exprs.append(idx)
            exprs.append(val)
        return exprs


def ranges_in(u):
    seen, result = set(), set()
    stack = [u]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        if x.op == Ops.RANGE:
            result.add(x)
        stack.extend(x.src)
    return result


def flat_index(idxs, shape):
    strides = strides_of(shape)
    result = UOp.const(dtypes.int32, 0)
    for idx, s, stride in zip(idxs, shape, strides):
        if s == 1:
            continue
        result = result + idx * UOp.const(dtypes.int32, stride)
    return result

# Step 16 - Lowerer
class Lowerer:
    def __init__(self, realized):
        self.realized, self.n = realized, 0
        self.memo, self.frames = {}, []

    def new_range(self, n):
        r = UOp.range(n, self.n)
        self.n += 1
        return r

    def new_acc(self, dtype):
        a = UOp(Ops.DEFINE_ACC, dtype, (), self.n)
        self.n += 1
        return a

    def lower(self, root):
        shape = shape_of(root)
        out_ranges = []
        for s in shape:
            out_ranges.append(UOp.const(dtypes.int32, 0) if s == 1 else self.new_range(s))
        idxs = tuple(out_ranges)
        out_set = {r for r in out_ranges if r.op == Ops.RANGE}
        self.frames.append([out_set, []])
        val = self.index(root, idxs, root)
        body = self.frames.pop()[1]
        return Kernel("k", [], out_ranges, body, [(flat_index(idxs, shape), val)])

    def index(self, u, idxs, root):
        key = (u, idxs)
        if key in self.memo:
            return self.memo[key]
        op = u.op
        if op == Ops.BUFFER or (u in self.realized and u is not root):
            res = load(self.realized[u], flat_index(idxs, shape_of(u)))
        elif op == Ops.CONST:
            res = u
        elif op in ALU:
            res = UOp(op, u.dtype, tuple(self.index(s, idxs, root) for s in u.src), u.arg)
        elif op == Ops.RESHAPE:
            src = u.src[0]
            f = flat_index(idxs, shape_of(u))
            if src.op == Ops.BUFFER or (src in self.realized and src is not root):
                # contiguous source: read the same flat offset
                res = load(self.realized[src], f)
            else:
                sshape = shape_of(src)
                sstrides = strides_of(sshape)
                src_idxs = tuple(
                    UOp.const(dtypes.int32, 0) if sj == 1
                    else (f // UOp.const(dtypes.int32, st)) % UOp.const(dtypes.int32, sj)
                    for sj, st in zip(sshape, sstrides)
                )
                res = self.index(src, src_idxs, root)
        elif op == Ops.EXPAND:
            src = u.src[0]
            sshape = shape_of(src)
            src_idxs = tuple(
                UOp.const(dtypes.int32, 0) if sj == 1 else idxs[i]
                for i, sj in enumerate(sshape)
            )
            res = self.index(src, src_idxs, root)
        elif op == Ops.PERMUTE:
            src = u.src[0]
            src_idxs = [None] * len(u.arg)
            for k, p in enumerate(u.arg):
                src_idxs[p] = idxs[k]
            res = self.index(src, tuple(src_idxs), root)
        elif op == Ops.FLIP:
            src = u.src[0]
            shape = shape_of(u)
            axes = set(u.arg)
            src_idxs = tuple(
                (UOp.const(dtypes.int32, shape[k] - 1) - idxs[k]) if k in axes else idxs[k]
                for k in range(len(shape))
            )
            res = self.index(src, src_idxs, root)
        elif op == Ops.SHRINK:
            src = u.src[0]
            src_idxs = tuple(
                idxs[k] if b == 0 else idxs[k] + UOp.const(dtypes.int32, b)
                for k, (b, e) in enumerate(u.arg)
            )
            res = self.index(src, src_idxs, root)
        elif op == Ops.PAD:
            src = u.src[0]
            sshape = shape_of(src)
            src_idxs = []
            valid = UOp.const(dtypes.bool, True)
            for k, (lo, hi) in enumerate(u.arg):
                src_idxs.append(idxs[k] if lo == 0 else idxs[k] - UOp.const(dtypes.int32, lo))
                if lo > 0:
                    valid = UOp(Ops.AND, dtypes.bool,
                                (valid, UOp.const(dtypes.int32, lo - 1) < idxs[k]))
                if hi > 0:
                    valid = UOp(Ops.AND, dtypes.bool,
                                (valid, idxs[k] < UOp.const(dtypes.int32, lo + sshape[k])))
            inner = self.index(src, tuple(src_idxs), root)
            res = UOp(Ops.WHERE, u.dtype, (valid, inner, UOp.const(u.dtype, 0)))
        elif op == Ops.REDUCE_AXIS:
            raise NotImplementedError("REDUCE_AXIS")
        else:
            raise NotImplementedError(f"index: {op}")
        self.memo[key] = res
        return res


def lower_kernel(root, inputs, name="k"):
    realized = {}
    params = [param("out", root.dtype, 0)]
    for i, (t, nm) in enumerate(inputs.items()):
        p = param(nm, t.dtype, i + 1)
        realized[t] = p
        params.append(p)
    low = Lowerer(realized)
    k = low.lower(root)
    k.name = name
    k.params = params
    return k.map_exprs(simplify)

# Step 17 - ReduceLowerer
def reduce_exprs(r):
    exprs = []
    for acc, init, update in r.accs:
        exprs.append(init)
        exprs.append(update)
    for b in r.body:
        exprs.extend(reduce_exprs(b))
    return exprs


class ReduceLowerer(Lowerer):
    def index(self, u, idxs, root):
        # A realized reduce that isn't the root becomes a plain buffer load
        # (handled by the parent); everything else defers too.
        if u.op is not Ops.REDUCE_AXIS or (u is not root and u in self.realized):
            return super().index(u, idxs, root)

        key = (u, idxs)
        if key in self.memo:
            return self.memo[key]

        op, axes = u.arg
        src = u.src[0]
        sshape = shape_of(src)

        # One fresh range per reduced axis, sized by the source.
        new_ranges = tuple(self.new_range(sshape[k]) for k in axes)
        axis_map = dict(zip(axes, new_ranges))
        src_idxs = tuple(axis_map.get(k, idxs[k]) for k in range(len(sshape)))

        # Push [enclosing ranges | new ranges, []] and lower the source inside it.
        enclosing = self.frames[-1][0] if self.frames else set()
        self.frames.append([enclosing | set(new_ranges), []])
        val = self.index(src, src_idxs, root)
        body = self.frames.pop()[1]

        # Accumulator + coupled init/update.
        dtype = u.dtype
        acc = self.new_acc(dtype)
        if op == Ops.ADD:
            init = UOp.const(dtype, 0.0)
            update = acc + val
        elif op == Ops.MAX:
            init = UOp.const(dtype, -INF)
            update = acc.maximum(val)
        else:
            raise NotImplementedError(f"reduce op {op}")

        stmt = Reduce(new_ranges, [[acc, init, update]], body)

        # Dependencies: ranges in init, update and all body expressions
        # (recursively through nested reduces), minus this reduce's own ranges.
        deps = set()
        for e in reduce_exprs(stmt):
            deps |= ranges_in(e)
        deps -= set(new_ranges)

        # Hoist to the outermost frame whose set covers all deps; else innermost.
        target = None
        for frame in self.frames:
            if deps <= frame[0]:
                target = frame
                break
        if target is None:
            target = self.frames[-1]
        target[1].append(stmt)

        self.memo[key] = acc
        return acc


def lower_kernel(root, inputs, name="k"):
    realized = {}
    params = [param("out", root.dtype, 0)]
    for i, (t, nm) in enumerate(inputs.items()):
        p = param(nm, t.dtype, i + 1)
        realized[t] = p
        params.append(p)

    low = ReduceLowerer(realized)
    k = low.lower(root)
    k.name = name
    k.params = params
    return k.map_exprs(simplify)

# Step 18 - run_kernel_np
import numpy as np


def _np_dt(dt):
    return getattr(dt, "np", dt)


def run_kernel_np(k, bufs):
    out = bufs[0]
    contiguous = out.flags["C_CONTIGUOUS"]
    out_flat = out.reshape(-1)

    # Output grid sizes: a CONST in out_ranges is a size-1 axis (index 0).
    out_sizes = [1 if r.op == Ops.CONST else int(r.src[0].arg) for r in k.out_ranges]
    grid_shape = tuple(out_sizes)

    env = {}
    if grid_shape:
        idx_arrs = np.indices(grid_shape, dtype=np.int64)
        for i, r in enumerate(k.out_ranges):
            if r.op != Ops.CONST:          # RANGE / SPECIAL bound to coordinates
                env[r] = idx_arrs[i]
            # CONST stays unevaluated-here; ev() returns its scalar 0.

    cache = {}

    def coerce(dt, r):
        kind = np.dtype(_np_dt(dt)).kind
        if kind == "f":
            return np.asarray(r, dtype=np.float32)
        if kind == "i":
            return np.asarray(r, dtype=np.int64)
        if kind == "b":
            return np.asarray(r, dtype=bool)
        return np.asarray(r)

    def ev(u):
        if u in cache:
            return cache[u]
        if u in env:
            r = env[u]
        else:
            op = u.op
            if op == Ops.CONST:
                r = coerce(u.dtype, u.arg)
            elif op == Ops.LOAD:
                index = u.src[0]
                p = index.src[0]
                idx = np.asarray(ev(index.src[1]), dtype=np.int64)
                flat = np.asarray(bufs[p.arg[1]]).reshape(-1)
                if flat.size:
                    r = flat[np.clip(idx, 0, flat.size - 1)]
                else:
                    r = np.zeros_like(idx)
                r = coerce(u.dtype, r)
            elif op == Ops.WHERE:
                c = ev(u.src[0]); a = ev(u.src[1]); b = ev(u.src[2])
                r = coerce(u.dtype, np.where(np.asarray(c, dtype=bool), a, b))
            elif op == Ops.CAST:
                r = coerce(u.dtype, ev(u.src[0]))
            elif op in ALU:
                s = [ev(x) for x in u.src]
                with np.errstate(all="ignore"):
                    if op == Ops.ADD:      r = s[0] + s[1]
                    elif op == Ops.MUL:    r = s[0] * s[1]
                    elif op == Ops.MAX:    r = np.maximum(s[0], s[1])
                    elif op == Ops.CMPLT:  r = np.asarray(s[0]) < np.asarray(s[1])
                    elif op == Ops.AND:    r = np.logical_and(s[0], s[1])
                    elif op == Ops.IDIV:
                        a = np.asarray(s[0], np.int64); b = np.asarray(s[1], np.int64)
                        r = np.where(b != 0, np.floor_divide(a, np.where(b != 0, b, 1)), 0)
                    elif op == Ops.MOD:
                        a = np.asarray(s[0], np.int64); b = np.asarray(s[1], np.int64)
                        r = np.where(b != 0, np.mod(a, np.where(b != 0, b, 1)), 0)
                    elif op == Ops.RECIP: r = 1.0 / np.asarray(s[0], np.float64)
                    elif op == Ops.EXP2:  r = np.exp2(np.asarray(s[0], np.float64))
                    elif op == Ops.LOG2:  r = np.log2(np.asarray(s[0], np.float64))
                    elif op == Ops.SQRT:  r = np.sqrt(np.asarray(s[0], np.float64))
                    else: raise NotImplementedError(op)
                r = coerce(u.dtype, r)
            else:
                raise NotImplementedError(f"ev: {op}")
        cache[u] = r
        return r

    def run_reduce(red):
        for acc, init, _ in red.accs:
            env[acc] = np.array(np.broadcast_to(ev(init), grid_shape))
        sizes = tuple(int(r.src[0].arg) for r in red.ranges)
        for coord in np.ndindex(*sizes):
            for r, c in zip(red.ranges, coord):
                env[r] = np.int64(c)
            cache.clear()
            for b in red.body:
                run_reduce(b)
            updates = [(acc, ev(update)) for acc, _, update in red.accs]
            for acc, val in updates:
                env[acc][...] = val
        for r in red.ranges:
            del env[r]
        cache.clear()

    for stmt in k.body:
        run_reduce(stmt)

    for idx_node, val_node in k.stores:
        fi = np.asarray(ev(idx_node), dtype=np.int64)
        out_flat[fi] = ev(val_node)

    if not contiguous:
        out[...] = out_flat.reshape(out.shape)
    return out

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

