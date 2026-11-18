# ----------------------------- #
# Bit level simplifications     #
# ----------------------------- #
"""Simplifications targeting bit-blasted, bit-serial and bit-sliced
computations, where word operations are computed bit by bit (ripple carry
adders, serial comparators, shift-and-add multipliers, masks made of
replicated bits, ...).

They are not enabled by default: use `expr_simp_bits`
(miasm.expression.simplifications), or enable
`ExpressionSimplifier.PASS_BITS` on your own simplifier.

Two kinds of passes are provided:
- syntactic ones: 1 bit arithmetic, masks to conditions, lifting of the same
  bitwise operation applied on each element of an ExprCompose
  (`{a[0:1] ^ b[0:1], a[1:2] ^ b[1:2], ...}` -> `a ^ b`), shift-and-add
  multiplication;
- semantic ones: the expression is bit-blasted to a BDD, and compared with
  the BDD of word level candidates (additions / subtractions with carry and
  constant, comparisons). They do not depend on the way the adder or the
  comparator is written, and are exact (no sampling).
"""

import random

from miasm.expression.expression import ExprInt, ExprId, ExprMem, ExprLoc, \
    ExprOp, ExprSlice, ExprCompose, ExprCond, TOK_EQUAL, TOK_INF_SIGNED, \
    TOK_INF_UNSIGNED, TOK_INF_EQUAL_SIGNED, TOK_INF_EQUAL_UNSIGNED

BITWISE_OPS = ("^", "&", "|")
COMPARE_OPS = (TOK_EQUAL, TOK_INF_SIGNED, TOK_INF_UNSIGNED,
               TOK_INF_EQUAL_SIGNED, TOK_INF_EQUAL_UNSIGNED)
# !(a op b) <=> b _NEGATED_COMPARE[op] a
_NEGATED_COMPARE = {
    TOK_INF_UNSIGNED: TOK_INF_EQUAL_UNSIGNED,
    TOK_INF_EQUAL_UNSIGNED: TOK_INF_UNSIGNED,
    TOK_INF_SIGNED: TOK_INF_EQUAL_SIGNED,
    TOK_INF_EQUAL_SIGNED: TOK_INF_SIGNED,
}

# Limits of the semantic passes
MAX_VECTOR_SIZE = 256
MAX_GRAPH_NODES = 20000
MAX_BDD_NODES = 200000
MAX_CACHE = 50000


# ----------------------------- #
# Syntactic passes              #
# ----------------------------- #

def simp_1bit_arith(_, expr):
    """Arithmetic on 1 bit operands is a bitwise operation
    a + b, a - b => a ^ b
    a * b => a & b
    -a => a
    a << b, a >> b => a & ~b
    a a>> b, a <<< b, a >>> b => a
    """
    if expr.size != 1:
        return expr
    op, args = expr.op, expr.args
    if any(arg.size != 1 for arg in args):
        return expr
    if op == "-" and len(args) == 1:
        return args[0]
    if op in ("+", "-") and len(args) >= 2:
        return ExprOp("^", *args)
    if op == "*" and len(args) >= 2:
        return ExprOp("&", *args)
    if op in ("<<", ">>") and len(args) == 2:
        return args[0] & (args[1] ^ ExprInt(1, 1))
    if op in ("a>>", "<<<", ">>>") and len(args) == 2:
        return args[0]
    return expr


def simp_1bit_cond(_, expr):
    """c ? 1 : 0 => c, c ? 0 : 1 => c ^ 1 (c on 1 bit)"""
    if expr.size != 1 or expr.cond.size != 1:
        return expr
    src1, src2 = expr.src1, expr.src2
    if src1.is_int() and src2.is_int():
        if int(src1) == 1 and int(src2) == 0:
            return expr.cond
        if int(src1) == 0 and int(src2) == 1:
            return expr.cond ^ ExprInt(1, 1)
    return expr


def simp_slice_signext_bit(_, expr):
    """signExt(c)[x:y] => signExt(c) on y - x bits (c on 1 bit)"""
    arg = expr.arg
    if (arg.is_op() and arg.op.startswith("signExt_") and
            arg.args[0].size == 1):
        return arg.args[0].signExtend(expr.size)
    return expr


def _mask_control(expr):
    """If @expr is a mask made of a replicated bit (all ones / all zeros),
    return (condition, inverted), else None"""
    if expr.size == 1:
        return None
    if (expr.is_op() and expr.op.startswith("signExt_") and
            expr.args[0].size == 1):
        return expr.args[0], False
    if expr.is_cond() and expr.src1.is_int() and expr.src2.is_int():
        mask = expr.mask
        if int(expr.src1) == int(mask) and int(expr.src2) == 0:
            return expr.cond, False
        if int(expr.src1) == 0 and int(expr.src2) == int(mask):
            return expr.cond, True
    return None


def simp_mask_to_cond(_, expr):
    """Masks made of a replicated bit are conditions
    X & signExt(c) => c ? X : 0
    signExt(c) ^ -1 => c ? 0 : -1
    """
    op, args = expr.op, expr.args
    if op == "^" and len(args) == 2:
        if args[1].is_int() and int(args[1]) == int(expr.mask):
            control = _mask_control(args[0])
            if (control is not None and args[0].is_op()):
                cond, inverted = control
                ones, zero = ExprInt(int(expr.mask), expr.size), ExprInt(0, expr.size)
                if inverted:
                    return ExprCond(cond, ones, zero)
                return ExprCond(cond, zero, ones)
        return expr
    if op != "&":
        return expr
    for index, arg in enumerate(args):
        control = _mask_control(arg)
        if control is None:
            continue
        others = args[:index] + args[index + 1:]
        if len(others) == 1:
            other = others[0]
        else:
            other = ExprOp("&", *others)
        cond, inverted = control
        zero = ExprInt(0, expr.size)
        if inverted:
            return ExprCond(cond, zero, other)
        return ExprCond(cond, other, zero)
    return expr


def _shifted(expr):
    """Decompose @expr as X << shift. Return a list of (X, shift)"""
    out = [(expr, 0)]
    if expr.is_op("<<") and len(expr.args) == 2 and expr.args[1].is_int():
        out.append((expr.args[0], int(expr.args[1])))
    elif expr.is_op("*") and len(expr.args) == 2 and expr.args[1].is_int():
        value = int(expr.args[1])
        if value and value & (value - 1) == 0:
            out.append((expr.args[0], value.bit_length() - 1))
    elif expr.is_compose() and len(expr.args) == 2:
        low, high = expr.args
        if (low.is_int() and int(low) == 0 and high.is_slice() and
                high.start == 0 and high.arg.size == expr.size):
            out.append((high.arg, low.size))
    return out


def _control_bit(cond):
    """Return (base, index) if @cond is a single bit of a word"""
    if cond.size != 1:
        return None
    if cond.is_slice():
        return cond.arg, cond.start
    if cond.is_id() or cond.is_mem():
        return cond, 0
    return None


def simp_shift_add_mul(_, expr):
    """Shift-and-add multiplication
    (Y[0] ? X : 0) + (Y[1] ? X << 1 : 0) + ... => X * zeroExt(Y[0:n])
    """
    if expr.op != "+" or len(expr.args) < 2:
        return expr
    size = expr.size
    # (X, Y base, shift - bit index) -> {bit index: argument}
    groups = {}
    for arg in expr.args:
        if not (arg.is_cond() and arg.src2.is_int() and int(arg.src2) == 0):
            continue
        control = _control_bit(arg.cond)
        if control is None:
            continue
        base, index = control
        for multiplicand, shift in _shifted(arg.src1):
            key = (multiplicand, base, shift - index)
            groups.setdefault(key, {}).setdefault(index, arg)
    best = None
    for key, terms in groups.items():
        # Keep the longest run of consecutive bits
        indexes = sorted(terms)
        start = 0
        for pos in range(1, len(indexes) + 1):
            if pos == len(indexes) or indexes[pos] != indexes[pos - 1] + 1:
                run = indexes[start:pos]
                if len(run) >= 2 and (best is None or len(run) > len(best[1])):
                    best = (key, run)
                start = pos
    if best is None:
        return expr
    (multiplicand, base, delta), run = best
    used = set(groups[(multiplicand, base, delta)][index] for index in run)
    if len(used) != len(run):
        return expr
    low, high = run[0], run[-1] + 1
    if high - low > size:
        return expr
    if base.size == 1:
        multiplier = base
    else:
        multiplier = base[low:high]
    product = multiplicand * multiplier.zeroExtend(size)
    shift = low + delta
    if shift < 0:
        return expr
    if shift:
        product = product << ExprInt(shift, size)
    others = [arg for arg in expr.args if arg not in used]
    if not others:
        return product
    return ExprOp("+", product, *others)


# Lifting of element wise operations

class _Shaper(object):
    """Compute the structure of expressions, with slices made relative to a
    bit offset. Structures are interned as integers, to compare them in
    constant time whatever the size of the expression"""

    def __init__(self):
        self.memo = {}
        self.table = {}

    def _intern(self, key):
        index = self.table.get(key)
        if index is None:
            index = len(self.table)
            self.table[key] = index
        return index

    def shape(self, expr, offset):
        key = (expr, offset)
        res = self.memo.get(key)
        if res is not None:
            return res
        if expr.is_slice():
            res = ("S", expr.arg, expr.start - offset, expr.size)
        elif expr.is_int():
            res = ("I", expr.size)
        elif expr.is_op():
            res = ("O", expr.op, expr.size,
                   tuple(self.shape(arg, offset) for arg in expr.args))
        elif expr.is_cond():
            res = ("C", self.shape(expr.cond, offset),
                   self.shape(expr.src1, offset),
                   self.shape(expr.src2, offset))
        elif expr.is_compose():
            res = ("M", tuple(self.shape(arg, offset) for arg in expr.args))
        else:
            res = ("X", expr)
        res = self._intern(res)
        self.memo[key] = res
        return res


def _lift(parts, shaper):
    """Lift the elements @parts (same size, lowest first) of a vector to a
    single word level expression, or return None"""
    count = len(parts)
    width = parts[0].size
    first = parts[0]
    if all(part.is_int() for part in parts):
        value = 0
        for index, part in enumerate(parts):
            value |= int(part) << (index * width)
        return ExprInt(value, count * width)
    if all(part == first for part in parts):
        # Replicated element
        if width == 1:
            return first.signExtend(count)
        return None
    if all(part.is_slice() for part in parts):
        base, start = first.arg, first.start
        if all(part.arg == base and part.start == start + index * width
               for index, part in enumerate(parts)):
            if start == 0 and base.size == count * width:
                return base
            return base[start:start + count * width]
        return None
    if all(part.is_cond() for part in parts):
        if all(part.cond == first.cond for part in parts):
            # Same condition for every element
            src1 = _lift([part.src1 for part in parts], shaper)
            src2 = _lift([part.src2 for part in parts], shaper)
            if src1 is None or src2 is None:
                return None
            return ExprCond(first.cond, src1, src2)
        if width == 1 and all(part.cond.size == 1 for part in parts):
            # Element wise multiplexer
            cond = _lift([part.cond for part in parts], shaper)
            src1 = _lift([part.src1 for part in parts], shaper)
            src2 = _lift([part.src2 for part in parts], shaper)
            if cond is None or src1 is None or src2 is None:
                return None
            mask = ExprInt(-1, cond.size)
            return (cond & src1) | ((cond ^ mask) & src2)
        return None

    # Bitwise operation: some elements may lack the constant operand, or
    # the operation itself
    ops = set(part.op for part in parts if part.is_op())
    ops.discard(None)
    if len(ops) != 1:
        return None
    op = ops.pop()
    if op not in BITWISE_OPS:
        return None
    identity = 0 if op in ("^", "|") else (1 << width) - 1
    all_args = []
    for part in parts:
        if part.is_op(op):
            all_args.append(list(part.args))
        else:
            all_args.append([part])
    # Operands shared by every element
    shared = [arg for arg in all_args[0] if not arg.is_int()]
    for args in all_args[1:]:
        if not shared:
            break
        args = set(args)
        shared = [arg for arg in shared if arg in args]
    shared_set = set(shared)
    columns = None
    constants = []
    for index, args in enumerate(all_args):
        constant = identity
        others = []
        for arg in args:
            if arg.is_int():
                constant = int(arg)
            elif arg not in shared_set:
                others.append(arg)
        constants.append(constant)
        offset = index * width
        shapes = [shaper.shape(arg, offset) for arg in others]
        if columns is None:
            # The first element gives the order of the operands
            columns = [[shape, []] for shape in shapes]
        if len(shapes) != len(columns):
            return None
        by_shape = {}
        for shape, arg in zip(shapes, others):
            by_shape.setdefault(shape, []).append(arg)
        for column in columns:
            candidates = by_shape.get(column[0])
            if not candidates:
                return None
            column[1].append(candidates.pop(0))
    lifted = []
    for _, column in columns:
        res = _lift(column, shaper)
        if res is None:
            return None
        lifted.append(res)
    for arg in shared:
        if width == 1:
            lifted.append(arg.signExtend(count))
        else:
            lifted.append(ExprCompose(*([arg] * count)))
    value = 0
    for index, constant in enumerate(constants):
        value |= constant << (index * width)
    full_identity = 0 if op in ("^", "|") else (1 << (count * width)) - 1
    if value != full_identity:
        lifted.append(ExprInt(value, count * width))
    if not lifted:
        return ExprInt(value, count * width)
    if len(lifted) == 1:
        return lifted[0]
    return ExprOp(op, *lifted)


def _is_plain(part):
    """True if @part is not a computation"""
    return part.is_int() or part.is_id() or part.is_mem() or (
        part.is_slice() and not (part.arg.is_op() or part.arg.is_cond() or
                                 part.arg.is_compose()))


def _worth_lifting(parts):
    """True if the run @parts is not already handled by the common passes"""
    if not all(_is_plain(part) for part in parts):
        return True
    # Replicated bit
    first = parts[0]
    return (first.size == 1 and not first.is_int() and
            all(part == first for part in parts))


def simp_compose_lift(_, expr):
    """Lift an operation applied on each element of an ExprCompose
    {a[0:1] ^ b[0:1], a[1:2] ^ b[1:2], ...} => a[0:n] ^ b[0:n]
    {c ? a[0:8] : b[0:8], c ? a[8:16] : b[8:16]} => c ? a[0:16] : b[0:16]
    {x, x, x, x} => signExt(x) (x on 1 bit)
    """
    args = expr.args
    count = len(args)
    if count < 2:
        return expr
    out = []
    modified = False
    index = 0
    shaper = _Shaper()
    while index < count:
        part = args[index]
        stop = index + 1
        while stop < count and args[stop].size == part.size:
            stop += 1
        best = None
        # Longest liftable run starting at @index
        if stop - index >= 2:
            end = stop
            while end - index >= 2:
                candidate = args[index:end]
                if _worth_lifting(candidate):
                    res = _lift(list(candidate), shaper)
                    if res is not None:
                        best = (res, end)
                        break
                end -= 1
        if best is None:
            out.append(part)
            index += 1
        else:
            out.append(best[0])
            index = best[1]
            modified = True
    if not modified:
        return expr
    if len(out) == 1:
        return out[0]
    return ExprCompose(*out)


# ----------------------------- #
# Bit-blasting                  #
# ----------------------------- #

class _TooBig(Exception):
    """The expression exceeds the limits of the semantic passes"""


class _BitGraph(object):
    """Bit level And / Xor / Ite graph of expressions.
    Leaves are the bits of the expressions which cannot be bit-blasted
    (identifiers, memories, arithmetic operations, ...)"""

    def __init__(self, limit=MAX_GRAPH_NODES):
        self.limit = limit
        self.nodes = []
        self.index = {}
        self.memo = {}
        # base expression -> set of bit indexes
        self.leaves = {}
        self.zero = self._new(("C", 0))
        self.one = self._new(("C", 1))

    def _new(self, node):
        index = self.index.get(node)
        if index is None:
            if len(self.nodes) >= self.limit:
                raise _TooBig()
            index = len(self.nodes)
            self.nodes.append(node)
            self.index[node] = index
        return index

    def var(self, base, index):
        self.leaves.setdefault(base, set()).add(index)
        return self._new(("V", base, index))

    def not_(self, a):
        if a == self.zero:
            return self.one
        if a == self.one:
            return self.zero
        node = self.nodes[a]
        if node[0] == "N":
            return node[1]
        return self._new(("N", a))

    def xor(self, a, b):
        if a == b:
            return self.zero
        if a == self.zero:
            return b
        if b == self.zero:
            return a
        if a == self.one:
            return self.not_(b)
        if b == self.one:
            return self.not_(a)
        return self._new(("X", min(a, b), max(a, b)))

    def and_(self, a, b):
        if a == b:
            return a
        if a == self.zero or b == self.zero:
            return self.zero
        if a == self.one:
            return b
        if b == self.one:
            return a
        return self._new(("A", min(a, b), max(a, b)))

    def or_(self, a, b):
        return self.not_(self.and_(self.not_(a), self.not_(b)))

    def ite(self, cond, a, b):
        if cond == self.one or a == b:
            return a
        if cond == self.zero:
            return b
        return self._new(("I", cond, a, b))

    def nonzero(self, expr):
        """Node of 'expr != 0'"""
        if expr.size == 1:
            return self.bit(expr, 0)
        if expr.size > MAX_VECTOR_SIZE:
            raise _TooBig()
        out = self.zero
        for index in range(expr.size):
            out = self.or_(out, self.bit(expr, index))
        return out

    def bits(self, expr):
        if expr.size > MAX_VECTOR_SIZE:
            raise _TooBig()
        return [self.bit(expr, index) for index in range(expr.size)]

    def bit(self, expr, index):
        """Node of the bit @index of @expr"""
        key = (expr, index)
        res = self.memo.get(key)
        if res is None:
            res = self._bit(expr, index)
            self.memo[key] = res
        return res

    def _lower(self, bits1, bits2, signed):
        """Node of bits1 < bits2"""
        lower = self.zero
        last = len(bits1) - 1
        for pos, (bit1, bit2) in enumerate(zip(bits1, bits2)):
            # If the bits differ, the comparison is decided by this rank
            decided = bit1 if (signed and pos == last) else bit2
            lower = self.ite(self.xor(bit1, bit2), decided, lower)
        return lower

    def _compare(self, expr):
        op = expr.op
        arg1, arg2 = expr.args
        bits1, bits2 = self.bits(arg1), self.bits(arg2)
        if op == TOK_EQUAL:
            diff = self.zero
            for bit1, bit2 in zip(bits1, bits2):
                diff = self.or_(diff, self.xor(bit1, bit2))
            return self.not_(diff)
        if op == TOK_INF_UNSIGNED:
            return self._lower(bits1, bits2, False)
        if op == TOK_INF_SIGNED:
            return self._lower(bits1, bits2, True)
        if op == TOK_INF_EQUAL_UNSIGNED:
            return self.not_(self._lower(bits2, bits1, False))
        return self.not_(self._lower(bits2, bits1, True))

    def _bit(self, expr, index):
        if expr.is_int():
            return self.one if (int(expr) >> index) & 1 else self.zero
        if expr.is_slice():
            return self.bit(expr.arg, expr.start + index)
        if expr.is_compose():
            start = 0
            for arg in expr.args:
                if index < start + arg.size:
                    return self.bit(arg, index - start)
                start += arg.size
            raise ValueError("Bad index")
        if expr.is_cond():
            return self.ite(self.nonzero(expr.cond),
                            self.bit(expr.src1, index),
                            self.bit(expr.src2, index))
        if not expr.is_op():
            return self.var(expr, index)
        op, args = expr.op, expr.args
        if op in BITWISE_OPS:
            func = {"^": self.xor, "&": self.and_, "|": self.or_}[op]
            out = self.bit(args[0], index)
            for arg in args[1:]:
                out = func(out, self.bit(arg, index))
            return out
        if op.startswith("zeroExt_"):
            if index < args[0].size:
                return self.bit(args[0], index)
            return self.zero
        if op.startswith("signExt_"):
            return self.bit(args[0], min(index, args[0].size - 1))
        if len(args) == 2 and args[1].is_int():
            shift = int(args[1])
            size = expr.size
            if op == "<<":
                if shift > index:
                    return self.zero
                return self.bit(args[0], index - shift)
            if op == ">>":
                if index + shift >= size:
                    return self.zero
                return self.bit(args[0], index + shift)
            if op == "a>>":
                return self.bit(args[0], min(index + shift, size - 1))
            if op == "<<<":
                return self.bit(args[0], (index - shift) % size)
            if op == ">>>":
                return self.bit(args[0], (index + shift) % size)
        if (op in COMPARE_OPS and len(args) == 2 and
                args[0].size <= MAX_VECTOR_SIZE):
            return self._compare(expr)
        return self.var(expr, index)


SIMULATION_MASK = (1 << 64) - 1
# Fixed seed: the simplifications must be deterministic
_simulation_rand = random.Random(0x5eed)


def _simulate(graph, zero=False):
    """Evaluate the nodes of @graph on 64 random assignments of its leaves at
    once (bit parallel), or on the all zeros assignment if @zero.
    Return the list of the values of the nodes.
    This is only a filter: equivalences are then proven with BDDs"""
    mask = SIMULATION_MASK
    out = []
    for node in graph.nodes:
        kind = node[0]
        if kind == "C":
            res = mask if node[1] else 0
        elif kind == "V":
            res = 0 if zero else _simulation_rand.getrandbits(64)
        elif kind == "N":
            res = out[node[1]] ^ mask
        elif kind == "X":
            res = out[node[1]] ^ out[node[2]]
        elif kind == "A":
            res = out[node[1]] & out[node[2]]
        else:
            cond = out[node[1]]
            res = (cond & out[node[2]]) | ((cond ^ mask) & out[node[3]])
        out.append(res)
    return out


def _samples(signatures, count=64):
    """Convert the bit signatures of a vector (lowest bit first) to the list
    of the @count simulated integer values"""
    out = []
    for sample in range(count):
        value = 0
        for index, signature in enumerate(signatures):
            value |= ((signature >> sample) & 1) << index
        out.append(value)
    return out


class _BDD(object):
    """Minimal reduced ordered binary decision diagram
    Nodes are integers, 0 and 1 being the terminals; variables are integers,
    ordered by their value"""

    TERMINAL = 1 << 62

    def __init__(self, limit=MAX_BDD_NODES):
        self.limit = limit
        self.nodes = [(self.TERMINAL, 0, 0), (self.TERMINAL, 1, 1)]
        self.unique = {}
        self.cache = {}

    def mk(self, var, low, high):
        if low == high:
            return low
        key = (var, low, high)
        node = self.unique.get(key)
        if node is None:
            if len(self.nodes) >= self.limit:
                raise _TooBig()
            node = len(self.nodes)
            self.nodes.append(key)
            self.unique[key] = node
        return node

    def var(self, var):
        return self.mk(var, 0, 1)

    def ite(self, f, g, h):
        if f == 1:
            return g
        if f == 0:
            return h
        if g == h:
            return g
        if g == 1 and h == 0:
            return f
        key = (f, g, h)
        res = self.cache.get(key)
        if res is not None:
            return res
        nodes = self.nodes
        vf, vg, vh = nodes[f][0], nodes[g][0], nodes[h][0]
        top = min(vf, vg, vh)
        f0, f1 = (nodes[f][1], nodes[f][2]) if vf == top else (f, f)
        g0, g1 = (nodes[g][1], nodes[g][2]) if vg == top else (g, g)
        h0, h1 = (nodes[h][1], nodes[h][2]) if vh == top else (h, h)
        res = self.mk(top, self.ite(f0, g0, h0), self.ite(f1, g1, h1))
        self.cache[key] = res
        return res

    def not_(self, f):
        return self.ite(f, 0, 1)

    def xor(self, f, g):
        return self.ite(f, self.not_(g), g)

    def and_(self, f, g):
        return self.ite(f, g, 0)

    def or_(self, f, g):
        return self.ite(f, 1, g)

    def add(self, bits1, bits2, carry):
        """Ripple carry addition of two vectors"""
        out = []
        for bit1, bit2 in zip(bits1, bits2):
            partial = self.xor(bit1, bit2)
            out.append(self.xor(partial, carry))
            carry = self.or_(self.and_(bit1, bit2), self.and_(partial, carry))
        return out

    def eval_zero(self, f):
        """Value of @f when every variable is 0"""
        while f > 1:
            f = self.nodes[f][1]
        return f

    def single_path(self, f, target):
        """If only one assignment of the variables of @f leads to the
        terminal @target, return it as a {var: value} dict"""
        out = {}
        other = 1 - target
        while f > 1:
            var, low, high = self.nodes[f]
            if low == other:
                out[var] = 1
                f = high
            elif high == other:
                out[var] = 0
                f = low
            else:
                return None
        return out if f == target else None

    def from_graph(self, graph, order, roots):
        """Translate the nodes of the _BitGraph @graph needed by @roots
        @order: (base, index) -> variable
        Return a list indexed by graph node (None if not translated)"""
        needed = set()
        todo = list(roots)
        while todo:
            index = todo.pop()
            if index in needed:
                continue
            needed.add(index)
            node = graph.nodes[index]
            if node[0] in ("N", "X", "A", "I"):
                todo.extend(node[1:])
        out = []
        for index, node in enumerate(graph.nodes):
            if index not in needed:
                out.append(None)
                continue
            kind = node[0]
            if kind == "C":
                res = node[1]
            elif kind == "V":
                res = self.var(order[(node[1], node[2])])
            elif kind == "N":
                res = self.not_(out[node[1]])
            elif kind == "X":
                res = self.xor(out[node[1]], out[node[2]])
            elif kind == "A":
                res = self.and_(out[node[1]], out[node[2]])
            else:
                res = self.ite(out[node[1]], out[node[2]], out[node[3]])
            out.append(res)
        return out


# ----------------------------- #
# Semantic passes               #
# ----------------------------- #

_cache_arith = {}
_cache_predicate = {}


def _cached(cache, func, expr):
    res = cache.get(expr, False)
    if res is False:
        try:
            res = func(expr)
        except (_TooBig, RecursionError):
            res = None
        if len(cache) > MAX_CACHE:
            cache.clear()
        cache[expr] = res
    return res


def _words(graph):
    """Split the leaves of @graph in words [(base, low, high)] and single
    bits [(base, index)]. Return None if a word is not contiguous enough to
    be handled"""
    words, singles = [], []
    for base, indexes in graph.leaves.items():
        if len(indexes) == 1:
            singles.append((base, min(indexes)))
        else:
            words.append((base, min(indexes), max(indexes) + 1))
    return words, singles


def _order(words, singles):
    """Variable order: the bits of the words are interleaved, most
    significant first, then the single bits.
    With this order, the carry (or the comparison) of the k low bits is a
    sub graph of the one of the k + 1 low bits: the BDDs of all the bits of
    an adder, with all their intermediate carries, share their nodes and
    have a linear size"""
    order = {}
    length = max(high - low for _, low, high in words) if words else 0
    for pos in reversed(range(length)):
        for base, low, high in words:
            if low + pos < high:
                order[(base, low + pos)] = len(order)
    for base, index in singles:
        order[(base, index)] = len(order)
    return order


def _word_expr(base, low, high):
    if low == 0 and high == base.size:
        return base
    return base[low:high]


def _recognize_arith(expr):
    """If @expr is, bit for bit, +/-A +/-B +/-c + K (A and B words, c a
    single bit, K a constant), return this expression, else None"""
    size = expr.size
    graph = _BitGraph()
    bits = graph.bits(expr)
    words, singles = _words(graph)
    if not 1 <= len(words) <= 2 or len(singles) > 1:
        return None
    if any(high - low > size for _, low, high in words):
        return None
    # Leaves, as vectors of graph nodes and as expressions
    terms = []
    for base, low, high in words:
        vector = [graph.var(base, index) for index in range(low, high)]
        vector += [graph.zero] * (size - len(vector))
        terms.append((vector, _word_expr(base, low, high).zeroExtend(size)))
    for base, index in singles:
        vector = [graph.var(base, index)] + [graph.zero] * (size - 1)
        bit = base if base.size == 1 else base[index:index + 1]
        terms.append((vector, bit.zeroExtend(size)))

    # The constant is the value of the expression when every leaf is null
    values = _simulate(graph, zero=True)
    constant = 0
    for index, bit in enumerate(bits):
        constant |= (values[bit] & 1) << index

    # Filter the candidates on random values
    values = _simulate(graph)
    mask = (1 << size) - 1
    target_samples = _samples([values[bit] for bit in bits])
    term_samples = [_samples([values[bit] for bit in vector])
                    for vector, _ in terms]
    candidates = []
    for signs in range(1 << len(terms)):
        for sample, expected in enumerate(target_samples):
            acc = constant
            for pos, samples in enumerate(term_samples):
                if (signs >> pos) & 1:
                    acc -= samples[sample]
                else:
                    acc += samples[sample]
            if acc & mask != expected:
                break
        else:
            candidates.append(signs)
    if not candidates:
        return None

    # Proof
    order = _order(words, singles)
    bdd = _BDD()
    roots = list(bits)
    for vector, _ in terms:
        roots += vector
    nodes = bdd.from_graph(graph, order, roots)
    target = [nodes[bit] for bit in bits]
    constant_bits = [(constant >> index) & 1 for index in range(size)]
    for signs in candidates:
        acc = constant_bits
        for pos, (vector, _) in enumerate(terms):
            vector = [nodes[bit] for bit in vector]
            if (signs >> pos) & 1:
                # acc - vector = acc + ~vector + 1
                acc = bdd.add(acc, [bdd.not_(bit) for bit in vector], 1)
            else:
                acc = bdd.add(acc, vector, 0)
        if acc != target:
            continue
        out = []
        for pos, (_, term) in enumerate(terms):
            out.append(-term if (signs >> pos) & 1 else term)
        if constant:
            out.append(ExprInt(constant, size))
        if len(out) == 1:
            return out[0]
        return ExprOp("+", *out)
    return None


def _has_bit_logic(args):
    """True if @args contains at least 3 computed single bits"""
    count = 0
    for arg in args:
        if arg.size == 1 and (arg.is_op() or arg.is_cond()):
            count += 1
            if count >= 3:
                return True
    return False


def simp_compose_arith(_, expr):
    """Recognize bit-serial additions / subtractions
    {a[0:1] ^ b[0:1], a[1:2] ^ b[1:2] ^ (a[0:1] & b[0:1]), ...} => a + b

    The whole ExprCompose is tried first (which handles zero extended
    results and carry out bits), then each run of computed single bits.
    """
    args = expr.args
    if expr.size > MAX_VECTOR_SIZE or not _has_bit_logic(args):
        return expr
    res = _cached(_cache_arith, _recognize_arith, expr)
    if res is not None:
        return res
    # Runs of computed single bits
    out = []
    modified = False
    index = 0
    count = len(args)
    while index < count:
        stop = index
        while (stop < count and args[stop].size == 1 and
               not args[stop].is_int()):
            stop += 1
        if stop - index >= 3 and stop - index < count and _has_bit_logic(
                args[index:stop]):
            run = ExprCompose(*args[index:stop])
            res = _cached(_cache_arith, _recognize_arith, run)
            if res is not None:
                out.append(res)
                index = stop
                modified = True
                continue
        out.append(args[index])
        index += 1
    if not modified:
        return expr
    return ExprCompose(*out)


def _recognize_predicate(expr):
    """If the 1 bit @expr is a comparison of two words (or of a word and a
    constant), return this comparison, else None"""
    graph = _BitGraph()
    root = graph.bit(expr, 0)
    words, singles = _words(graph)
    if singles or not 1 <= len(words) <= 2:
        return None
    for base, low, high in words:
        if len(graph.leaves[base]) != high - low:
            return None
    if len(words) == 2:
        length = words[0][2] - words[0][1]
        if length != words[1][2] - words[1][1] or length < 2:
            return None
    elif words[0][2] - words[0][1] < 3:
        return None
    order = _order(words, singles)
    one = ExprInt(1, 1)

    if len(words) == 1:
        # Comparison with a constant: a single assignment gives 1 (or 0)
        base, low, high = words[0]
        if high - low > 6:
            # Only one assignment over 2^n gives a different result
            signature = _simulate(graph)[root]
            if signature not in (0, SIMULATION_MASK):
                return None
        bdd = _BDD()
        node = bdd.from_graph(graph, order, [root])[root]
        word = _word_expr(base, low, high)
        for target in (1, 0):
            path = bdd.single_path(node, target)
            if path is None or len(path) != high - low:
                continue
            value = 0
            for index in range(low, high):
                value |= path[order[(base, index)]] << (index - low)
            if value == 0:
                res = ExprCond(word, ExprInt(0, 1), one)
            else:
                res = ExprOp(TOK_EQUAL, word, ExprInt(value, word.size))
            return res if target == 1 else res ^ one
        return None

    word1 = _word_expr(*words[0])
    word2 = _word_expr(*words[1])
    candidates = [ExprOp(TOK_EQUAL, word1, word2)]
    for op in (TOK_INF_UNSIGNED, TOK_INF_EQUAL_UNSIGNED, TOK_INF_SIGNED,
               TOK_INF_EQUAL_SIGNED):
        candidates.append(ExprOp(op, word1, word2))
        candidates.append(ExprOp(op, word2, word1))
    candidate_nodes = [graph.bit(candidate, 0) for candidate in candidates]
    # a != b
    candidates.append(candidates[0] ^ one)
    candidate_nodes.append(graph.not_(candidate_nodes[0]))

    # Filter the candidates on random values
    values = _simulate(graph)
    kept = [(candidate, node)
            for candidate, node in zip(candidates, candidate_nodes)
            if values[node] == values[root]]
    if not kept:
        return None

    # Proof
    bdd = _BDD()
    nodes = bdd.from_graph(graph, order,
                           [root] + [node for _, node in kept])
    for candidate, node in kept:
        if nodes[node] == nodes[root]:
            return candidate
    return None


def _predicate(expr):
    if expr.size != 1:
        return expr
    res = _cached(_cache_predicate, _recognize_predicate, expr)
    if res is None or res == expr:
        return expr
    return res


def simp_bit_predicate_op(_, expr):
    """Recognize bit-serial comparisons
    (a[0:1] ^ b[0:1]) | (a[1:2] ^ b[1:2]) | ... => (a == b) ^ 1
    """
    if expr.size != 1 or expr.op in COMPARE_OPS:
        return expr
    if expr.op == "^" and len(expr.args) == 2 and expr.args[1].is_int():
        # Negation of an already recognized comparison
        arg = expr.args[0]
        if int(expr.args[1]) == 1 and arg.is_op() and len(arg.args) == 2:
            # !(a < b) => b <= a, !(a <= b) => b < a
            negated = _NEGATED_COMPARE.get(arg.op)
            if negated is not None:
                return ExprOp(negated, arg.args[1], arg.args[0])
        if (arg.is_op() and arg.op in COMPARE_OPS) or arg.is_cond():
            return expr
    return _predicate(expr)


def simp_bit_predicate_cond(_, expr):
    """Recognize bit-serial comparisons built with multiplexers"""
    if expr.size != 1:
        return expr
    if expr.src1.is_int() and expr.src2.is_int():
        return expr
    return _predicate(expr)
