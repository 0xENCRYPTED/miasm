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
# Truth table based resynthesis: maximum number of input bits, and of
# monomials of the resulting algebraic normal form
MAX_TRUTH_TABLE_INPUTS = 8
MAX_ANF_TERMS = 4


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
    if op == TOK_EQUAL and len(args) == 2:
        return args[0] ^ args[1] ^ ExprInt(1, 1)
    return expr


def simp_neg_to_not(_, expr):
    """Two's complement: -a - 1 => a ^ -1
    (-a) + 0xFF..FF + X => (a ^ 0xFF..FF) + X
    """
    if expr.op != "+" or len(expr.args) < 2:
        return expr
    args = list(expr.args)
    constant = args[-1]
    if not (constant.is_int() and int(constant) == int(expr.mask)):
        return expr
    for index, arg in enumerate(args[:-1]):
        if arg.is_op("-") and len(arg.args) == 1:
            args[index] = arg.args[0] ^ constant
            args.pop()
            if len(args) == 1:
                return args[0]
            return ExprOp("+", *args)
    return expr


def simp_ashr_sign(_, expr):
    """x a>> n => signExt(msb(x)) if n >= size - 1"""
    if expr.op != "a>>" or len(expr.args) != 2 or not expr.args[1].is_int():
        return expr
    if int(expr.args[1]) < expr.size - 1:
        return expr
    return expr.args[0].msb().signExtend(expr.size)


def simp_slice_cond(_, expr):
    """(c ? a : b)[x:y] => c ? a[x:y] : b[x:y]"""
    arg = expr.arg
    if not arg.is_cond():
        return expr
    return ExprCond(arg.cond, arg.src1[expr.start:expr.stop],
                    arg.src2[expr.start:expr.stop])


def simp_cond_signext(_, expr):
    """msb(X) ? {X, -1} : {X, 0} => signExt(X)"""
    src1, src2 = expr.src1, expr.src2
    if not (src1.is_compose() and src2.is_compose()):
        return expr
    if len(src1.args) != 2 or len(src2.args) != 2:
        return expr
    word, ones = src1.args
    if src2.args[0] != word:
        return expr
    zero = src2.args[1]
    if not (ones.is_int() and zero.is_int() and ones.size == zero.size):
        return expr
    if int(zero) != 0 or int(ones) != int(ones.mask):
        return expr
    cond = expr.cond
    if not (cond.is_slice() and cond.size == 1 and cond.arg == word and
            cond.start == word.size - 1):
        return expr
    return word.signExtend(expr.size)


def simp_compose_signext(_, expr):
    """{X, signExt(msb(X))} => signExt(X)
    {X, msb(X) ? -1 : 0} => signExt(X)"""
    args = expr.args
    if len(args) < 2:
        return expr
    last, prev = args[-1], args[-2]
    sign = None
    control = _mask_control(last)
    if control is not None and not control[1]:
        sign = control[0]
    elif last.size == 1:
        sign = last
    if sign is None or sign.size != 1:
        return expr
    if not (sign.is_slice() and sign.arg == prev and
            sign.start == prev.size - 1):
        return expr
    extended = prev.signExtend(prev.size + last.size)
    if len(args) == 2:
        return extended
    return ExprCompose(*(list(args[:-2]) + [extended]))


def simp_and_not(_, expr):
    """X & (X ^ -1) => 0, X | (X ^ -1) => -1"""
    if expr.op not in ("&", "|"):
        return expr
    args = set(expr.args)
    mask = int(expr.mask)
    for arg in expr.args:
        if (arg.is_op("^") and len(arg.args) == 2 and arg.args[1].is_int()
                and int(arg.args[1]) == mask and arg.args[0] in args):
            return ExprInt(0 if expr.op == "&" else mask, expr.size)
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

    def __init__(self, limit=MAX_GRAPH_NODES, blast_arith=False,
                 max_leaves=None):
        """
        @blast_arith: also bit-blast additions, negations and
        multiplications by a constant, except the ones on plain words
        (identifiers, memories, constants and their slices); comparisons of
        plain words are then kept as leaves (atoms)
        @max_leaves: maximum number of leaf bits
        """
        self.limit = limit
        self.blast_arith = blast_arith
        self.max_leaves = max_leaves
        self.leaf_count = 0
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
        indexes = self.leaves.setdefault(base, set())
        if index not in indexes:
            indexes.add(index)
            self.leaf_count += 1
            if (self.max_leaves is not None and
                    self.leaf_count > self.max_leaves):
                raise _TooBig()
        return self._new(("V", base, index))

    def _add(self, bits1, bits2, carry):
        """Ripple carry addition of two vectors of nodes"""
        out = []
        for bit1, bit2 in zip(bits1, bits2):
            partial = self.xor(bit1, bit2)
            out.append(self.xor(partial, carry))
            carry = self.or_(self.and_(bit1, bit2),
                             self.and_(partial, carry))
        return out

    def _mul_const(self, arg, constant, index):
        """Node of the bit @index of arg * constant (shift and add)"""
        count = index + 1
        bits = [self.bit(arg, pos) for pos in range(count)]
        acc = [self.zero] * count
        for shift in range(count):
            if (constant >> shift) & 1:
                shifted = [self.zero] * shift + bits[:count - shift]
                acc = self._add(acc, shifted, self.zero)
        return acc[index]

    def _arith(self, expr, index):
        """Node of the bit @index of an addition / negation"""
        count = index + 1
        args = expr.args
        if expr.op == "-":
            bits = [self.not_(self.bit(args[0], pos)) for pos in range(count)]
            return self._add(bits, [self.zero] * count, self.one)[index]
        acc = [self.bit(args[0], pos) for pos in range(count)]
        for arg in args[1:]:
            acc = self._add(acc, [self.bit(arg, pos) for pos in range(count)],
                            self.zero)
        return acc[index]

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
            if self.blast_arith and all(_is_plain(arg) for arg in args):
                # Comparison of words: atom
                return self.var(expr, index)
            return self._compare(expr)
        if op == "*" and len(args) == 2 and args[1].is_int():
            constant = int(args[1])
            if constant and constant & (constant - 1) == 0:
                # Power of 2: shift
                shift = constant.bit_length() - 1
                if shift > index:
                    return self.zero
                return self.bit(args[0], index - shift)
            if self.blast_arith and not _is_plain(args[0]):
                return self._mul_const(args[0], constant, index)
        if (self.blast_arith and
                (op == "+" or (op == "-" and len(args) == 1)) and
                not all(_is_plain(arg) for arg in args)):
            # An operation on plain words is kept as a word: it is the
            # business of the word level passes
            return self._arith(expr, index)
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

    def eval(self, f, values):
        """Value of @f for the assignment @values ({var: 0 / 1})"""
        while f > 1:
            var, low, high = self.nodes[f]
            f = high if values[var] else low
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
        return _recognize_threshold(graph, root, bdd, node, order, words[0])

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


def _recognize_threshold(graph, root, bdd, node, order, word_info):
    """Comparison of a word with a constant: word < K or K <= word (signed
    or unsigned). The constant is found by dichotomy, the candidate is then
    proven"""
    base, low, high = word_info
    length = high - low
    word = _word_expr(base, low, high)

    def value_at(value):
        values = dict((order[(base, low + pos)], (value >> pos) & 1)
                      for pos in range(length))
        return bdd.eval(node, values)

    last = (1 << length) - 1
    for signed in (False, True):
        # Rank in the order -> value
        flip = (1 << (length - 1)) if signed else 0
        first = value_at(flip)
        if first == value_at(last ^ flip):
            continue
        rank_low, rank_high = 0, last
        while rank_high - rank_low > 1:
            middle = (rank_low + rank_high) // 2
            if value_at(middle ^ flip) == first:
                rank_low = middle
            else:
                rank_high = middle
        constant = ExprInt(rank_high ^ flip, length)
        if first:
            # True below the constant
            op = TOK_INF_SIGNED if signed else TOK_INF_UNSIGNED
            candidate = ExprOp(op, word, constant)
        else:
            op = TOK_INF_EQUAL_SIGNED if signed else TOK_INF_EQUAL_UNSIGNED
            candidate = ExprOp(op, constant, word)
        candidate_node = graph.bit(candidate, 0)
        proof = _BDD()
        nodes = proof.from_graph(graph, order, [root, candidate_node])
        if nodes[root] == nodes[candidate_node]:
            return candidate
    return None


def _expr_cost(expr):
    """Number of distinct operations of @expr"""
    seen = set()
    todo = [expr]
    cost = 0
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur.is_op():
            cost += max(1, len(cur.args) - 1)
            todo.extend(cur.args)
        elif cur.is_cond():
            cost += 1
            todo += [cur.cond, cur.src1, cur.src2]
        elif cur.is_compose():
            cost += 1
            todo.extend(cur.args)
        elif cur.is_slice():
            todo.append(cur.arg)
    return cost


def _truth_table(graph, root, count):
    """Truth table (as an integer, one bit by assignment) of the node @root
    of @graph, the leaves being numbered in their order of creation"""
    rows = 1 << count
    mask = (1 << rows) - 1
    out = {}
    index = 0
    for pos, node in enumerate(graph.nodes[:root + 1]):
        kind = node[0]
        if kind == "C":
            res = mask if node[1] else 0
        elif kind == "V":
            # Leaf number @index: alternating blocks of 2^index bits
            block = (1 << (1 << index)) - 1
            res = 0
            for start in range(1 << index, rows, 2 << index):
                res |= block << start
            index += 1
        elif kind == "N":
            res = out[node[1]] ^ mask
        elif kind == "X":
            res = out[node[1]] ^ out[node[2]]
        elif kind == "A":
            res = out[node[1]] & out[node[2]]
        else:
            cond = out[node[1]]
            res = (cond & out[node[2]]) | ((cond ^ mask) & out[node[3]])
        out[pos] = res
    return out[root]


def _resynthesize(expr):
    """If the 1 bit @expr depends on a few input bits only, rebuild it from
    its truth table: constant, literal, conjunction / disjunction of
    literals, or XOR of a few conjunctions. Additions are bit-blasted;
    comparisons of words are inputs. Return None if nothing simpler is
    found"""
    graph = _BitGraph(limit=MAX_GRAPH_NODES // 4, blast_arith=True,
                      max_leaves=MAX_TRUTH_TABLE_INPUTS)
    root = graph.bit(expr, 0)
    leaves = [(node[1], node[2]) for node in graph.nodes if node[0] == "V"]
    count = len(leaves)
    table = _truth_table(graph, root, count)
    rows = 1 << count

    # Inputs the function really depends on
    support = []
    for index in range(count):
        for row in range(rows):
            if not (row >> index) & 1:
                if ((table >> row) & 1) != ((table >> (row | (1 << index))) & 1):
                    support.append(index)
                    break
    one = ExprInt(1, 1)
    if not support:
        return ExprInt(table & 1, 1)

    def literal(index):
        base, bit = leaves[index]
        return base if base.size == 1 else base[bit:bit + 1]

    # Truth table restricted to the support
    size = len(support)
    values = []
    for row in range(1 << size):
        full = 0
        for pos, index in enumerate(support):
            if (row >> pos) & 1:
                full |= 1 << index
        values.append((table >> full) & 1)

    def cube(row):
        """Literals (negated if needed) of the assignment @row"""
        return [literal(index) if (row >> pos) & 1 else literal(index) ^ one
                for pos, index in enumerate(support)]

    candidate = None
    if size == 1:
        candidate = literal(support[0])
        if values[0]:
            candidate = candidate ^ one
    elif sum(values) == 1:
        # Conjunction
        candidate = ExprOp("&", *cube(values.index(1)))
    elif sum(values) == len(values) - 1:
        # Disjunction: false for a single assignment
        row = values.index(0) ^ ((1 << size) - 1)
        candidate = ExprOp("|", *cube(row))
    else:
        # Algebraic normal form (Moebius transform)
        anf = list(values)
        for pos in range(size):
            for row in range(1 << size):
                if (row >> pos) & 1:
                    anf[row] ^= anf[row ^ (1 << pos)]
        monomials = [row for row in range(1, 1 << size) if anf[row]]
        if len(monomials) > MAX_ANF_TERMS:
            return None
        terms = []
        for row in monomials:
            lits = [literal(index) for pos, index in enumerate(support)
                    if (row >> pos) & 1]
            terms.append(lits[0] if len(lits) == 1 else ExprOp("&", *lits))
        if anf[0]:
            terms.append(one)
        candidate = terms[0] if len(terms) == 1 else ExprOp("^", *terms)
    if _expr_cost(candidate) >= _expr_cost(expr):
        return None
    return candidate


_cache_resynth = {}


def _predicate(expr):
    if expr.size != 1:
        return expr
    res = _cached(_cache_resynth, _resynthesize, expr)
    if res is not None and res != expr:
        return res
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


def simp_bit_predicate_slice(_, expr):
    """Recognize the predicates computed by a single bit of a word
    (saturate(w)[7:8] => 0x7F <s w, ...)"""
    if expr.size != 1:
        return expr
    arg = expr.arg
    if not (arg.is_op() or arg.is_cond() or arg.is_compose()):
        return expr
    return _predicate(expr)


# ----------------------------- #
# Mixed Boolean-Arithmetic      #
# ----------------------------- #
#
# A linear MBA is a sum of bitwise functions of the same words, with integer
# coefficients: 3 * (a & b) - (a | b) + 2 * a + 7.
# Two linear MBA are equal for every value of the words if and only if they
# are equal when each word is either 0 or -1 (the bits do not interact, and
# the sum is linear). A linear MBA is thus characterized by its values on
# the 2^k combinations of 0 / -1 of its k words: its signature. The
# simplification computes the signature, then rebuilds the simplest
# expression having the same one.

MAX_MBA_WORDS = 4


def _mba_atoms(expr, out, stop=frozenset()):
    """Add to the list @out the words (non bitwise sub expressions) of the
    bitwise expression @expr
    @stop: bitwise sub expressions to consider as words"""
    if expr.is_int() and int(expr) in (0, int(expr.mask)):
        return
    if expr.is_op() and expr.op in BITWISE_OPS and expr not in stop:
        for arg in expr.args:
            _mba_atoms(arg, out, stop)
        return
    if expr not in out:
        out.append(expr)


def _mba_table(expr, atoms, stop=frozenset()):
    """Truth table (one bit by combination of the @atoms) of the bitwise
    expression @expr"""
    rows = 1 << len(atoms)
    full = (1 << rows) - 1
    if expr.is_int() and int(expr) in (0, int(expr.mask)):
        return full if int(expr) else 0
    if expr.is_op() and expr.op in BITWISE_OPS and expr not in stop:
        tables = [_mba_table(arg, atoms, stop) for arg in expr.args]
        out = tables[0]
        for table in tables[1:]:
            if expr.op == "^":
                out ^= table
            elif expr.op == "&":
                out &= table
            else:
                out |= table
        return out
    index = atoms.index(expr)
    out = 0
    for row in range(rows):
        if (row >> index) & 1:
            out |= 1 << row
    return out


def _and_all(exprs):
    return exprs[0] if len(exprs) == 1 else ExprOp("&", *exprs)


def _bitwise_from_table(table, atoms):
    """Build a bitwise expression of the words @atoms from its truth
    @table"""
    count = len(atoms)
    size = atoms[0].size
    ones = ExprInt(-1, size)
    rows = 1 << count
    # Words the function depends on
    support = []
    for index in range(count):
        for row in range(rows):
            if not (row >> index) & 1:
                if ((table >> row) & 1) != ((table >> (row | (1 << index))) & 1):
                    support.append(index)
                    break
    if not support:
        return ExprInt(-1 if table & 1 else 0, size)
    values = []
    for row in range(1 << len(support)):
        full = 0
        for pos, index in enumerate(support):
            if (row >> pos) & 1:
                full |= 1 << index
        values.append((table >> full) & 1)
    words = [atoms[index] for index in support]
    last = len(values) - 1
    if sum(values) == 1:
        # Conjunction, with negations
        row = values.index(1)
        return _and_all([word if (row >> pos) & 1 else word ^ ones
                         for pos, word in enumerate(words)])
    if sum(values) == last:
        # Disjunction, with negations
        row = values.index(0)
        lits = [word ^ ones if (row >> pos) & 1 else word
                for pos, word in enumerate(words)]
        return lits[0] if len(lits) == 1 else ExprOp("|", *lits)
    # Algebraic normal form
    anf = list(values)
    for pos in range(len(words)):
        for row in range(len(values)):
            if (row >> pos) & 1:
                anf[row] ^= anf[row ^ (1 << pos)]
    terms = []
    for row in range(1, len(values)):
        if anf[row]:
            terms.append(_and_all([word for pos, word in enumerate(words)
                                   if (row >> pos) & 1]))
    if anf[0]:
        terms.append(ones)
    return terms[0] if len(terms) == 1 else ExprOp("^", *terms)


def _scaled(expr, coeff, size):
    """@coeff * @expr, with small negative coefficients written as such"""
    coeff %= 1 << size
    if coeff == 0:
        return None
    if coeff == 1:
        return expr
    if coeff == (1 << size) - 1:
        return -expr
    if coeff >> (size - 1):
        return -(expr * ExprInt((1 << size) - coeff, size))
    return expr * ExprInt(coeff, size)


def _mba_candidates(signature, atoms, size):
    """Yield (list of terms, constant) having the @signature"""
    mod = 1 << size
    count = len(atoms)
    rows = 1 << count
    constant = signature[0]

    # coeff * (bitwise function): the signature takes two values
    others = set(signature) - set([constant])
    if not others:
        yield [], constant
        return
    if len(others) == 1:
        other = others.pop()
        table = 0
        for row in range(rows):
            if signature[row] == other:
                table |= 1 << row
        term = _scaled(_bitwise_from_table(table, atoms),
                       constant - other, size)
        yield [term], constant

    # Sum of the words
    coeffs = [(constant - signature[1 << index]) % mod
              for index in range(count)]
    for row in range(rows):
        expected = constant
        for index in range(count):
            if (row >> index) & 1:
                expected -= coeffs[index]
        if expected % mod != signature[row]:
            break
    else:
        yield [term for term in (_scaled(atom, coeff, size)
                                 for atom, coeff in zip(atoms, coeffs))
               if term is not None], constant

    # Sum of conjunctions (Moebius transform)
    coeffs = [(constant - value) % mod for value in signature]
    for index in range(count):
        for row in range(rows):
            if (row >> index) & 1:
                coeffs[row] = (coeffs[row] - coeffs[row ^ (1 << index)]) % mod
    terms = []
    for row in range(1, rows):
        term = _scaled(_and_all([atom for index, atom in enumerate(atoms)
                                 if (row >> index) & 1]), coeffs[row], size)
        if term is not None:
            terms.append(term)
    yield terms, constant


def _mba_term(arg):
    """Split a term of a sum in (coefficient, bitwise expression)"""
    coeff = 1
    while True:
        if arg.is_op("-") and len(arg.args) == 1:
            coeff = -coeff
            arg = arg.args[0]
        elif (arg.is_op("*") and len(arg.args) == 2 and
              arg.args[1].is_int()):
            coeff *= int(arg.args[1])
            arg = arg.args[0]
        else:
            return coeff, arg


def simp_linear_mba(_, expr):
    """Linear Mixed Boolean-Arithmetic expressions
    (a | b) - (a & b) => a ^ b
    (a ^ b) + 2 * (a & b) => a + b
    (a | b) + (a & b) => a + b
    a + b - (a | b) => a & b
    (a ^ -1) + 1 => -a
    2 * (a | b) - (a ^ b) - a - b => 0
    """
    if expr.op != "+" or expr.size == 1:
        return expr
    size = expr.size
    mod = 1 << size
    constant = 0
    # (coefficient, bitwise expression, words)
    terms = []
    for arg in expr.args:
        if arg.is_int():
            constant += int(arg)
            continue
        coeff, cur = _mba_term(arg)
        atoms = []
        _mba_atoms(cur, atoms)
        terms.append((coeff, cur, atoms, arg))
    if not any(cur.is_op() and cur.op in BITWISE_OPS
               for _, cur, _, _ in terms):
        return expr

    # Group the terms sharing words
    groups = []
    for term in terms:
        merged = [term]
        words = set(term[2])
        remaining = []
        for group_words, group_terms in groups:
            if words & group_words:
                words |= group_words
                merged += group_terms
            else:
                remaining.append((group_words, group_terms))
        groups = remaining + [(words, merged)]

    def build(parts, value):
        """Sum of the terms @parts and of the constant @value"""
        parts = list(parts)
        if value % mod:
            parts.append(ExprInt(value % mod, size))
        if not parts:
            return ExprInt(0, size)
        if len(parts) == 1:
            return parts[0]
        return ExprOp("+", *parts)

    def shared_logic(subset):
        """Bitwise sub expressions used by several terms of @subset: they
        can be considered as words"""
        counts = {}
        for _, cur, _, _ in subset:
            seen = set()
            todo = [cur]
            while todo:
                node = todo.pop()
                if not (node.is_op() and node.op in BITWISE_OPS):
                    continue
                if node in seen:
                    continue
                seen.add(node)
                todo.extend(node.args)
            for node in seen:
                counts[node] = counts.get(node, 0) + 1
        return frozenset(node for node, count in counts.items() if count > 1)

    def simplify(subset, value):
        """Best rewriting of the terms @subset (the constant @value of the
        sum being taken in account for the cost).
        Return (gain, new terms, new constant) or None"""
        best = None
        stops = [frozenset()]
        shared = shared_logic(subset)
        if shared:
            stops.append(shared)
        for stop in stops:
            res = simplify_with(subset, value, stop)
            if res is not None and (best is None or res[0] > best[0]):
                best = res
        return best

    def simplify_with(subset, value, stop):
        atoms = []
        for _, cur, _, _ in subset:
            _mba_atoms(cur, atoms, stop)
        if (not atoms or len(atoms) > MAX_MBA_WORDS or
                not any(cur.is_op() and cur.op in BITWISE_OPS
                        and cur not in stop
                        for _, cur, _, _ in subset)):
            return None
        tables = [(coeff, _mba_table(cur, atoms, stop))
                  for coeff, cur, _, _ in subset]
        # Value of the sum when the words are 0 / -1: a bitwise function
        # is then 0 or -1
        signature = []
        for row in range(1 << len(atoms)):
            total = 0
            for coeff, table in tables:
                if (table >> row) & 1:
                    total -= coeff
            signature.append(total % mod)
        cost = _expr_cost(build([arg for _, _, _, arg in subset], value))
        best = None
        for new_terms, new_constant in _mba_candidates(signature, atoms,
                                                       size):
            gain = cost - _expr_cost(build(new_terms, value + new_constant))
            if gain > 0 and (best is None or gain > best[0]):
                best = (gain, new_terms, new_constant)
        return best

    out = []
    modified = False
    for _, group in groups:
        # The whole group, then its sub sums
        subsets = [group]
        if 2 < len(group) <= 6:
            for pattern in range(1, 1 << len(group)):
                subset = [term for index, term in enumerate(group)
                          if (pattern >> index) & 1]
                if 2 <= len(subset) < len(group):
                    subsets.append(subset)
        best = None
        for subset in subsets:
            res = simplify(subset, constant)
            if res is not None and (best is None or res[0] > best[0]):
                best = res + (subset,)
        if best is None:
            out += [arg for _, _, _, arg in group]
            continue
        _, new_terms, new_constant, subset = best
        out += [term[3] for term in group
                if not any(term is other for other in subset)]
        out += new_terms
        constant += new_constant
        modified = True
    if not modified:
        return expr
    return build(out, constant)


def simp_not_arith(_, expr):
    """Two's complement identities
    (a + -1) ^ -1 => -a
    (-a) ^ -1 => a + -1
    """
    if expr.op != "^" or len(expr.args) != 2 or expr.size == 1:
        return expr
    arg, ones = expr.args
    if not (ones.is_int() and int(ones) == int(expr.mask)):
        return expr
    if arg.is_op("-") and len(arg.args) == 1:
        return arg.args[0] + ones
    if arg.is_op("+") and arg.args[-1] == ones:
        others = arg.args[:-1]
        if len(others) == 1:
            return -others[0]
        return -ExprOp("+", *others)
    return expr


# Truth tables of the bitwise functions of (a, b), a being the bit 0 of the
# row: a & b, a | b, a & ~b, ~a & b
_TABLE_AND, _TABLE_OR, _TABLE_A_NOT_B, _TABLE_B_NOT_A = 0b1000, 0b1110, 0b0010, 0b0100


def simp_mba_product(_, expr):
    """Product identity
    (a & b) * (a | b) + (a & ~b) * (~a & b) => a * b
    """
    if expr.op != "+" or len(expr.args) < 2 or expr.size == 1:
        return expr
    products = [arg for arg in expr.args
                if arg.is_op("*") and len(arg.args) == 2 and
                not arg.args[1].is_int()]
    for index, prod1 in enumerate(products):
        for prod2 in products[index + 1:]:
            atoms = []
            for factor in prod1.args + prod2.args:
                _mba_atoms(factor, atoms)
            if len(atoms) != 2:
                continue
            tables1 = set(_mba_table(factor, atoms) for factor in prod1.args)
            tables2 = set(_mba_table(factor, atoms) for factor in prod2.args)
            expected = [set([_TABLE_AND, _TABLE_OR]),
                        set([_TABLE_A_NOT_B, _TABLE_B_NOT_A])]
            if tables1 in expected and tables2 in expected and tables1 != tables2:
                others = [arg for arg in expr.args
                          if arg is not prod1 and arg is not prod2]
                product = atoms[0] * atoms[1]
                if not others:
                    return product
                return ExprOp("+", product, *others)
    return expr
