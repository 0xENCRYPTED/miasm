#! /usr/bin/env python
"""Regression tests for the bit level simplifications
(miasm.expression.simplifications_bits)"""
from __future__ import print_function

import random

from miasm.expression.expression import ExprId, ExprInt, ExprOp, ExprCond, \
    ExprCompose, ExprMem, TOK_EQUAL, TOK_INF_UNSIGNED, TOK_INF_SIGNED, \
    TOK_INF_EQUAL_UNSIGNED
from miasm.expression.simplifications import expr_simp, expr_simp_bits

try:
    import z3
    from miasm.ir.translators.z3_ir import TranslatorZ3
    TRANS = TranslatorZ3()
except ImportError:
    z3 = None

random.seed(0)

a = ExprId("a", 32)
b = ExprId("b", 32)
c = ExprId("c", 1)
d = ExprId("d", 1)
x = ExprId("x", 8)
y = ExprId("y", 8)
m = ExprMem(ExprId("p", 64), 16)
w1 = ExprId("w1", 128)
w2 = ExprId("w2", 128)

ONE = ExprInt(1, 1)
ZERO = ExprInt(0, 1)


def bit(expr, index):
    return expr[index:index + 1]


def maj1(p, q, r):
    return (p & q) | (p & r) | (q & r)


def maj2(p, q, r):
    return (p & q) | (r & (p ^ q))


def maj3(p, q, r):
    return (p & q) ^ (r & (p ^ q))


def maj4(p, q, r):
    # Multiplexer based carry
    return ExprCond(p ^ q, r, p)


def ripple_add(op1, op2, size, carry=ZERO, maj=maj1, with_carry=False):
    """Bit-serial adder of the @size low bits"""
    out = []
    for i in range(size):
        p, q = bit(op1, i), bit(op2, i)
        out.append(p ^ q ^ carry)
        carry = maj(p, q, carry)
    if with_carry:
        out.append(carry)
    return ExprCompose(*out)


def ripple_sub(op1, op2, size, maj=maj1):
    """Bit-serial subtractor (borrow chain)"""
    out = []
    borrow = ZERO
    for i in range(size):
        p, q = bit(op1, i), bit(op2, i)
        out.append(p ^ q ^ borrow)
        borrow = maj(p ^ ONE, q, borrow)
    return ExprCompose(*out)


def serial_lower(op1, op2, size, signed=False):
    lower = ZERO
    for i in range(size):
        p, q = bit(op1, i), bit(op2, i)
        decided = p if (signed and i == size - 1) else q
        lower = ExprCond(p ^ q, decided, lower)
    return lower


def shift_add_mul(op1, op2, size):
    terms = []
    for i in range(size):
        terms.append(ExprCond(bit(op2, i), op1 << ExprInt(i, op1.size),
                              ExprInt(0, op1.size)))
    return ExprOp("+", *terms)


def evaluate(expr, values):
    res = expr_simp(expr.replace_expr(values))
    assert res.is_int(), "Cannot evaluate %s" % res
    return int(res)


def check_equiv(before, after):
    """@before and @after must be equivalent: random evaluations, then z3"""
    assert before.size == after.size
    leaves = set()
    for expr in (before, after):
        leaves.update(leaf for leaf in expr.get_r(mem_read=True)
                      if leaf.is_id() or leaf.is_mem())
        leaves.update(expr.get_r())
    leaves = [leaf for leaf in leaves if leaf.is_id()]
    if not any(leaf.is_mem() for leaf in before.get_r(mem_read=True)):
        for _ in range(50):
            values = dict((leaf, ExprInt(random.getrandbits(leaf.size),
                                         leaf.size)) for leaf in leaves)
            assert evaluate(before, values) == evaluate(after, values), \
                "Not equivalent: %s" % after
    if z3 is not None:
        solver = z3.Solver()
        # Some proofs (multiplications) are out of reach: only a
        # counterexample is an error
        solver.set("timeout", 3000)
        solver.add(TRANS.from_expr(before) != TRANS.from_expr(after))
        assert solver.check() != z3.sat, "Not equivalent: %s" % after


# (input, expected result)
tests = [
    # 1 bit arithmetic
    (bit(a, 0) + bit(b, 0), bit(a, 0) ^ bit(b, 0)),
    (bit(a, 0) - bit(b, 0), bit(a, 0) ^ bit(b, 0)),
    (bit(a, 0) * bit(b, 0), bit(a, 0) & bit(b, 0)),
    (-c, c),
    (ExprCond(c, ONE, ZERO), c),
    (ExprCond(c, ZERO, ONE), c ^ ONE),

    # Masks
    (c.signExtend(8)[7:8], c),
    (c.signExtend(32)[8:16], c.signExtend(8)),
    (a & c.signExtend(32), ExprCond(c, a, ExprInt(0, 32))),
    ((a & c.signExtend(32)) | (b & (c.signExtend(32) ^ ExprInt(-1, 32))),
     ExprCond(c, a, b)),
    (b ^ ((a ^ b) & c.signExtend(32)), ExprCond(c, a, b)),

    # Lifting
    (ExprCompose(*[bit(a, i) ^ bit(b, i) for i in range(32)]), a ^ b),
    (ExprCompose(*[bit(a, i) & bit(b, i) for i in range(32)]), a & b),
    (ExprCompose(*[bit(a, i) | bit(b, i) for i in range(8)]),
     a[0:8] | b[0:8]),
    (ExprCompose(*[bit(a, i) ^ ONE for i in range(32)]),
     a ^ ExprInt(-1, 32)),
    (ExprCompose(*[(bit(a, i) & bit(b, i)) ^ bit(a, i + 8)
                   for i in range(8)]),
     (a[0:8] & b[0:8]) ^ a[8:16]),
    # Constant operand on some elements only
    (ExprCompose(bit(x, 0), bit(x, 1) ^ ONE, bit(x, 2), bit(x, 3) ^ ONE,
                 bit(x, 4), bit(x, 5), bit(x, 6), bit(x, 7)),
     x ^ ExprInt(0b1010, 8)),
    # Shared operand
    (ExprCompose(*[bit(x, i) & c for i in range(8)]),
     ExprCond(c, x, ExprInt(0, 8))),
    (ExprCompose(c, c, c, c), c.signExtend(4)),
    # Same condition
    (ExprCompose(*[ExprCond(c, bit(x, i), bit(y, i)) for i in range(8)]),
     ExprCond(c, x, y)),
    # Wider elements (bit-slicing with byte masks)
    (ExprCompose(*[a[i:i + 8] ^ b[i:i + 8] for i in range(0, 32, 8)]), a ^ b),
    # Only a part of the vector
    (ExprCompose(*([bit(x, i) ^ bit(y, i) for i in range(4)] + [m[0:4]])),
     ExprCompose(x[0:4] ^ y[0:4], m[0:4])),

    # Adders
    (ripple_add(a, b, 32), a + b),
    (ripple_add(a, b, 32, maj=maj2), a + b),
    (ripple_add(a, b, 32, maj=maj3), a + b),
    (ripple_add(a, b, 32, maj=maj4), a + b),
    (ripple_add(x, y, 8), x + y),
    (ripple_add(a, b, 16), a[0:16] + b[0:16]),
    (ripple_add(x, y, 8, carry=c), x + y + c.zeroExtend(8)),
    (ripple_add(x, y, 8, with_carry=True),
     x.zeroExtend(9) + y.zeroExtend(9)),
    (ripple_add(x, ExprInt(1, 8), 8), x + ExprInt(1, 8)),
    (ripple_add(x, ExprInt(0x35, 8), 8), x + ExprInt(0x35, 8)),
    (ripple_sub(a, b, 32), a - b),
    (ripple_sub(x, y, 8, maj=maj2), x - y),
    # Two's complement: ~x + 1
    (ripple_add(x ^ ExprInt(0xff, 8), ExprInt(1, 8), 8), -x),
    # Nested: (x + y) + x
    (ripple_add(ripple_add(x, y, 8), x, 8), x + y + x),
    # Adder next to something else
    (ExprCompose(ripple_add(x, y, 8), m),
     ExprCompose(x + y, m)),

    # Comparisons
    (ExprOp("|", *[bit(x, i) ^ bit(y, i) for i in range(8)]),
     ExprOp(TOK_EQUAL, x, y) ^ ONE),
    (ExprOp("&", *[bit(x, i) ^ bit(y, i) ^ ONE for i in range(8)]),
     ExprOp(TOK_EQUAL, x, y)),
    (serial_lower(x, y, 8), ExprOp(TOK_INF_UNSIGNED, x, y)),
    (serial_lower(a, b, 32), ExprOp(TOK_INF_UNSIGNED, a, b)),
    (serial_lower(x, y, 8, signed=True), ExprOp(TOK_INF_SIGNED, x, y)),
    (serial_lower(y, x, 8) ^ ONE, ExprOp(TOK_INF_EQUAL_UNSIGNED, x, y)),
    (ExprOp("|", *[bit(x, i) for i in range(8)]),
     ExprCond(x, ONE, ZERO)),
    (ExprOp("&", *[bit(x, i) for i in range(8)]),
     ExprOp(TOK_EQUAL, x, ExprInt(0xff, 8))),
    (ExprOp("&", bit(x, 0), bit(x, 1) ^ ONE, bit(x, 2), bit(x, 3) ^ ONE),
     ExprOp(TOK_EQUAL, x[0:4], ExprInt(0b0101, 4))),

    # Wide operands
    (ripple_add(w1, w2, 128), w1 + w2),
    (ripple_sub(w1, w2, 128, maj=maj3), w1 - w2),
    (serial_lower(w1, w2, 128), ExprOp(TOK_INF_UNSIGNED, w1, w2)),

    # Shift and add multiplication
    (shift_add_mul(a, b, 32), a * b),
    (shift_add_mul(a, b, 8), a * b[0:8].zeroExtend(32)),
]

for before, expected in tests:
    after = expr_simp_bits(before)
    expected = expr_simp_bits(expected)
    # Do not print @before: its string may be exponentially big
    print("->", str(after)[:100])
    assert after == expected, "got      %s\n expected %s" % (
        str(after)[:300], str(expected)[:300])
    check_equiv(before, after)

# Expressions which must be left untouched (and stay equivalent)
untouched = [
    a + b,
    a ^ b,
    ExprCompose(x, y),
    ExprCompose(bit(x, 3), bit(y, 1), bit(x, 0)),
    ExprCond(a, b, a),
    # Not an adder: wrong carry
    ExprCompose(*[bit(x, i) ^ bit(y, i) ^ bit(x, (i + 3) % 8)
                  for i in range(8)]),
]
for expr in untouched:
    after = expr_simp_bits(expr)
    check_equiv(expr, after)

# Soundness on random bit level expressions
LEAVES = [bit(x, i) for i in range(8)] + [bit(y, i) for i in range(8)] + \
    [c, d, ONE, ZERO]


def random_bit(depth):
    if depth == 0 or random.random() < 0.2:
        return random.choice(LEAVES)
    kind = random.random()
    if kind < 0.6:
        op = random.choice(["^", "&", "|", "+", "-", "*"])
        return ExprOp(op, random_bit(depth - 1), random_bit(depth - 1))
    if kind < 0.8:
        return ExprCond(random_bit(depth - 1), random_bit(depth - 1),
                        random_bit(depth - 1))
    return random_bit(depth - 1) ^ ONE


for _ in range(300):
    count = random.choice([2, 4, 8])
    expr = ExprCompose(*[random_bit(3) for _ in range(count)])
    after = expr_simp_bits(expr)
    check_equiv(expr, after)

print("OK")
