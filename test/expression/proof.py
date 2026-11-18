#! /usr/bin/env python
"""Sample then prove (miasm.expression.proof)"""
from __future__ import print_function

from miasm.expression.expression import ExprId, ExprInt, ExprMem, ExprOp, \
    ExprCond, ExprCompose, TOK_EQUAL
from miasm.expression.proof import find_constant, prove_equal, simp_by_proof, \
    sample, get_leaves, PROVEN, SAMPLED, REFUTED, UNKNOWN

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False
# Status of a true property
OK = PROVEN if HAS_Z3 else SAMPLED

a = ExprId("a", 64)
b = ExprId("b", 64)
m = ExprMem(ExprId("p", 64), 64)


def I(value, size=64):
    return ExprInt(value, size)


# Leaves: identifiers and memory accesses
assert get_leaves((a + m) ^ ExprMem(a + b, 8).zeroExtend(64)) == set(
    [a, m, ExprMem(a + b, 8)])
assert len(sample(a + b, 10)) == 10

# Obfuscated constant target: opaque null term, multiplied by something
target = ((a | b) * I(2) - (a ^ b) - a - b) * (m | I(1)) + I(0x14001700C)
result = find_constant(target)
assert result.status == OK and result.value == I(0x14001700C)
if HAS_Z3:
    assert result and simp_by_proof(target) == I(0x14001700C)

# Mask with constants, null byte in a compose, comparison terms: the kind of
# non linear expression out of reach of the simplifications
x8 = ExprCompose(I(0, 8), a[8:64])
term = ExprCond(ExprOp(TOK_EQUAL, x8 & I(0xFF), I(0)), I(0x1000), I(0x2000))
target = (term + (a & I(0xF0F0)) - (a & I(0xF000)) - (a & I(0xF0))) + I(0x400000)
result = find_constant(target)
assert result.status == OK and result.value == I(0x401000)

# Opaque predicates
predicate = (a * (a + I(1)))[0:1]
result = find_constant(predicate)
assert result.status == OK and int(result.value) == 0
predicate = ExprOp(TOK_EQUAL, (a * a + a)[0:1], ExprInt(1, 1))
result = find_constant(predicate)
assert result.status == OK and int(result.value) == 0

# Not constant: refuted by the samples, with the inputs
result = find_constant(a & I(0xFF))
assert result.status == REFUTED and result.counterexample
assert not result

# Constant on every sample, but not constant: only the solver sees it
rare = ExprCond(ExprOp(TOK_EQUAL, a & I(0xFFFFFFFF), I(0x12345678)),
                I(1), I(0))
result = find_constant(rare)
if HAS_Z3:
    assert result.status == REFUTED
else:
    assert result.status == SAMPLED

# Unknown operator: no evaluation
result = find_constant(ExprOp("x86_cpuid", a, I(0)))
assert result.status == UNKNOWN

# Equivalences
assert prove_equal((a ^ b) + (a & b) * I(2), a + b).status == OK
assert prove_equal(a + b, a | b).status == REFUTED
assert prove_equal(m + m, m * I(2)).status == OK

print("OK")
