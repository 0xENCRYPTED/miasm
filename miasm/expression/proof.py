"""Sample then prove: cheap discovery of the value of an expression on random
inputs, then proof with an SMT solver.

Typical use: opaque predicates and obfuscated (MBA) jump targets, which are
constant whatever the inputs but out of reach of the simplifications.

    from miasm.expression.proof import find_constant, PROVEN

    result = find_constant(expr)
    if result.status == PROVEN:
        expr = result.value

The free variables of an expression are its identifiers and its memory
accesses. z3 is only needed for the proof: without it, the results are
reported as SAMPLED.
"""

import random

from miasm.expression.expression import ExprInt, ExprId
from miasm.expression.simplifications import expr_simp

# The value is the same for every input (proven by the solver)
PROVEN = "proven"
# The value is the same on every random input, without proof (no solver, time
# out, or operator unknown from the solver)
SAMPLED = "sampled"
# At least two inputs give different values: a counterexample is available
REFUTED = "refuted"
# The expression cannot be evaluated (unknown operator, ...)
UNKNOWN = "unknown"

# Interesting values for the inputs, used in addition to random ones
_CORNER_CASES = [0, 1, 2, -1, -2]


class ProofResult(object):
    """
    @status: PROVEN, SAMPLED, REFUTED or UNKNOWN
    @value: the constant (ExprInt) if @status is PROVEN or SAMPLED; for
    equivalences, ExprInt(1, 1)
    @counterexample: if @status is REFUTED, a {leaf: ExprInt} dictionary for
    which the property does not hold (None if only the solver refuted it
    without a usable model)
    """

    def __init__(self, status, value=None, counterexample=None):
        self.status = status
        self.value = value
        self.counterexample = counterexample

    def __bool__(self):
        return self.status == PROVEN

    __nonzero__ = __bool__

    def __repr__(self):
        return "<ProofResult %s %s>" % (self.status, self.value)


def get_leaves(expr):
    """Return the free variables of @expr: identifiers and memory accesses
    (a memory access whose pointer depends on an other one is still a single
    variable)"""
    leaves = set()

    def visit(node):
        if node.is_id() or node.is_mem():
            leaves.add(node)
        return node

    todo = [expr]
    seen = set()
    while todo:
        node = todo.pop()
        if node in seen:
            continue
        seen.add(node)
        if node.is_id() or node.is_mem():
            leaves.add(node)
            # The content of the pointer is not explored: the access is
            # replaced as a whole
            continue
        if node.is_op() or node.is_compose():
            todo.extend(node.args)
        elif node.is_cond():
            todo += [node.cond, node.src1, node.src2]
        elif node.is_slice():
            todo.append(node.arg)
    return leaves


def _random_value(rand, size, iteration):
    if iteration < len(_CORNER_CASES):
        return _CORNER_CASES[iteration] & ((1 << size) - 1)
    kind = rand.random()
    if kind < 0.2:
        # Small values
        return rand.getrandbits(min(size, 4))
    if kind < 0.3:
        # Single bit
        return 1 << rand.randrange(size)
    return rand.getrandbits(size)


def evaluate(expr, values, simplifier=expr_simp):
    """Evaluate @expr with the leaves replaced by @values ({leaf: ExprInt}).
    Return an integer, or None if the result is not concrete"""
    result = simplifier(expr.replace_expr(values))
    if not result.is_int():
        return None
    return int(result)


def sample(expr, count=64, seed=0, simplifier=expr_simp):
    """Evaluate @expr on @count random inputs.
    Return a list of (values, result), result being None if @expr cannot be
    evaluated"""
    rand = random.Random(seed)
    leaves = sorted(get_leaves(expr), key=str)
    out = []
    for iteration in range(count):
        values = {}
        for leaf in leaves:
            values[leaf] = ExprInt(_random_value(rand, leaf.size, iteration),
                                   leaf.size)
        result = evaluate(expr, values, simplifier)
        out.append((values, result))
        if result is None:
            break
    return out


def _prove_unsat(expr, timeout):
    """Check with z3 that the 1 bit @expr is never true.
    Return PROVEN, REFUTED or SAMPLED (no answer)"""
    try:
        import z3
        from miasm.ir.translators.z3_ir import TranslatorZ3
    except ImportError:
        return SAMPLED
    try:
        formula = TranslatorZ3().from_expr(expr)
    except NotImplementedError:
        # Operator unknown from the translator
        return SAMPLED
    solver = z3.Solver()
    if timeout:
        solver.set("timeout", int(timeout))
    solver.add(formula == 1)
    answer = solver.check()
    if answer == z3.unsat:
        return PROVEN
    if answer == z3.sat:
        return REFUTED
    return SAMPLED


def prove_equal(expr1, expr2, samples=64, timeout=10000, seed=0,
                simplifier=expr_simp):
    """Check that @expr1 and @expr2 are always equal: random evaluations,
    then proof with z3.
    @samples: number of random inputs
    @timeout: time allowed to the solver, in milliseconds (0: no limit)
    Return a ProofResult"""
    assert expr1.size == expr2.size
    diff = expr1 ^ expr2
    for values, result in sample(diff, samples, seed, simplifier):
        if result is None:
            return ProofResult(UNKNOWN)
        if result != 0:
            return ProofResult(REFUTED, counterexample=values)
    from miasm.expression.expression import ExprCond
    different = ExprCond(diff, ExprInt(1, 1), ExprInt(0, 1))
    status = _prove_unsat(different, timeout)
    if status == REFUTED:
        return ProofResult(REFUTED)
    return ProofResult(status, ExprInt(1, 1))


def find_constant(expr, samples=64, timeout=10000, seed=0,
                  simplifier=expr_simp):
    """If @expr has the same value on @samples random inputs, try to prove
    with z3 that it is constant.
    @timeout: time allowed to the solver, in milliseconds (0: no limit)
    Return a ProofResult: its value is the constant if the status is PROVEN
    or SAMPLED"""
    if expr.is_int():
        return ProofResult(PROVEN, expr)
    first = None
    first_values = None
    for values, result in sample(expr, samples, seed, simplifier):
        if result is None:
            return ProofResult(UNKNOWN)
        if first is None:
            first, first_values = result, values
        elif result != first:
            return ProofResult(REFUTED, counterexample=values)
    constant = ExprInt(first, expr.size)
    result = prove_equal(expr, constant, samples=0, timeout=timeout,
                         simplifier=simplifier)
    if result.status == REFUTED:
        return ProofResult(REFUTED)
    return ProofResult(result.status, constant)


def simp_by_proof(expr, samples=64, timeout=10000):
    """Return the constant value of @expr if it is proven, else @expr"""
    result = find_constant(expr, samples, timeout)
    if result.status == PROVEN:
        return result.value
    return expr
