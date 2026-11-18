"""Constant folding of the floating point operators (fadd, fsub, fmul,
fdiv, fsqrt, sint_to_fp, fp_to_sint, fpconvert_fp, fpround, ucomis*).

Not part of the default simplifications: the operations are evaluated with
the host floats, in "round to nearest" mode, and the NaN produced are the x86
default ones. This is what the python jitter needs.

    simplifier.enable_passes(PASS_FLOAT)
"""
import math
import struct

from miasm.expression.expression import ExprInt, ExprOp


def f2i(value, size):
    """Float -> IEEE754 encoding, on @size bits"""
    fmt = {32: ("<f", "<I"), 64: ("<d", "<Q")}[size]
    if math.isnan(value):
        # x86 default NaN
        return {32: 0xFFC00000, 64: 0xFFF8000000000000}[size]
    try:
        raw = struct.pack(fmt[0], value)
    except OverflowError:
        raw = struct.pack(fmt[0], math.copysign(float("inf"), value))
    return struct.unpack(fmt[1], raw)[0]


def i2f(value, size):
    """IEEE754 encoding on @size bits -> float"""
    fmt = {32: ("<f", "<I"), 64: ("<d", "<Q")}[size]
    return struct.unpack(fmt[0], struct.pack(fmt[1], value))[0]


def fp_div(a, b):
    if b == 0:
        if a == 0 or math.isnan(a):
            return float("nan")
        sign = math.copysign(1, a) * math.copysign(1, b)
        return sign * float("inf")
    return a / b


def fp_round(mode, a):
    if math.isnan(a) or math.isinf(a):
        return a
    if mode == "fpround_towardsnearest":
        res = float(round(a))
    elif mode == "fpround_down":
        res = float(math.floor(a))
    elif mode == "fpround_up":
        res = float(math.ceil(a))
    else:
        res = float(math.trunc(a))
    if res == 0:
        res = math.copysign(0.0, a)
    return res


def fp_to_sint(a, size):
    """Round to nearest (even), 'integer indefinite' on overflow"""
    if math.isnan(a) or math.isinf(a):
        return 1 << (size - 1)
    res = round(a)
    if not -(1 << (size - 1)) <= res < (1 << (size - 1)):
        return 1 << (size - 1)
    return res & ((1 << size) - 1)


def to_signed(value, size):
    return value - (1 << size) if value >> (size - 1) else value


def simp_float_cst(_, expr):
    """Simplification pass: evaluate float operators on constants"""
    if not expr.is_op() or not all(arg.is_int() for arg in expr.args):
        return expr
    op = expr.op
    args = expr.args
    size = args[0].size
    if op in ("fadd", "fsub", "fmul", "fdiv"):
        a, b = i2f(int(args[0]), size), i2f(int(args[1]), size)
        if op == "fadd":
            res = a + b
        elif op == "fsub":
            res = a - b
        elif op == "fmul":
            res = a * b
        else:
            res = fp_div(a, b)
        return ExprInt(f2i(res, size), size)
    if op == "fsqrt":
        a = i2f(int(args[0]), size)
        if a < 0:
            res = float("nan")
        else:
            res = math.sqrt(a) if a != 0 else a
        return ExprInt(f2i(res, size), size)
    if op == "sint_to_fp":
        return ExprInt(f2i(float(to_signed(int(args[0]), size)), size), size)
    if op.startswith("fp_to_sint"):
        return ExprInt(fp_to_sint(i2f(int(args[0]), size), expr.size),
                       expr.size)
    if op.startswith("fpconvert_fp"):
        return ExprInt(f2i(i2f(int(args[0]), size), expr.size), expr.size)
    if op.startswith("fpround_"):
        return ExprInt(f2i(fp_round(op, i2f(int(args[0]), size)), size), size)
    if op.startswith("ucomis"):
        a, b = i2f(int(args[0]), size), i2f(int(args[1]), size)
        unordered = math.isnan(a) or math.isnan(b)
        kind = op[-2:]
        if kind == "zf":
            return ExprInt(int(unordered or a == b), 1)
        if kind == "pf":
            return ExprInt(int(unordered), 1)
        if kind == "cf":
            return ExprInt(int(unordered or a < b), 1)
    return expr


PASS_FLOAT = {ExprOp: [simp_float_cst]}
