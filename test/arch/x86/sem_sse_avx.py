#! /usr/bin/env python
"""Regression tests for the semantic of SSE (SSE3, SSSE3, SSE4.1, SSE4.2, AES,
SHA additions) and AVX / AVX2 instructions.

Each instruction is symbolically executed from a concrete, pseudo random
state, and the result is compared with the one obtained by running the same
instruction on a real CPU (reference values are stored in sem_sse_avx.json,
see sem_sse_avx_native.py to regenerate them).

The jitter is not needed.
"""
from __future__ import print_function

import json
import math
import os
import re
import struct
import sys
import zlib

from miasm.analysis.machine import Machine
from miasm.arch.x86 import regs
from miasm.arch.x86.arch import mn_x86
from miasm.core.locationdb import LocationDB
from miasm.expression.expression import ExprInt, ExprMem, ExprOp, ExprLoc
from miasm.expression.simplifications import ExpressionSimplifier
from miasm.ir.symbexec import SymbolicExecutionEngine

VECTORS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "sem_sse_avx.json")

# State layout
MEM_SIZE = 256
# Fake address of the memory area (the reference run uses a real one)
MEM_BASE = 0x10000000
# Registers usable by a tested instruction
FREE_GPRS = ["RAX", "RCX", "RDX", "RSI", "R8", "R9", "R10", "R11"]
# RBX and RDI point in the memory area
RBX_OFFSET = 48
RDI_OFFSET = 128
ARITH_FLAGS = {"cf": 0, "pf": 2, "af": 4, "zf": 6, "nf": 7, "of": 11}


###############################################################################
# Deterministic inputs
###############################################################################

class DetRandom(object):
    """Minimal PRNG (xorshift64*), to be independent of the Python version"""

    def __init__(self, seed):
        self.state = (seed * 0x9E3779B97F4A7C15 + 0x1234567) & (2 ** 64 - 1)
        if self.state == 0:
            self.state = 1

    def next64(self):
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & (2 ** 64 - 1)
        x ^= x >> 27
        self.state = x
        return (x * 0x2545F4914F6CDD1D) & (2 ** 64 - 1)

    def getrandbits(self, count):
        out = 0
        for i in range(0, count, 64):
            out |= self.next64() << i
        return out & ((1 << count) - 1)

    def random(self):
        return (self.next64() >> 11) / float(1 << 53)

    def randrange(self, start, stop):
        return start + self.next64() % (stop - start)

    def choice(self, seq):
        return seq[self.next64() % len(seq)]

    def uniform(self, low, high):
        return low + (high - low) * self.random()


FLOAT_VALUES = [0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5, -2.5, 3.0, 100.25,
                -7.75, 1e10, -1e-10, 123456.5, 0.1, 1 / 3.0, 2.0 ** 31,
                -2.0 ** 31, 2.0 ** 40, float("inf"), -float("inf")]
NAN_VALUES = {
    32: [0x7fc00000, 0xffc00001, 0x7f800001],
    64: [0x7ff8000000000000, 0xfff8000000000001, 0x7ff0000000000001],
}
INT_BYTES = [0, 0, 1, 2, 0x7f, 0x80, 0xff, 0x41, 0x61]


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


def gen_vector(rnd, kind):
    """Return a 256 bit value
    @kind: 'int', 'str', 'f32', 'f64', 'f32nan', 'f64nan'"""
    if kind == "int":
        if rnd.random() < 0.5:
            return rnd.getrandbits(256)
        out = 0
        for i in range(32):
            if rnd.random() < 0.1:
                byte = rnd.getrandbits(8)
            else:
                byte = rnd.choice(INT_BYTES)
            out |= byte << (8 * i)
        return out
    if kind == "str":
        out = 0
        for i in range(32):
            out |= rnd.choice(list(b"abcdefgh\x00")) << (8 * i)
        return out
    size = 32 if "32" in kind else 64
    out = 0
    for i in range(256 // size):
        choice = rnd.random()
        if kind.endswith("nan") and choice < 0.15:
            value = rnd.choice(NAN_VALUES[size])
        elif choice < 0.6:
            value = f2i(rnd.choice(FLOAT_VALUES), size)
        else:
            value = f2i(rnd.uniform(-1000, 1000), size)
        out |= value << (i * size)
    return out


def gen_state(seed, kind, mem_base=MEM_BASE):
    """Generate the initial state of a test"""
    rnd = DetRandom(seed)
    state = {"ymm": [gen_vector(rnd, kind) for _ in range(16)], "gpr": {}}
    for name in FREE_GPRS:
        choice = rnd.random()
        if choice < 0.4:
            value = rnd.getrandbits(64)
        elif choice < 0.7:
            value = rnd.randrange(0, 20)
        else:
            value = (-rnd.randrange(0, 20)) & ((1 << 64) - 1)
        state["gpr"][name] = value
    state["gpr"]["RBX"] = mem_base + RBX_OFFSET
    state["gpr"]["RDI"] = mem_base + RDI_OFFSET
    flags = 0x202
    for bit in ARITH_FLAGS.values():
        if rnd.random() < 0.5:
            flags |= 1 << bit
    state["flags"] = flags
    mem = 0
    for i in range(MEM_SIZE // 32):
        mem |= gen_vector(rnd, kind) << (256 * i)
    state["mem"] = mem
    state["mm"] = [gen_vector(rnd, kind) & (2 ** 64 - 1) for _ in range(8)]
    return state


FLOAT32 = re.compile(
    r"^V?(\w+PS|\w+SS|CVTPS2\w+|CVTTPS2\w+|CVTSS2\w+|CVTTSS2\w+|"
    r"BROADCASTSS|EXTRACTPS|INSERTPS)$")
FLOAT64 = re.compile(
    r"^V?(\w+PD|\w+SD|CVTPD2\w+|CVTTPD2\w+|CVTSD2\w+|CVTTSD2\w+|"
    r"BROADCASTSD|MOVDDUP)$")
# Pure bit manipulations: NaN and random bits are welcome
NAN_OK = re.compile(r"^V?(CMP|AND|ANDN|OR|XOR|MOV|UNPCK|SHUF|BLEND|PERM|"
                    r"BROADCAST|INSERT|EXTRACT|TEST|MASKMOV|UCOMI)")
INT_SOURCE = re.compile(r"^V?(CVTDQ2|CVTSI2|CVTPI2|MOVD$|MOVQ$)")


def state_kinds(name):
    """Kinds of input states suitable for the mnemonic @name"""
    if INT_SOURCE.match(name):
        return ["int"]
    if re.match(r"^V?PCMP[EI]STR", name):
        return ["str", "str", "int"]
    for regexp, base in ((FLOAT32, "f32"), (FLOAT64, "f64")):
        if regexp.match(name):
            if NAN_OK.match(name):
                return [base, base + "nan", "int"]
            return [base]
    return ["int"]


def test_seed(text, iteration):
    return (zlib.crc32(text.encode()) & 0xffffffff) * 16 + iteration


###############################################################################
# Float operators evaluation (they are not folded by miasm)
###############################################################################

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


def fold_float(_, expr):
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


simp = ExpressionSimplifier()
simp.enable_passes(ExpressionSimplifier.PASS_COMMONS)
simp.enable_passes(ExpressionSimplifier.PASS_HEAVY)
simp.enable_passes({ExprOp: [fold_float]})

###############################################################################
# Symbolic execution
###############################################################################
loc_db = LocationDB()
machine = Machine("x86_64")
lifter = machine.lifter(loc_db)
FLAG_IDS = dict((name, getattr(regs, name)) for name in ARITH_FLAGS)


def run_miasm(data, state, mem_base=MEM_BASE):
    """Execute the instruction encoded by @data on @state
    Return the resulting state. Only the flags written by the instruction
    are present in the result"""
    instr = mn_x86.dis(data, 64)
    instr.offset = 0x1000
    assert instr.l == len(data)
    ircfg = lifter.new_ircfg()
    lifter.add_instr_to_ircfg(instr, ircfg)

    init = {}
    for i in range(16):
        init[regs.regs_xmm_expr[i]] = ExprInt(
            state["ymm"][i] & ((1 << 128) - 1), 128)
        init[regs.regs_ymmh_expr[i]] = ExprInt(state["ymm"][i] >> 128, 128)
    for i in range(8):
        init[regs.regs_mm_expr[i]] = ExprInt(state["mm"][i], 64)
    for name, value in state["gpr"].items():
        init[getattr(regs, name)] = ExprInt(value, 64)
    for name, bit in ARITH_FLAGS.items():
        init[FLAG_IDS[name]] = ExprInt((state["flags"] >> bit) & 1, 1)
    mem = ExprMem(ExprInt(mem_base, 64), MEM_SIZE * 8)
    init[mem] = ExprInt(state["mem"], MEM_SIZE * 8)
    symbexec = SymbolicExecutionEngine(lifter, init)

    # An instruction may be made of several IR blocks
    cur = ExprLoc(loc_db.get_or_create_offset_location(0x1000), 64)
    written_flags = set()
    for _ in range(200):
        irblock = ircfg.get_block(cur.loc_key) if cur.is_loc() else None
        if irblock is None:
            break
        for assignblk in irblock:
            for name, flag in FLAG_IDS.items():
                if flag in assignblk:
                    written_flags.add(name)
        cur = simp(symbexec.eval_updt_irblock(irblock))
    else:
        raise RuntimeError("Too many IR blocks")

    def get(expr):
        value = simp(symbexec.eval_expr(expr))
        if not value.is_int():
            raise ValueError("%s is not concrete: %s" % (expr,
                                                         str(value)[:300]))
        return int(value)

    result = {"ymm": [], "gpr": {}, "flags": {}}
    result["mm"] = [get(regs.regs_mm_expr[i]) for i in range(8)]
    for i in range(16):
        result["ymm"].append(get(regs.regs_xmm_expr[i]) |
                             (get(regs.regs_ymmh_expr[i]) << 128))
    for name in state["gpr"]:
        result["gpr"][name] = get(getattr(regs, name))
    for name in written_flags:
        result["flags"][name] = get(FLAG_IDS[name])
    result["mem"] = get(mem)
    return result


def diff_state(before, after):
    """Return what changed between two states, as a JSON friendly dict"""
    out = {}
    ymm = dict(("%d" % i, "%x" % after["ymm"][i]) for i in range(16)
               if after["ymm"][i] != before["ymm"][i])
    if ymm:
        out["ymm"] = ymm
    mmx = dict(("%d" % i, "%x" % after["mm"][i]) for i in range(8)
               if after["mm"][i] != before["mm"][i])
    if mmx:
        out["mm"] = mmx
    gpr = dict((name, "%x" % value) for name, value in after["gpr"].items()
               if value != before["gpr"][name])
    if gpr:
        out["gpr"] = gpr
    if after["mem"] != before["mem"]:
        out["mem"] = "%x" % after["mem"]
    return out


def check(test):
    """Run the test @test (entry of the JSON file). Return a list of errors"""
    data = bytes(bytearray.fromhex(test["bytes"]))
    state = gen_state(test["seed"], test["kind"])
    result = run_miasm(data, state)
    errors = []
    expected = test["out"]
    got = diff_state(state, result)
    for key in ("ymm", "mm", "gpr"):
        if got.get(key, {}) != expected.get(key, {}):
            errors.append("%s: got %r, expected %r" % (
                key, got.get(key, {}), expected.get(key, {})))
    if got.get("mem") != expected.get("mem"):
        errors.append("mem: got %r, expected %r" % (got.get("mem"),
                                                     expected.get("mem")))
    for name, value in result["flags"].items():
        if value != expected["flags"][name]:
            errors.append("%s: got %d, expected %d" % (
                name, value, expected["flags"][name]))
    return errors


def main():
    with open(VECTORS) as fdesc:
        tests = json.load(fdesc)
    filt = re.compile(sys.argv[1]) if len(sys.argv) > 1 else None
    failed = total = 0
    for test in tests:
        if filt and not filt.search(test["text"]):
            continue
        total += 1
        errors = check(test)
        if errors:
            failed += 1
            print("FAIL %s (%s)" % (test["text"], test["bytes"]))
            for error in errors:
                print("    " + error[:300])
    print("%d tests, %d failed" % (total, failed))
    assert total > 0
    assert failed == 0


if __name__ == "__main__":
    main()
