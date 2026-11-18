#! /usr/bin/env python
"""Differential test of the SSE / AVX semantic against the host CPU, and
generator of the reference values used by sem_sse_avx.py.

Each instruction is executed natively (the code is written in an executable
page and called through ctypes) and symbolically by miasm, from the same
pseudo random state; the resulting states are then compared.

Requirements: Windows x86_64 host with AVX2 (AES-NI and SHA-NI instructions
are skipped if they are not supported by the CPU).

Usage:
    sem_sse_avx_native.py [-i iterations] [-f regexp] [-o out.json] [vectors]

@vectors is a text file with one "mode|instruction|hex bytes" by line. By
default, the instructions of sem_sse_avx.json are replayed.
With -o, the reference values (one state by instruction) are written.
"""
from __future__ import print_function

import argparse
import ctypes
import json
import re
import struct

from miasm.arch.x86.arch import mn_x86

import sem_sse_avx as common
from sem_sse_avx import MEM_SIZE, FREE_GPRS, ARITH_FLAGS

###############################################################################
# Native execution
###############################################################################
OFF_GPR = 512
OFF_FLAGS = 640
OFF_MEM = 704
OFF_MM = 960
GPRS = ["RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI",
        "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15"]

kernel32 = ctypes.windll.kernel32
kernel32.VirtualAlloc.restype = ctypes.c_void_p
kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_uint32, ctypes.c_uint32]
# MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE / PAGE_READWRITE
CODE = kernel32.VirtualAlloc(None, 0x2000, 0x3000, 0x40)
STATE = kernel32.VirtualAlloc(None, 0x1000, 0x3000, 0x04)
assert CODE and STATE
MEM_BASE = STATE + OFF_MEM
FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(CODE)


def asm(text):
    instr = mn_x86.fromstring(text, common.loc_db, 64)
    return mn_x86.asm(instr)[0]


def build_stub():
    """Return the code surrounding the tested instruction:
    void stub(state *RCX)"""
    before = b""
    for reg in ["RBX", "RSI", "RDI", "RBP", "R12", "R13", "R14", "R15"]:
        before += asm("PUSH %s" % reg)
    before += asm("SUB RSP, 0xA8")
    for i in range(6, 16):
        before += asm("MOVDQU XMMWORD PTR [RSP+0x%x], XMM%d" % (
            (i - 6) * 16, i))
    before += asm("MOV R15, RCX")
    for i in range(16):
        before += asm("VMOVDQU YMM%d, YMMWORD PTR [R15+0x%x]" % (i, 32 * i))
    for i in range(8):
        before += asm("MOVQ MM%d, QWORD PTR [R15+0x%x]" % (i, OFF_MM + 8 * i))
    before += asm("PUSH QWORD PTR [R15+0x%x]" % OFF_FLAGS)
    before += asm("POPFQ")
    for reg in FREE_GPRS + ["RBX", "RDI"]:
        before += asm("MOV %s, QWORD PTR [R15+0x%x]" % (
            reg, OFF_GPR + 8 * GPRS.index(reg)))

    after = b""
    for reg in FREE_GPRS + ["RBX", "RDI"]:
        after += asm("MOV QWORD PTR [R15+0x%x], %s" % (
            OFF_GPR + 8 * GPRS.index(reg), reg))
    after += asm("PUSHFQ")
    after += asm("POP RAX")
    after += asm("MOV QWORD PTR [R15+0x%x], RAX" % OFF_FLAGS)
    for i in range(16):
        after += asm("VMOVDQU YMMWORD PTR [R15+0x%x], YMM%d" % (32 * i, i))
    for i in range(8):
        after += asm("MOVQ QWORD PTR [R15+0x%x], MM%d" % (OFF_MM + 8 * i, i))
    # Restore the x87 state
    after += asm("EMMS")
    after += asm("VZEROUPPER")
    for i in range(6, 16):
        after += asm("MOVDQU XMM%d, XMMWORD PTR [RSP+0x%x]" % (
            i, (i - 6) * 16))
    after += asm("ADD RSP, 0xA8")
    for reg in ["R15", "R14", "R13", "R12", "RBP", "RDI", "RSI", "RBX"]:
        after += asm("POP %s" % reg)
    after += asm("RET")
    return before, after


STUB_BEFORE, STUB_AFTER = build_stub()


def call(code):
    ctypes.memmove(CODE, code, len(code))
    kernel32.FlushInstructionCache(ctypes.c_void_p(-1),
                                   ctypes.c_void_p(CODE), len(code))
    FUNC(STATE)


def run_native(data, state):
    """Execute the instruction encoded by @data on @state"""
    buf = b"".join(value.to_bytes(32, "little") for value in state["ymm"])
    buf += b"".join(state["gpr"].get(name, 0).to_bytes(8, "little")
                    for name in GPRS)
    buf += state["flags"].to_bytes(8, "little")
    buf = buf.ljust(OFF_MEM, b"\x00")
    buf += state["mem"].to_bytes(MEM_SIZE, "little")
    assert len(buf) == OFF_MM
    buf += b"".join(value.to_bytes(8, "little") for value in state["mm"])
    ctypes.memmove(STATE, buf, len(buf))
    call(STUB_BEFORE + data + STUB_AFTER)
    out = ctypes.string_at(STATE, OFF_MM + 64)
    result = {
        "mm": [int.from_bytes(out[OFF_MM + 8 * i:OFF_MM + 8 * i + 8],
                              "little") for i in range(8)],
        "ymm": [int.from_bytes(out[32 * i:32 * i + 32], "little")
                for i in range(16)],
        "gpr": {},
        "flags": int.from_bytes(out[OFF_FLAGS:OFF_FLAGS + 8], "little"),
        "mem": int.from_bytes(out[OFF_MEM:OFF_MEM + MEM_SIZE], "little"),
    }
    for name in state["gpr"]:
        off = OFF_GPR + 8 * GPRS.index(name)
        result["gpr"][name] = int.from_bytes(out[off:off + 8], "little")
    return result


def cpuid(leaf, subleaf=0):
    code = (asm("PUSH RBX") + asm("MOV R8, RCX") +
            asm("MOV EAX, 0x%x" % leaf) + asm("MOV ECX, 0x%x" % subleaf) +
            asm("CPUID") +
            asm("MOV DWORD PTR [R8], EAX") +
            asm("MOV DWORD PTR [R8+0x4], EBX") +
            asm("MOV DWORD PTR [R8+0x8], ECX") +
            asm("MOV DWORD PTR [R8+0xC], EDX") +
            asm("POP RBX") + asm("RET"))
    call(code)
    return struct.unpack("<4I", ctypes.string_at(STATE, 16))


def cpu_features():
    out = set()
    _, _, ecx, _ = cpuid(1)
    for name, bit in [("pclmul", 1), ("sse42", 20), ("aes", 25), ("avx", 28)]:
        if (ecx >> bit) & 1:
            out.add(name)
    _, ebx, _, _ = cpuid(7)
    for name, bit in [("avx2", 5), ("sha", 29)]:
        if (ebx >> bit) & 1:
            out.add(name)
    return out


###############################################################################
# Tests
###############################################################################
# - MXCSR accesses: the loaded value is random / the stored one depends on
#   the previous operations
# - MWAIT is privileged
# - COMISS / COMISD legacy models are not bit accurate
# - RCPxS / RSQRTxS: the CPU only returns an approximation (12 bits), the
#   exact value is used in the model
SKIP = re.compile(r"^(V?LDMXCSR|V?STMXCSR|MWAIT|V?COMISS|V?COMISD|"
                  r"V?RCP[PS]S|V?RSQRT[PS]S)$")
REQUIRED = [
    (re.compile(r"^SHA"), "sha"),
    (re.compile(r"^V?AES"), "aes"),
    (re.compile(r"^V?PCLMUL"), "pclmul"),
]


def load_tests(path, features, filt):
    """Return a list of (mnemonic, text, bytes)"""
    tests = []
    if path is None:
        with open(common.VECTORS) as fdesc:
            entries = [(64, test["text"], test["bytes"])
                       for test in json.load(fdesc)]
    else:
        entries = []
        for line in open(path):
            mode, text, hexa = line.rstrip("\n").split("|")
            entries.append((int(mode), text, hexa))
    seen = set()
    for mode, text, hexa in entries:
        name = text.split()[0]
        if mode != 64 or SKIP.match(name):
            continue
        if text in seen:
            # Alternative encoding of the same instruction
            continue
        seen.add(text)
        if filt and not filt.search(text):
            continue
        if any(regexp.match(name) and feature not in features
               for regexp, feature in REQUIRED):
            continue
        if "[RBX]" in text:
            # RBX + 0x10 is the only address aligned on 32 bytes
            continue
        tests.append((name, text, bytes.fromhex(hexa)))
    return tests


def compare(state, got, ref):
    """Return the differences between the miasm result @got and the native
    one @ref"""
    errors = []
    for i in range(16):
        if got["ymm"][i] != ref["ymm"][i]:
            errors.append("YMM%d miasm=%064x native=%064x" % (
                i, got["ymm"][i], ref["ymm"][i]))
    for i in range(8):
        if got["mm"][i] != ref["mm"][i]:
            errors.append("MM%d miasm=%016x native=%016x" % (
                i, got["mm"][i], ref["mm"][i]))
    for reg in state["gpr"]:
        if got["gpr"][reg] != ref["gpr"][reg]:
            errors.append("%s miasm=%x native=%x" % (
                reg, got["gpr"][reg], ref["gpr"][reg]))
    for flag, bit in ARITH_FLAGS.items():
        native = (ref["flags"] >> bit) & 1
        if flag in got["flags"]:
            if got["flags"][flag] != native:
                errors.append("%s miasm=%d native=%d" % (
                    flag, got["flags"][flag], native))
        elif native != (state["flags"] >> bit) & 1:
            errors.append("%s modified by the CPU only" % flag)
    if got["mem"] != ref["mem"]:
        errors.append("MEM miasm=%x native=%x" % (got["mem"], ref["mem"]))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("vectors", nargs="?")
    parser.add_argument("-i", "--iterations", type=int, default=3)
    parser.add_argument("-f", "--filter")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    features = cpu_features()
    print("CPU features:", sorted(features))
    assert "avx2" in features, "AVX2 is required"
    filt = re.compile(args.filter) if args.filter else None
    tests = load_tests(args.vectors, features, filt)

    total = failed = 0
    failures = {}
    references = []
    for name, text, data in tests:
        has_imm = bool(re.search(r", 0x[0-9A-F]+$", text))
        kinds = common.state_kinds(name)
        for iteration in range(args.iterations):
            seed = common.test_seed(text, iteration)
            cur_data, cur_text = data, text
            if has_imm and args.vectors is not None:
                # Vary the immediate
                imm = common.DetRandom(seed ^ 0xabcdef).getrandbits(8)
                cur_data = data[:-1] + bytes([imm])
                cur_text = str(mn_x86.dis(cur_data, 64))
            kind = kinds[iteration % len(kinds)]
            total += 1
            state = common.gen_state(seed, kind, MEM_BASE)
            try:
                got = common.run_miasm(cur_data, state, MEM_BASE)
            except Exception as error:
                failed += 1
                failures.setdefault(name, []).append(
                    "%s: miasm error %s" % (cur_text, str(error)[:200]))
                continue
            ref = run_native(cur_data, state)
            errors = compare(state, got, ref)
            if errors:
                failed += 1
                failures.setdefault(name, []).append("%s (%s, %s): %s" % (
                    cur_text, cur_data.hex(), kind, "; ".join(errors[:2])))
            elif iteration == 0:
                # The reference state is regenerated by the test from the
                # seed; the memory area is then located at another address
                ref_state = common.gen_state(seed, kind)
                ref_out = dict(ref)
                ref_out["gpr"] = dict(ref["gpr"])
                for reg in ("RBX", "RDI"):
                    ref_out["gpr"][reg] = ref_state["gpr"][reg]
                out = common.diff_state(ref_state, ref_out)
                out["flags"] = dict((flag, (ref["flags"] >> bit) & 1)
                                    for flag, bit in ARITH_FLAGS.items())
                references.append({
                    "text": cur_text, "bytes": cur_data.hex(),
                    "seed": seed, "kind": kind, "out": out,
                })
    for name in sorted(failures):
        print("FAIL %s (%d)" % (name, len(failures[name])))
        for line in failures[name][:2]:
            print("    " + line[:500])
    print("%d tests, %d failed (%d mnemonics)" % (total, failed,
                                                   len(failures)))
    if args.output:
        with open(args.output, "w") as fdesc:
            fdesc.write("[\n")
            fdesc.write(",\n".join(json.dumps(ref, sort_keys=True)
                                   for ref in references))
            fdesc.write("\n]\n")
        print("%d references written" % len(references))


if __name__ == "__main__":
    main()
