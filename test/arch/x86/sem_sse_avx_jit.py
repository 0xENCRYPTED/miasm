#! /usr/bin/env python
"""Replay the SSE / AVX reference states (sem_sse_avx.json, recorded on a real
CPU) through the jitter.

Usage: sem_sse_avx_jit.py [jit engine] [regexp]
"""
from __future__ import print_function

import json
import re
import sys

from miasm.analysis.machine import Machine
from miasm.core.locationdb import LocationDB
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE

import sem_sse_avx as common
from sem_sse_avx import MEM_BASE, MEM_SIZE, ARITH_FLAGS

CODE_BASE = 0x40000000
CODE_STEP = 0x20


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "python"
    filt = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
    with open(common.VECTORS) as fdesc:
        tests = json.load(fdesc)
    if filt:
        tests = [test for test in tests if filt.search(test["text"])]

    loc_db = LocationDB()
    jitter = Machine("x86_64").jitter(loc_db, engine)
    jitter.vm.add_memory_page(MEM_BASE, PAGE_READ | PAGE_WRITE,
                              b"\x00" * MEM_SIZE, "data")
    jitter.vm.add_memory_page(CODE_BASE, PAGE_READ | PAGE_WRITE,
                              b"\x90" * (CODE_STEP * (len(tests) + 1)),
                              "code")
    cpu = jitter.cpu
    mask128 = (1 << 128) - 1

    failed = 0
    failures = {}
    for index, test in enumerate(tests):
        data = bytes(bytearray.fromhex(test["bytes"]))
        state = common.gen_state(test["seed"], test["kind"])
        # Each test has its own address: no stale translation
        addr = CODE_BASE + index * CODE_STEP
        jitter.vm.set_mem(addr, data)
        jitter.vm.set_mem(MEM_BASE, state["mem"].to_bytes(MEM_SIZE, "little"))
        for i in range(16):
            setattr(cpu, "XMM%d" % i, state["ymm"][i] & mask128)
            setattr(cpu, "YMM%d_H" % i, state["ymm"][i] >> 128)
        for i in range(8):
            setattr(cpu, "MM%d" % i, state["mm"][i])
        for name, value in state["gpr"].items():
            setattr(cpu, name, value)
        for name, bit in ARITH_FLAGS.items():
            setattr(cpu, name, (state["flags"] >> bit) & 1)

        stop = addr + len(data)
        jitter.add_breakpoint(stop, lambda jit: False)
        errors = []
        try:
            jitter.init_run(addr)
            jitter.continue_run()
        except Exception as error:
            errors.append("%s: %s" % (type(error).__name__,
                                      str(error).split("\n")[0][:150]))
        jitter.remove_breakpoints_by_address(stop)
        if errors:
            # Leave a clean state for the next test
            jitter.cpu.set_exception(0)
            jitter.vm.set_exception(0)

        if not errors:
            result = {
                "ymm": [getattr(cpu, "XMM%d" % i) |
                        (getattr(cpu, "YMM%d_H" % i) << 128)
                        for i in range(16)],
                "mm": [getattr(cpu, "MM%d" % i) for i in range(8)],
                "gpr": dict((name, getattr(cpu, name))
                            for name in state["gpr"]),
                "mem": int.from_bytes(jitter.vm.get_mem(MEM_BASE, MEM_SIZE),
                                      "little"),
            }
            expected = test["out"]
            got = common.diff_state(state, result)
            for key in ("ymm", "mm", "gpr"):
                if got.get(key, {}) != expected.get(key, {}):
                    errors.append("%s: got %r, expected %r" % (
                        key, got.get(key, {}), expected.get(key, {})))
            if got.get("mem") != expected.get("mem"):
                errors.append("mem differs")
            if jitter.pc != stop:
                errors.append("pc = %x" % jitter.pc)
        if errors:
            failed += 1
            name = test["text"].split()[0]
            failures.setdefault(name, []).append(
                "%s (%s): %s" % (test["text"], test["bytes"], errors[0]))
    for name in sorted(failures):
        print("FAIL %s (%d)" % (name, len(failures[name])))
        print("    " + failures[name][0][:330])
    print("%s: %d tests, %d failed (%d mnemonics)" % (
        engine, len(tests), failed, len(failures)))
    assert tests
    assert failed == 0


if __name__ == "__main__":
    main()
