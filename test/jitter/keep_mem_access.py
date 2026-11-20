"""keep_mem_access: the memory access log survives the block, so the caller can
read what every instruction read and wrote (one instruction per block)."""
from __future__ import print_function
import sys

from miasm.core.utils import decode_hex
from miasm.analysis.machine import Machine
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE
from miasm.core.locationdb import LocationDB

CODE = 0x40000000
DATA = 0x50000000
STACK = 0x10000000

# mov rax, [rbx]; mov [rbx+0x10], rax; push rax; add rcx, rax; pop rdx; ret
code = decode_hex("488b03" "48894310" "50" "4801c1" "5a" "c3")


def run(keep):
    loc_db = LocationDB()
    jitter = Machine("x86_64").jitter(loc_db, sys.argv[1] if len(sys.argv) > 1 else "llvm")
    jitter.jit.keep_mem_access = keep
    jitter.jit.options["jit_maxline"] = 1
    jitter.jit.options["max_exec_per_call"] = 1
    jitter.vm.add_memory_page(CODE, PAGE_READ | PAGE_WRITE, code + b"\x00" * 0x100, "code")
    jitter.vm.add_memory_page(DATA, PAGE_READ | PAGE_WRITE, b"\x11" * 0x100, "data")
    jitter.vm.add_memory_page(STACK, PAGE_READ | PAGE_WRITE, b"\x00" * 0x1000, "stack")
    jitter.cpu.RSP = STACK + 0x800
    jitter.cpu.RBX = DATA
    jitter.cpu.RCX = 0
    jitter.push_uint64_t(0x1337BEEF)
    trace = []
    state = {"pc": None}

    def flush():
        if state["pc"] is not None:
            trace.append((state["pc"], list(jitter.vm.get_memory_read()), list(jitter.vm.get_memory_write())))
        jitter.vm.reset_memory_access()

    def cb(j):
        flush()
        state["pc"] = j.pc
        return True

    jitter.exec_cb = cb
    jitter.add_breakpoint(0x1337BEEF, lambda j: False)
    jitter.vm.reset_memory_access()
    jitter.run(CODE)
    # (the callback at the return address has flushed the last instruction)
    return jitter, trace


jitter, trace = run(True)
sp = STACK + 0x800 - 8
expected = [
    (CODE + 0, [(DATA, DATA + 8)], []),
    (CODE + 3, [], [(DATA + 0x10, DATA + 0x18)]),
    (CODE + 7, [], [(sp - 8, sp)]),
    (CODE + 8, [], []),
    (CODE + 11, [(sp - 8, sp)], []),
    (CODE + 12, [(sp, sp + 8)], []),
]
for got, exp in zip(trace, expected):
    print("%x read %s write %s" % (got[0], got[1], got[2]))
    assert got == exp, (got, exp)
assert len(trace) == len(expected)
assert jitter.cpu.RDX == 0x1111111111111111
assert jitter.cpu.RCX == 0x1111111111111111

# Default behaviour is unchanged: the log is cleared after each instruction
_, trace = run(False)
assert all(r == [] and w == [] for _, r, w in trace), trace
