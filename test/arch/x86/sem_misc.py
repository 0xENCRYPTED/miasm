#! /usr/bin/env python
"""x86 lifter: options removing the internal IR blocks, SMSW / XGETBV, image
backed symbolic execution, stack variables with overlapping slots.

The jitter is not needed.
"""
from __future__ import print_function

import random

from miasm.analysis.data_flow import replace_stack_vars, \
    retrieve_stack_accesses
from miasm.analysis.machine import Machine
from miasm.arch.x86 import regs
from miasm.arch.x86.lifter_model_call import LifterModelCall_x86_64_Win64
from miasm.core.bin_stream import bin_stream_str
from miasm.core.locationdb import LocationDB
from miasm.expression.expression import ExprInt, ExprMem, ExprAssign, \
    ExprId, ExprLoc
from miasm.expression.simplifications import expr_simp
from miasm.ir.ir import IRBlock, AssignBlock
from miasm.ir.symbexec import SymbolicExecutionEngine, \
    SymbolicExecutionEngineImage

random.seed(0)
machine = Machine("x86_64")


def lift(line, **options):
    """Return (lifter, instruction, ircfg) for the assembly @line"""
    loc_db = LocationDB()
    lifter = machine.lifter(loc_db)
    for name, value in options.items():
        setattr(lifter, name, value)
    instr = machine.mn.fromstring(line, loc_db, 64)
    instr = machine.mn.dis(machine.mn.asm(instr)[0], 64)
    instr.offset = 0x1000
    ircfg = lifter.new_ircfg()
    lifter.add_instr_to_ircfg(instr, ircfg)
    return lifter, instr, ircfg


###############################################################################
# SMSW on a 64 bit register, XGETBV
###############################################################################
for line in ["SMSW R8", "SMSW EAX", "SMSW WORD PTR [RAX]", "XGETBV"]:
    lifter, instr, ircfg = lift(line)
    assert len(ircfg.blocks) == 1
_, _, ircfg = lift("XGETBV")
assignblk = list(ircfg.blocks.values())[0][0]
assert regs.RAX in assignblk and regs.RDX in assignblk
assert "x86_xgetbv" in str(assignblk[regs.RAX])

###############################################################################
# Lifter options: no internal IR block
###############################################################################
GPRS = ["RAX", "RBX", "RCX", "RDX", "RSI", "RDI"]
FLAGS = ["zf", "nf", "pf", "of", "cf", "af", "df"]
MEM_BASE = 0x200000
MEM_SIZE = 0x100


def random_state(df=None):
    state = {}
    for name in GPRS:
        value = random.getrandbits(64)
        if name == "RCX" and random.random() < 0.5:
            # Shift counts: null, small, multiple of the size, ...
            value = random.choice([0, 1, 5, 31, 32, 63, 64, 65])
        state[getattr(regs, name)] = ExprInt(value, 64)
    # Pointers in the memory area
    for name in ("RSI", "RDI"):
        state[getattr(regs, name)] = ExprInt(
            MEM_BASE + 0x40 + random.randrange(0x40), 64)
    for name in FLAGS:
        state[getattr(regs, name)] = ExprInt(random.getrandbits(1), 1)
    if df is not None:
        state[regs.df] = ExprInt(df, 1)
    state[ExprMem(ExprInt(MEM_BASE, 64), MEM_SIZE * 8)] = ExprInt(
        random.getrandbits(MEM_SIZE * 8), MEM_SIZE * 8)
    return state


def run(lifter, ircfg, state):
    """Execute every IR block of the instruction, return the final state"""
    symbexec = SymbolicExecutionEngine(lifter, dict(state))
    cur = ExprLoc(lifter.loc_db.get_or_create_offset_location(0x1000), 64)
    steps = 0
    while cur.is_loc():
        irblock = ircfg.get_block(cur.loc_key)
        if irblock is None:
            break
        cur = expr_simp(symbexec.eval_updt_irblock(irblock))
        steps += 1
        assert steps < 1000
    out = {}
    for reg in state:
        value = expr_simp(symbexec.eval_expr(reg))
        assert value.is_int(), "%s = %s" % (reg, value)
        out[reg] = int(value)
    return out


FLATTENED = [
    "SHL RAX, CL", "SHR EAX, CL", "SAR RBX, CL", "SAR BX, CL",
    "ROL RAX, CL", "ROR EBX, CL", "RCL RAX, CL", "RCR EAX, CL",
    "SHLD RAX, RBX, CL", "SHRD EAX, EBX, CL",
    "SHL QWORD PTR [RSI], CL", "ROR DWORD PTR [RDI], CL",
    "CMOVZ RAX, RBX", "CMOVB EAX, EBX", "CMOVG RAX, QWORD PTR [RSI]",
    "MOVSB", "MOVSQ", "STOSB", "STOSD", "LODSB", "LODSQ", "SCASB", "CMPSB",
    "BSF RAX, RBX", "BSR EAX, EBX",
]
for line in FLATTENED:
    lifter_ref, _, ircfg_ref = lift(line)
    lifter_flat, _, ircfg_flat = lift(line, flatten_internal_branches=True)
    assert len(ircfg_ref.blocks) > 1, line
    # A single block, going to the next instruction
    assert len(ircfg_flat.blocks) == 1, line
    irblock = list(ircfg_flat.blocks.values())[0]
    assert irblock.dst.is_loc(), line
    # Same semantic
    for _ in range(30):
        state = random_state()
        assert run(lifter_ref, ircfg_ref, state) == \
            run(lifter_flat, ircfg_flat, state), line
print("flatten_internal_branches: OK")

for line in ["MOVSB", "MOVSQ", "STOSB", "STOSQ", "LODSB", "SCASB", "CMPSB"]:
    lifter_ref, _, ircfg_ref = lift(line)
    lifter_df, _, ircfg_df = lift(line, assume_df_zero=True)
    assert len(ircfg_df.blocks) == 1, line
    irblock = list(ircfg_df.blocks.values())[0]
    assert irblock.dst.is_loc(), line
    # df is not read anymore
    for assignblk in irblock:
        for dst, src in assignblk.items():
            assert regs.df not in src.get_r(mem_read=True), line
    for _ in range(20):
        # Same semantic as the reference when df is clear
        state = random_state(df=0)
        assert run(lifter_ref, ircfg_ref, state) == \
            run(lifter_df, ircfg_df, state), line
# REP: the loop remains, but its direction does not depend on df
_, _, ircfg = lift("REP MOVSB", assume_df_zero=True)
for irblock in ircfg.blocks.values():
    for assignblk in irblock:
        for dst, src in assignblk.items():
            assert regs.df not in src.get_r(mem_read=True)
print("assume_df_zero: OK")

# The symbolic execution goes through a variable shift
loc_db = LocationDB()
lifter = machine.lifter(loc_db)
lifter.flatten_internal_branches = True
symbexec = SymbolicExecutionEngine(lifter)
offset = 0x1000
for line in ["SHL RAX, CL", "ROR RBX, CL", "ADD RAX, RBX"]:
    instr = machine.mn.fromstring(line, loc_db, 64)
    instr = machine.mn.dis(machine.mn.asm(instr)[0], 64)
    instr.offset = offset
    offset += instr.l
    ircfg = lifter.new_ircfg()
    lifter.add_instr_to_ircfg(instr, ircfg)
    dst = symbexec.run_block_at(ircfg, instr.offset)
    assert dst == ExprInt(offset, 64), "%s: %s" % (line, dst)

###############################################################################
# Symbolic execution reading the image
###############################################################################
data = bytes(bytearray(range(64)))
bin_stream = bin_stream_str(data, base_address=0x400000)
lifter = machine.lifter(LocationDB())
symbexec = SymbolicExecutionEngineImage(
    lifter, bin_stream, read_only=[(0x400000, 0x400020)])


def read(addr, size):
    return symbexec.eval_expr(ExprMem(ExprInt(addr, 64), size))


assert read(0x400004, 32) == ExprInt(0x07060504, 32)
assert read(0x400000, 8) == ExprInt(0, 8)
# Writable area: unknown
assert read(0x400024, 32) == ExprMem(ExprInt(0x400024, 64), 32)
# Access crossing the limit of the read only area
assert read(0x40001e, 32) == ExprMem(ExprInt(0x40001e, 64), 32)
# Out of the image
assert read(0x500000, 32) == ExprMem(ExprInt(0x500000, 64), 32)
# Symbolic pointer
pointer = ExprMem(regs.RAX, 32)
assert symbexec.eval_expr(pointer) == pointer
# A write shadows the image
symbexec.eval_updt_expr(ExprAssign(ExprMem(ExprInt(0x400004, 64), 16),
                                   ExprInt(0xAAAA, 16)))
assert read(0x400004, 32) == ExprInt(0x0706AAAA, 32)
assert read(0x400000, 64) == ExprInt(0x0706AAAA03020100, 64)
# Every address is read only by default, a function can be given as well
symbexec = SymbolicExecutionEngineImage(lifter, bin_stream)
assert read(0x400024, 16) == ExprInt(0x2524, 16)
symbexec = SymbolicExecutionEngineImage(
    lifter, bin_stream, read_only=lambda addr, size: addr >= 0x400030)
assert read(0x400030, 8) == ExprInt(0x30, 8)
assert read(0x400000, 8) == ExprMem(ExprInt(0x400000, 64), 8)
print("SymbolicExecutionEngineImage: OK")

###############################################################################
# Stack variables: overlapping slots
###############################################################################
loc_db = LocationDB()
lifter = LifterModelCall_x86_64_Win64(loc_db)
RSP = regs.RSP


def slot(offset, size):
    return ExprMem(RSP + ExprInt(offset, 64), size)


loc_key = loc_db.add_location()
assignblks = [
    # 128 bit access, then 16 bit access in the same area
    AssignBlock([ExprAssign(slot(-0x40, 128), regs.XMM0)]),
    AssignBlock([ExprAssign(regs.AX, slot(-0x3c, 16))]),
    # Independent slots
    AssignBlock([ExprAssign(slot(-0x60, 64), regs.RBX)]),
    AssignBlock([ExprAssign(regs.RCX, slot(-0x60, 64))]),
    AssignBlock([ExprAssign(slot(-0x70, 32), regs.EDX)]),
    AssignBlock([ExprAssign(lifter.IRDst, ExprInt(0x1000, 64))]),
]
ircfg = lifter.new_ircfg()
ircfg.add_irblock(IRBlock(loc_db, loc_key, assignblks))
info = retrieve_stack_accesses(lifter, ircfg)
# The overlapping slots are skipped, the other ones are variables
assert set(expr_simp(addr - RSP) for addr in info) == set(
    [ExprInt(-0x60, 64), ExprInt(-0x70, 64)])
assert replace_stack_vars(lifter, ircfg)
irblock = ircfg.blocks[loc_key]
assert irblock[0][slot(-0x40, 128)] == regs.XMM0
assert irblock[1][regs.AX] == slot(-0x3c, 16)
assert irblock[3][regs.RCX].is_id()
assert slot(-0x60, 64) not in irblock[2]
print("replace_stack_vars: OK")

###############################################################################
# Win64 call model
###############################################################################
target = ExprInt(0x140001000, 64)
assignblks, extra = lifter.call_effects(target, None)
assert not extra and len(assignblks) == 1
effects = assignblks[0]
ret = effects[regs.RAX]
assert ret.is_op("call_func_ret")
assert list(ret.args) == [target, RSP, regs.RCX, regs.RDX, regs.R8, regs.R9]
# The stack pointer is preserved, the volatile registers are not
assert RSP not in effects
for name in ["RCX", "RDX", "R8", "R9", "R10", "R11"]:
    assert getattr(regs, name) in effects
for name in ["RBX", "RSI", "RDI", "R12"]:
    assert getattr(regs, name) not in effects
print("Win64 call model: OK")
