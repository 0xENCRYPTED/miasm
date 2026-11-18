#! /usr/bin/env python
"""Path walker (miasm.analysis.pathwalk) on a small obfuscated function"""
from __future__ import print_function

from future.utils import viewitems

from miasm.analysis.machine import Machine
from miasm.analysis.pathwalk import PathWalker, EDGE_OPAQUE, EDGE_COMPUTED, \
    EDGE_TRUE, EDGE_FALSE, EDGE_CALL, EDGE_AFTER_CALL, EDGE_RETURN, \
    END_RETURN, END_SEEN
from miasm.arch.x86.arch import mn_x86
from miasm.core import parse_asm, asmblock
from miasm.core.bin_stream import bin_stream_str
from miasm.core.interval import interval
from miasm.core.locationdb import LocationDB
from miasm.expression.expression import ExprInt
from miasm.expression.proof import PROVEN
from miasm.loader.strpatchwork import StrPatchwork

BASE = 0x140001000

SOURCE = '''
main:
    ; Opaque predicate: x * (x + 1) is always even
    MOV     RCX, RDI
    INC     RCX
    IMUL    RCX, RDI
    TEST    CL, 0x1
    JNZ     dead
    ; Linear MBA, folded by the simplifications: (a | b) + (a & b) - a - b
    MOV     RDX, real
    MOV     R8, RSI
    OR      R8, RDI
    MOV     R9, RSI
    AND     R9, RDI
    ADD     R8, R9
    SUB     R8, RSI
    SUB     R8, RDI
    ADD     RDX, R8
    ; Non linear term, only reachable by the solver: (y * (y + 1)) & 1
    MOV     R8, RSI
    INC     R8
    IMUL    R8, RSI
    AND     R8, 0x1
    ADD     RDX, R8
    JMP     RDX
dead:
    ; Never executed: jumps in the middle of nowhere
    MOV     RAX, 0x1234
    JMP     RAX
real:
    ; Loop driven by a concrete counter
    MOV     ECX, 0x3
loop:
    ADD     RAX, RDI
    DEC     ECX
    JNZ     loop
    ; Jump table in the read only data, concrete index
    MOV     RBX, table
    MOV     RBX, QWORD PTR [RBX + 0x8]
    JMP     RBX
case0:
    MOV     RAX, 0x1111
    RET
case1:
    ; Input dependent condition
    CMP     RDI, 0x5
    JZ      equal
    CALL    helper
    RET
equal:
    XOR     EAX, EAX
    RET
helper:
    SHL     RAX, CL
    RET
table:
.qword case0, case1
'''


def assemble(loc_db):
    source = "\n".join(line for line in SOURCE.splitlines()
                       if not line.strip().startswith(";"))
    asmcfg = parse_asm.parse_txt(mn_x86, 64, source, loc_db)
    loc_db.set_location_offset(loc_db.get_name_location("main"), BASE)
    # Every block has to be near the base address
    patches = asmblock.asm_resolve_final(
        mn_x86, asmcfg, dst_interval=interval([(BASE, BASE + 0x1000)]))
    data = StrPatchwork()
    for offset, raw in viewitems(patches):
        data[offset - BASE] = raw
    return bytes(data)


loc_db = LocationDB()
data = assemble(loc_db)
bin_stream = bin_stream_str(data, base_address=BASE)
machine = Machine("x86_64")


def addr_of(name):
    return loc_db.get_location_offset(loc_db.get_name_location(name))


ADDR = dict((name, addr_of(name)) for name in [
    "main", "dead", "real", "loop", "case0", "case1", "equal", "helper",
    "table"])
print(dict((name, hex(addr)) for name, addr in ADDR.items()))


def walk(**kwargs):
    walker = PathWalker(machine, bin_stream, LocationDB(),
                        read_only=[(BASE, BASE + len(data))], **kwargs)
    return walker.run(BASE)


###############################################################################
# Calls skipped
###############################################################################
result = walk()
print(result.summary())

# The dead code and the other case of the table are never reached
assert ADDR["dead"] not in result.blocks
assert ADDR["case0"] not in result.blocks
for name in ("main", "real", "loop", "case1", "equal"):
    assert ADDR[name] in result.blocks, name
# The helper is not walked: the call is skipped
assert ADDR["helper"] not in result.blocks

# Opaque predicate, proven
assert len(result.opaque_predicates) == 1
predicate = result.opaque_predicates[0]
assert predicate.addr == ADDR["main"] and predicate.status == PROVEN
# JNZ is never taken
assert predicate.value == 0

# MBA target, proven
assert len(result.computed_targets) == 1
target = result.computed_targets[0]
assert target.target == ADDR["real"] and target.status == PROVEN
assert not result.unresolved

kinds = dict(((src, dst), kind) for src, dst, kind in result.edges)
after_opaque = [dst for (src, dst), kind in kinds.items()
                if kind == EDGE_OPAQUE]
assert len(after_opaque) == 1
assert kinds[(after_opaque[0], ADDR["real"])] == EDGE_COMPUTED
# The concrete loop is unrolled, then left
assert (ADDR["loop"], ADDR["loop"]) in kinds
# Input dependent condition: both sides
assert kinds[(ADDR["case1"], ADDR["equal"])] == EDGE_TRUE
false_edges = [dst for (src, dst), kind in kinds.items()
               if src == ADDR["case1"] and kind == EDGE_FALSE]
assert len(false_edges) == 1
assert any(kind == EDGE_AFTER_CALL for kind in kinds.values())
assert len(result.calls) == 1 and not result.calls[0].followed
assert result.calls[0].target == ExprInt(ADDR["helper"], 64)

# Two paths, both ending by the return of the function, with the condition
returns = [end for end in result.ends if end.reason == END_RETURN]
assert len(returns) == 2
assert sorted(len(end.constraints) for end in returns) == [1, 1]
assert sorted(expected for end in returns
              for _, expected in end.constraints) == [0, 1]

# Graph
graph = result.graph()
assert ADDR["real"] in graph.nodes()
assert "digraph" in result.dot()

###############################################################################
# Calls followed
###############################################################################
result = walk(follow_calls=True)
assert ADDR["helper"] in result.blocks
kinds = dict(((src, dst), kind) for src, dst, kind in result.edges)
assert EDGE_CALL in kinds.values() and EDGE_RETURN in kinds.values()
assert len(result.calls) == 1 and result.calls[0].followed
assert len([end for end in result.ends if end.reason == END_RETURN]) == 2
assert not result.unresolved

###############################################################################
# Without proof: the opaque values are only "sampled", and not trusted
###############################################################################
result = walk(prove=False)
assert result.opaque_predicates[0].status != PROVEN
# Both sides of the predicate are walked, and the target is not followed
assert ADDR["dead"] in result.blocks
assert result.unresolved
# ... unless asked
result = walk(prove=False, follow_sampled=True)
assert ADDR["dead"] not in result.blocks
assert ADDR["case1"] in result.blocks
assert not result.unresolved

###############################################################################
# Limits
###############################################################################
result = walk(max_steps=3)
assert result.steps == 3

print("OK")
