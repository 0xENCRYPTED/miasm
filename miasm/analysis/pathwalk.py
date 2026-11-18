"""Path walker: recover the control flow graph of obfuscated code by symbolic
execution.

The code is executed symbolically block by block. At the end of a block, the
destination is classified:
- constant: the walk goes on;
- condition: if it is proven (or sampled) to be constant, it is an opaque
  predicate and only one side is followed, else the state is forked;
- computed value (indirect jump, MBA target): it is simplified, then proven
  constant if possible;
- anything else is reported as unresolved.

A (address, call context, state fingerprint) key avoids to walk again an
already seen situation. With the default fingerprint (the concrete values of
the registers), a loop driven by concrete values is unrolled (dispatcher of a
virtual machine), while a loop driven by symbolic values is walked once.

Example:
    walker = PathWalker(machine, cont.bin_stream, loc_db,
                        read_only=[(text_start, rdata_end)])
    result = walker.run(entry)
    print(result.summary())
    open("cfg.dot", "w").write(result.dot())
"""

from collections import namedtuple

from miasm.core.asmblock import AsmBlockBad
from miasm.core.graph import DiGraph
from miasm.expression.expression import ExprInt, ExprId, ExprCond, \
    ExprAssign
from miasm.expression.proof import find_constant, sample, PROVEN, SAMPLED
from miasm.expression.simplifications import expr_simp, expr_simp_bits
from miasm.ir.symbexec import SymbolicExecutionEngineImage

# Kinds of edges
EDGE_DIRECT = "direct"
EDGE_TRUE = "true"
EDGE_FALSE = "false"
EDGE_OPAQUE = "opaque"
EDGE_COMPUTED = "computed"
EDGE_CALL = "call"
EDGE_RETURN = "return"
EDGE_AFTER_CALL = "after_call"

# Classification of a condition
COND_OPAQUE = "opaque"
COND_OPAQUE_SAMPLED = "opaque (sampled)"
COND_INPUT = "input dependent"

# Reasons ending a path
END_RETURN = "return"
END_UNRESOLVED = "unresolved destination"
END_BAD_CODE = "cannot disassemble"
END_SEEN = "already seen"
END_LIMIT = "limit reached"
END_INTERNAL = "symbolic internal branch"
END_ERROR = "lifting error"

OpaquePredicate = namedtuple(
    "OpaquePredicate", ["addr", "condition", "value", "status"])
ComputedTarget = namedtuple(
    "ComputedTarget", ["addr", "expr", "target", "status"])
Unresolved = namedtuple("Unresolved", ["addr", "expr", "reason"])
Call = namedtuple("Call", ["addr", "target", "followed"])
PathEnd = namedtuple("PathEnd", ["addr", "reason", "state", "constraints"])

_Work = namedtuple("_Work", ["addr", "state", "context", "constraints"])


class WalkResult(object):
    """Result of PathWalker.run
    @blocks: {address: AsmBlock}
    @edges: set of (source address, destination address, kind)
    @opaque_predicates, @computed_targets, @unresolved, @calls, @ends: lists
    of OpaquePredicate, ComputedTarget, Unresolved, Call, PathEnd
    @steps: number of executed blocks
    """

    def __init__(self):
        self.blocks = {}
        self.edges = set()
        self.opaque_predicates = []
        self.computed_targets = []
        self.unresolved = []
        self.calls = []
        self.ends = []
        self.steps = 0

    def successors(self, addr):
        return sorted(set(dst for src, dst, _ in self.edges if src == addr))

    def graph(self):
        """Return the control flow graph as a DiGraph of addresses"""
        graph = DiGraph()
        for addr in self.blocks:
            graph.add_node(addr)
        for src, dst, _ in self.edges:
            graph.add_uniq_edge(src, dst)
        return graph

    def dot(self):
        """Return the control flow graph in the Graphviz format"""
        out = ["digraph cfg {", '  node [shape=box fontname="monospace"];']
        nodes = set(self.blocks)
        for src, dst, _ in self.edges:
            nodes.update([src, dst])
        for addr in sorted(nodes):
            block = self.blocks.get(addr)
            if block is None:
                label = "%X" % addr
            else:
                label = "\\l".join(
                    ["%X" % addr] + [str(line) for line in block.lines]
                ) + "\\l"
            out.append('  n%X [label="%s"];' % (addr,
                                               label.replace('"', "'")))
        for src, dst, kind in sorted(self.edges):
            out.append('  n%X -> n%X [label="%s"];' % (src, dst, kind))
        out.append("}")
        return "\n".join(out)

    def summary(self):
        lines = [
            "%d blocks, %d edges, %d executed blocks" % (
                len(self.blocks), len(self.edges), self.steps),
            "%d opaque predicates" % len(self.opaque_predicates),
            "%d computed targets" % len(self.computed_targets),
            "%d unresolved destinations" % len(self.unresolved),
            "%d calls" % len(self.calls),
        ]
        reasons = {}
        for end in self.ends:
            reasons[end.reason] = reasons.get(end.reason, 0) + 1
        for reason in sorted(reasons):
            lines.append("paths ended by '%s': %d" % (reason,
                                                      reasons[reason]))
        return "\n".join(lines)


class PathWalker(object):
    """Symbolic walk of the paths of a function / of a piece of code"""

    def __init__(self, machine, bin_stream, loc_db, lifter=None,
                 read_only=None, follow_calls=False, prove=True,
                 follow_sampled=False, samples=64, timeout=5000,
                 use_bit_simplifications=True, fingerprint="registers",
                 max_steps=10000, max_visits=64, max_internal_steps=100000):
        """
        @machine: Machine instance
        @bin_stream: content of the binary (Container.bin_stream)
        @lifter: lifter to use. By default, a new one without internal
        branches (see flatten_internal_branches / assume_df_zero for x86)
        @read_only: addresses whose content never changes, see
        SymbolicExecutionEngineImage
        @follow_calls: True / False, or a function (target address or None,
        address of the call) -> bool. A call which is not followed is
        skipped with skip_call
        @prove: use the solver to prove the opaque predicates and the
        computed targets
        @follow_sampled: consider as constant a value which is only constant
        on the samples (not proven). Unsound, but often needed
        @samples: number of random inputs
        @timeout: time allowed to the solver for each proof (ms)
        @use_bit_simplifications: simplify the destinations with
        expr_simp_bits before sampling
        @fingerprint: part of the state used in the "already seen" key:
        "registers" (concrete registers), None, or a function
        (state symbols dictionary) -> hashable
        @max_steps: maximum number of executed blocks
        @max_visits: maximum number of visits of an address in a given call
        context
        @max_internal_steps: maximum number of IR blocks executed inside an
        instruction block (REP prefix with a concrete counter, ...)
        """
        self.machine = machine
        self.bin_stream = bin_stream
        self.loc_db = loc_db
        if lifter is None:
            lifter = machine.lifter(loc_db)
            # x86: no internal IR block
            if hasattr(lifter, "flatten_internal_branches"):
                lifter.flatten_internal_branches = True
                lifter.assume_df_zero = True
        self.lifter = lifter
        self.mdis = machine.dis_engine(bin_stream, loc_db=loc_db)
        self.mdis.follow_call = False
        self.read_only = read_only
        self.follow_calls = follow_calls
        self.prove = prove
        self.follow_sampled = follow_sampled
        self.samples = samples
        self.timeout = timeout
        self.use_bit_simplifications = use_bit_simplifications
        self.fingerprint = fingerprint
        self.max_steps = max_steps
        self.max_visits = max_visits
        self.max_internal_steps = max_internal_steps
        self._fresh = 0

    # Overridable parts

    def new_symbexec(self, state):
        """Return the symbolic execution engine used for a block"""
        is_snapshot = state is not None and not isinstance(state, dict)
        symbexec = SymbolicExecutionEngineImage(
            self.lifter, self.bin_stream,
            state=None if is_snapshot else state,
            read_only=self.read_only
        )
        if is_snapshot:
            symbexec.set_state(state)
        return symbexec

    def simplify(self, expr):
        """Simplify a destination / a condition"""
        expr = expr_simp(expr)
        if self.use_bit_simplifications and not expr.is_int():
            expr = expr_simp_bits(expr)
        return expr

    def is_return(self, instr):
        """True if @instr returns from a function"""
        return instr.name.upper().startswith("RET")

    def fresh_symbol(self, name, size):
        """Return a new unknown value"""
        self._fresh += 1
        return ExprId("%s_%d" % (name, self._fresh), size)

    def skip_call(self, symbexec, instr, target):
        """Model a call which is not followed: the callee returns to the
        next instruction, its result is unknown.
        The default implementation handles the architectures pushing the
        return address on the stack, and only clobbers the result register:
        override it for an other calling convention.
        Return the address where the execution goes on"""
        sp = self.lifter.sp
        size = sp.size
        # Pop the return address pushed by the call
        symbexec.eval_updt_expr(
            ExprAssign(sp, symbexec.eval_expr(sp) + ExprInt(size // 8, size)))
        ret_reg = self.result_register()
        if ret_reg is not None:
            symbexec.eval_updt_expr(
                ExprAssign(ret_reg, self.fresh_symbol("ret", ret_reg.size)))
        return instr.offset + instr.l

    def result_register(self):
        """Register holding the result of a call (x86 by default)"""
        regs = self.machine.mn.regs
        for name in ("RAX", "EAX", "AX"):
            reg = getattr(regs, name, None)
            if reg is not None and reg.size == self.lifter.sp.size:
                return reg
        return None

    def state_fingerprint(self, symbols):
        """Part of the state identifying an already seen situation"""
        if self.fingerprint is None:
            return None
        if callable(self.fingerprint):
            return self.fingerprint(symbols)
        out = []
        for dst, src in symbols.items():
            if dst.is_id() and src.is_int():
                out.append((dst.name, int(src)))
        return tuple(sorted(out))

    # Walk

    def _to_addr(self, expr):
        """Return the address designated by @expr, or None"""
        if expr.is_int():
            return int(expr)
        if expr.is_loc():
            return self.loc_db.get_location_offset(expr.loc_key)
        return None

    def _constant(self, expr):
        """Sample, then prove if enabled: return (ExprInt or None, status)"""
        if self.prove:
            result = find_constant(expr, samples=self.samples,
                                   timeout=self.timeout)
            if result.status in (PROVEN, SAMPLED):
                return result.value, result.status
            return None, result.status
        values = set(value for _, value in sample(expr, self.samples))
        if len(values) == 1 and None not in values:
            return ExprInt(values.pop(), expr.size), SAMPLED
        return None, None

    def _execute(self, work, result):
        """Execute the block of @work. Return (symbexec, destination, last
        instruction) or None if the path ends"""
        addr = work.addr
        block = result.blocks.get(addr)
        if block is None:
            try:
                block = self.mdis.dis_block(addr)
            except Exception:
                block = None
            if block is None or isinstance(block, AsmBlockBad) or \
                    not block.lines:
                result.ends.append(PathEnd(addr, END_BAD_CODE, work.state,
                                           work.constraints))
                return None
            result.blocks[addr] = block
        ircfg = self.lifter.new_ircfg()
        try:
            self.lifter.add_asmblock_to_ircfg(block, ircfg)
        except NotImplementedError as error:
            result.unresolved.append(Unresolved(addr, None, str(error)))
            result.ends.append(PathEnd(addr, END_ERROR, work.state,
                                       work.constraints))
            return None
        symbexec = self.new_symbexec(work.state)
        dst = symbexec.run_block_at(ircfg, addr)
        steps = 0
        # Internal IR blocks (REP prefix, ...)
        while True:
            dst = expr_simp(dst)
            if not (dst.is_loc() and
                    self.loc_db.get_location_offset(dst.loc_key) is None):
                break
            if ircfg.get_block(dst.loc_key) is None:
                break
            steps += 1
            if steps > self.max_internal_steps:
                result.ends.append(PathEnd(addr, END_LIMIT, work.state,
                                           work.constraints))
                return None
            dst = symbexec.run_block_at(ircfg, dst.loc_key)
        if self._has_internal_loc(dst):
            result.unresolved.append(Unresolved(addr, dst, END_INTERNAL))
            result.ends.append(PathEnd(addr, END_INTERNAL, work.state,
                                       work.constraints))
            return None
        return symbexec, dst, block.lines[-1]

    def _has_internal_loc(self, expr):
        todo = [expr]
        while todo:
            node = todo.pop()
            if node.is_loc():
                if self.loc_db.get_location_offset(node.loc_key) is None:
                    return True
            elif node.is_cond():
                todo += [node.src1, node.src2]
        return False

    def _destinations(self, addr, dst, result):
        """Classify the destination @dst of the block at @addr.
        Return a list of (target address, kind, constraint or None), and
        report the unresolved ones"""
        dst = self.simplify(dst)
        target = self._to_addr(dst)
        if target is not None:
            return [(target, EDGE_DIRECT, None)]
        if dst.is_cond():
            cond = dst.cond
            # Is the condition always true / always false?
            boolean = ExprCond(cond, ExprInt(1, 1), ExprInt(0, 1))
            value, status = self._constant(boolean)
            if value is not None and (status == PROVEN or
                                      self.follow_sampled):
                result.opaque_predicates.append(
                    OpaquePredicate(addr, cond, int(value), status))
                taken = dst.src1 if int(value) else dst.src2
                out = []
                for target, kind, constraint in self._destinations(
                        addr, taken, result):
                    if kind == EDGE_DIRECT:
                        kind = EDGE_OPAQUE
                    out.append((target, kind, constraint))
                return out
            if value is not None:
                # Constant on the samples only: reported, both sides walked
                result.opaque_predicates.append(
                    OpaquePredicate(addr, cond, int(value), status))
            out = []
            for branch, kind, expected in ((dst.src1, EDGE_TRUE, 1),
                                           (dst.src2, EDGE_FALSE, 0)):
                for target, sub_kind, constraint in self._destinations(
                        addr, branch, result):
                    if sub_kind != EDGE_DIRECT:
                        kind = sub_kind
                    out.append((target, kind, (cond, expected)))
            return out
        # Computed destination
        value, status = self._constant(dst)
        if value is not None and (status == PROVEN or self.follow_sampled):
            result.computed_targets.append(
                ComputedTarget(addr, dst, int(value), status))
            return [(int(value), EDGE_COMPUTED, None)]
        if value is not None:
            result.computed_targets.append(
                ComputedTarget(addr, dst, int(value), status))
        result.unresolved.append(Unresolved(addr, dst, END_UNRESOLVED))
        return []

    def _follow(self, target, addr):
        if callable(self.follow_calls):
            return bool(self.follow_calls(target, addr))
        return bool(self.follow_calls) and target is not None

    def run(self, entry, init_state=None):
        """Walk the paths starting at the address @entry
        @init_state: initial symbolic state ({Expr: Expr} or SymbolicState),
        every register being symbolic by default
        Return a WalkResult"""
        result = WalkResult()
        symbexec = self.new_symbexec(init_state)
        todo = [_Work(entry, symbexec.get_state(), (), ())]
        seen = set()
        visits = {}
        while todo:
            work = todo.pop()
            addr = work.addr
            symbols = dict(work.state.symbols)
            key = (addr, work.context, self.state_fingerprint(symbols))
            if key in seen:
                result.ends.append(PathEnd(addr, END_SEEN, work.state,
                                           work.constraints))
                continue
            seen.add(key)
            count = visits.get((addr, work.context), 0) + 1
            visits[(addr, work.context)] = count
            if count > self.max_visits or result.steps >= self.max_steps:
                result.ends.append(PathEnd(addr, END_LIMIT, work.state,
                                           work.constraints))
                continue
            result.steps += 1
            executed = self._execute(work, result)
            if executed is None:
                continue
            symbexec, dst, last = executed
            context = work.context

            if last.is_subcall():
                target = self._to_addr(self.simplify(dst))
                followed = self._follow(target, addr)
                result.calls.append(Call(addr, self.simplify(dst), followed))
                after = last.offset + last.l
                if followed and target is not None:
                    result.edges.add((addr, target, EDGE_CALL))
                    todo.append(_Work(target, symbexec.get_state(),
                                      context + (after,), work.constraints))
                else:
                    next_addr = self.skip_call(symbexec, last, target)
                    result.edges.add((addr, next_addr, EDGE_AFTER_CALL))
                    todo.append(_Work(next_addr, symbexec.get_state(),
                                      context, work.constraints))
                continue

            if self.is_return(last):
                target = self._to_addr(self.simplify(dst))
                if target is not None and context and context[-1] == target:
                    result.edges.add((addr, target, EDGE_RETURN))
                    todo.append(_Work(target, symbexec.get_state(),
                                      context[:-1], work.constraints))
                    continue
                if target is None:
                    # Return of the walked function
                    result.ends.append(PathEnd(addr, END_RETURN,
                                               symbexec.get_state(),
                                               work.constraints))
                    continue
                # Return to a computed address (push / ret, ...)

            destinations = self._destinations(addr, dst, result)
            if not destinations:
                result.ends.append(PathEnd(addr, END_UNRESOLVED,
                                           symbexec.get_state(),
                                           work.constraints))
                continue
            state = symbexec.get_state()
            for target, kind, constraint in destinations:
                result.edges.add((addr, target, kind))
                constraints = work.constraints
                if constraint is not None:
                    constraints = constraints + (constraint,)
                todo.append(_Work(target, state, context, constraints))
        return result
