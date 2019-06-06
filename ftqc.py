from typing import Dict, Callable, List, Union

import pyquil.gates as gates
from pyquil.paulis import PauliTerm
import pyquil.quil as quil
from pyquil.quil import Program
from pyquil.quilatom import Label, MemoryReference, Qubit, QubitPlaceholder
from pyquil.quilbase import (
    Gate,
    Measurement,
    ResetQubit,
    DefGate,
    JumpTarget,
    JumpConditional,
    Jump,
    Halt,
    Wait,
    Reset,
    Pragma,
    Declare,
    UnaryClassicalInstruction,
    LogicalBinaryOp,
    ArithmeticBinaryOp,
    ClassicalMove,
    ClassicalExchange,
    ClassicalConvert,
    ClassicalLoad,
    ClassicalStore,
    ClassicalComparison,
)

from qecc import CodeBlock, QECC
from quil_classical import MemoryChunk


def rewrite_program(raw_prog: Program, qecc: QECC) -> Program:
    if qecc.k != 1:
        raise UnsupportedQECCError("code must have k = 1")

    if raw_prog.defined_gates:
        raise UnsupportedProgramError("does not support DEFGATE")

    # Assign indices to qubit placeholders in the raw program.
    raw_prog = quil.address_qubits(raw_prog)

    new_prog = Program()

    logical_qubits = {
        index: new_logical_qubit(new_prog, qecc, "logical_qubit_{}".format(index))
        for index in raw_prog.get_qubits(indices=True)
    }

    # Construct ancilla code blocks.
    ancilla_x = new_logical_qubit(new_prog, qecc, "ancilla_x")
    ancilla_z = new_logical_qubit(new_prog, qecc, "ancilla_y")

    # Classical scratch BIT registers for gates/measurements.
    scratch_size = max(qecc.n, qecc.measure_scratch_size)
    raw_scratch = new_prog.declare('scratch', 'BIT', scratch_size)
    scratch = MemoryChunk(raw_scratch, 0, raw_scratch.declared_size)
    _initialize_memory(new_prog, raw_scratch, ancilla_x.qubits + ancilla_z.qubits)

    # Classical scratch INTEGER registers.
    raw_scratch_int = new_prog.declare('scratch_int', 'INTEGER', 2)
    scratch_int = MemoryChunk(raw_scratch_int, 0, raw_scratch_int.declared_size)
    _initialize_memory(new_prog, raw_scratch_int, ancilla_x.qubits + ancilla_z.qubits)

    perform_error_correction = _make_error_corrector(new_prog, qecc, ancilla_x, ancilla_z)

    # Reset all logical qubits.
    for block in logical_qubits.values():
        qecc.encode_zero(new_prog, block)

    for inst in raw_prog.instructions:
        if isinstance(inst, Gate):
            gate_qubits = [logical_qubits[index] for index in _gate_qubits(inst)]
            qecc.apply_gate(new_prog, inst.name, *gate_qubits)

            # Perform error correction after every logical gate.
            perform_error_correction(logical_qubits.values())
        elif isinstance(inst, Measurement):
            qubit = logical_qubits[_extract_qubit_index(inst.qubit)]
            # This should really use its own ancilla instead of sharing with the error correction,
            # but we need be extremely conservative with the number of qubits.
            for _ in qecc.measure(new_prog, qubit, 0, inst.classical_reg, ancilla_z,
                                  scratch, scratch_int):
                # Since measurements are taken multiple times for redundancy, we need to perform
                # rounds of error correction during the measurement routine.
                perform_error_correction(logical_qubits.values())
        elif isinstance(inst, ResetQubit):
            raise NotImplementedError("this instruction is not in the Quil spec")
        elif isinstance(inst, JumpTarget):
            new_prog.inst(JumpTarget(_mangle_label(inst.label)))
        elif isinstance(inst, JumpConditional):
            new_prog.inst(type(inst)(_mangle_label(inst.target), inst.condition))
        elif isinstance(inst, Jump):
            new_prog.inst(Jump(_mangle_label(inst.target)))
        elif isinstance(inst, Halt):
            new_prog.append(inst)
        elif isinstance(inst, Wait):
            raise NotImplementedError()
        elif isinstance(inst, Reset):
            for block in logical_qubits.values():
                qecc.encode_zero(new_prog, block.qubits)
        elif isinstance(inst, Declare):
            new_prog.inst(inst)
        elif isinstance(inst, Pragma):
            new_prog.inst(inst)
        elif any(isinstance(inst, ClassicalInst) for ClassicalInst in CLASSICAL_INSTRUCTIONS):
            new_prog.inst(inst)
        else:
            raise UnsupportedProgramError("unsupported instruction: {}", inst)

    return quil.address_qubits(new_prog)

def new_logical_qubit(prog: Program, qecc: QECC, name: str) -> CodeBlock:
    n = qecc.n
    raw_mem = prog.declare(name, 'BIT', 2 * n)
    mem = MemoryChunk(raw_mem, 0, raw_mem.declared_size)
    qubits = [QubitPlaceholder() for _ in range(n)]
    _initialize_memory(prog, raw_mem, qubits)
    return CodeBlock(qubits, mem[:n], mem[n:])

def _gate_qubits(gate: Gate) -> List[int]:
    return [_extract_qubit_index(qubit) for qubit in gate.qubits]

def _extract_qubit_index(qubit: Union[Qubit, int]) -> int:
    if isinstance(qubit, Qubit):
        return qubit.index
    return qubit

def _initialize_memory(prog: Program, mem: MemoryReference, qubits: List[QubitPlaceholder]):
    """
    The QVM has some weird behavior where classical memory registers have to be initialized with
    a MEASURE before any values can be MOVE'd to them. So for memory regions used internally,
    initialize memory by performing superfluous measurements.
    """
    prog += (gates.MEASURE(qubits[i % len(qubits)], mem[i]) for i in range(mem.declared_size))
    prog += (gates.MOVE(mem[i], 0) for i in range(mem.declared_size))

def _mangle_label(label: Label) -> Label:
    """Mangle a label to avoid namespace collisions."""
    if isinstance(label, Label):
        return Label("NESTED_{}".format(label.name))
    return label

def _make_error_corrector(
        prog: Program, qecc: QECC, ancilla_x: CodeBlock, ancilla_z: CodeBlock
) -> Callable[[List[CodeBlock]], None]:
    # All error corrections share the same ancilla qubits and classical memory
    # chunk. This limits parallelism, which significantly reduces fault tolerance.
    # However, keeping the number of ancilla qubits low is necessary in order to
    # have any chance of simulating with the QVM.

    # Classical scratch BIT registers for error correction.
    scratch_size = max(qecc.n, qecc.error_correct_scratch_size)
    raw_scratch = prog.declare('error_correct_scratch', 'BIT', scratch_size)
    scratch = MemoryChunk(raw_scratch, 0, raw_scratch.declared_size)
    _initialize_memory(prog, raw_scratch, ancilla_x.qubits + ancilla_z.qubits)

    def perform_error_correction(logical_qubits: List[CodeBlock]):
        for block in logical_qubits:
            qecc.encode_plus(prog, ancilla_x)
            qecc.encode_zero(prog, ancilla_z)
            qecc.error_correct(prog, block, ancilla_x.qubits, ancilla_z.qubits, scratch)

    return perform_error_correction

