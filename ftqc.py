from typing import Dict, Callable, List, Union

import pyquil.gates as gates
from pyquil.paulis import PauliTerm
import pyquil.quil as quil
from pyquil.quil import Program
from pyquil.quilatom import Qubit, QubitPlaceholder
from pyquil.quilbase import (
    Gate,
)

from qecc import CodeBlock, QECC
from quil_classical import MemoryChunk


def rewrite_program(raw_prog: Program, qecc: QECC, correction_interval: int) -> Program:
    """
    :param: correction_interval The fault tolerant program performs an error correction on each
        code block after this many logical gate applications.
    """
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
    for block in logical_qubits.values():
        qecc.encode_zero(new_prog, block.qubits)

    # Construct ancilla code blocks.
    ancilla_x = [QubitPlaceholder() for _ in range(qecc.n)]
    ancilla_z = [QubitPlaceholder() for _ in range(qecc.n)]

    scratch_size = max(qecc.n, qecc.error_correct_scratch_size)
    raw_scratch = new_prog.declare('scratch', 'BIT', scratch_size)
    scratch = MemoryChunk(raw_scratch, 0, raw_scratch.declared_size)

    gates_until_correction = correction_interval
    for inst in raw_prog.instructions:
        if isinstance(inst, Gate):
            # Apply the logical gate.
            gate_qubits = [logical_qubits[index] for index in _gate_qubits(inst)]
            if qecc.apply_transversal(inst.name, *gate_qubits) is None:
                qecc.apply_universal(inst.name, *gate_qubits)

            # Perform a round of error correction.
            gates_until_correction -= 1
            if gates_until_correction == 0:
                for block in logical_qubits.values():
                    # All error corrections share the same ancilla qubits and classical memory
                    # chunk. This limits parallelism, which significantly reduces fault tolerance.
                    # However, keeping the number of ancilla qubits low is necessary in order to
                    # have any chance of simulating with the QVM.
                    perform_error_correction(new_prog, qecc, block, ancilla_x, ancilla_z, scratch)
                gates_until_correction = correction_interval

        elif isinstance(inst, Measurement):
            pass
        elif isinstance(inst, ResetQubit):
            pass
        elif isinstance(inst, JumpTarget):
            pass
        elif isinstance(inst, JumpConditional):
            pass
        elif isinstance(inst, Jump):
            pass
        elif isinstance(inst, Halt):
            new_prog.append(inst)
        elif isinstance(inst, Wait):
            pass
        elif isinstance(inst, Reset):
            raise NotImplementedError()
        elif isinstance(inst, Pragma):
            new_prog.append(inst)
        elif isinstance(inst, Declare):
            new_prog.append(inst)
        elif any(isinstance(inst, ClassicalInst) for ClassicalInst in CLASSICAL_INSTRUCTIONS):
            new_prog.append(inst)
        else:
            raise UnsupportedProgramError("unsupported instruction: {}", inst)

    return quil.address_qubits(new_prog)

def new_logical_qubit(prog: Program, qecc: QECC, name: str):
    n = qecc.n
    raw_mem = prog.declare(name, 'BIT', 2 * n)
    mem = MemoryChunk(raw_mem, 0, raw_mem.declared_size)
    qubits = [QubitPlaceholder() for _ in range(n)]
    return CodeBlock(qubits, mem[:n], mem[n:])

def _reset_physical_qubits(prog: Program, qubits: List[QubitPlaceholder], scratch: MemoryChunk):
    n = len(qubits)
    assert len(scratch) >= n

    for i in range(n):
        prog += gates.MEASURE(qubits[i], scratch[i])
        prog.if_then(scratch[i], gates.X(qubits[i]))

def _gate_qubits(gate: Gate) -> List[int]:
    return [_extract_qubit_index(qubit) for qubit in gate.qubits]

def _extract_qubit_index(qubit: Union[Qubit, int]) -> int:
    if isinstance(qubit, Qubit):
        return qubit.index
    return qubit

def perform_error_correction(prog: Program, qecc: QECC, block: CodeBlock,
                             ancilla_x: List[QubitPlaceholder], ancilla_z: List[QubitPlaceholder],
                             scratch: MemoryChunk):
    _reset_physical_qubits(prog, ancilla_x, scratch)
    qecc.encode_plus(prog, ancilla_x)

    _reset_physical_qubits(prog, ancilla_z, scratch)
    qecc.encode_zero(prog, ancilla_z)

    qecc.error_correct(prog, block, ancilla_x, ancilla_z, scratch)
