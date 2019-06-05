from pyquil import Program
import pyquil.gates as gates
from pyquil.quilatom import QubitPlaceholder
from typing import List


def apply_transversal(gate, *blocks) -> Program:
    return Program(gate(*qubits) for qubits in zip(*blocks))

def transversal_cnot(control_block: List[QubitPlaceholder],
                     target_block: List[QubitPlaceholder]) -> Program:
    return apply_transversal(gates.CNOT, control_block, target_block)

def transversal_hadamard(block: List[QubitPlaceholder]) -> Program:
    return apply_transversal(gates.H, block)

def transversal_phase(control_block: List[QubitPlaceholder],
                     target_block: List[QubitPlaceholder]) -> Program:
    return apply_transversal(gates.CNOT, control_block, target_block)
