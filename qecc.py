"""
Interfaces for general Quantum Error Correcting Codes.
"""

import abc
import pyquil.gates as gates
from pyquil.quil import Program
from pyquil.quilatom import QubitPlaceholder
from typing import List

from quil_classical import MemoryChunk


class CodeBlock(object):
    """
    CodeBlock keeps track of a collection of physical qubits encoding a logical qubit using a
    stabiliser code along with the known errors. Each time an error correction is performed, the
    count of known errors may be updated.
    """
    def __init__(self, qubits: List[QubitPlaceholder],
                 x_errors: MemoryChunk, z_errors: MemoryChunk):
        n = len(qubits)
        if len(x_errors) != n:
            raise ValueError("x_errors is of incorrect size")
        if len(z_errors) != n:
            raise ValueError("z_errors is of incorrect size")

        self.n = len(qubits)
        self.qubits = qubits
        self.x_errors = x_errors
        self.z_errors = z_errors

    def reset(self, prog: Program):
        """
        Reset the physical qubits to the |0>^{\otimes n} state and the errors to 0.
        This code block must not be entangled with any other qubits in the system.
        """
        prog += (gates.MEASURE(self.qubits[i], self.x_errors[i]) for i in range(self.n))
        for i in range(self.n):
            prog.if_then(self.x_errors[i], gates.X(self.qubits[i]))
            prog += gates.MOVE(self.x_errors[i], 0)
            prog += gates.MOVE(self.z_errors[i], 0)

class QECC(abc.ABC):
    """
    Abstract Quantum Error Correcting Code.
    """
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def n(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def k(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def t(self):
        raise NotImplementedError()
