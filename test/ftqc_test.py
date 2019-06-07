import numpy as np
import pyquil
import pyquil.gates as gates
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
import unittest

from css_code import CSSCode
import ftqc


class TestFTQC(unittest.TestCase):
    def setUp(self):
        # Parity check matrix for the [7, 4, 3] Hamming code.
        h = np.array([
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ])
        self.steane_7bit = CSSCode(h, h)

    def test_single_X_program(self):
        raw_prog = Program()
        ro = raw_prog.declare('ro', 'BIT', 1)
        raw_prog += gates.X(0)
        raw_prog += gates.X(0)
        raw_prog += gates.X(0)
        raw_prog += gates.MEASURE(0, ro[0])

        new_prog = ftqc.rewrite_program(raw_prog, self.steane_7bit)

        results = self.run_program(new_prog)
        for result in results:
            self.assertEqual(result[0], 1)

    def test_single_Y_program(self):
        raw_prog = Program()
        ro = raw_prog.declare('ro', 'BIT', 1)
        raw_prog += gates.Y(0)
        raw_prog += gates.MEASURE(0, ro[0])

        new_prog = ftqc.rewrite_program(raw_prog, self.steane_7bit)

        results = self.run_program(new_prog)
        for result in results:
            self.assertEqual(result[0], 1)

    def test_single_Z_program(self):
        raw_prog = Program()
        ro = raw_prog.declare('ro', 'BIT', 1)
        raw_prog += gates.Y(0)
        raw_prog += gates.MEASURE(0, ro[0])

        new_prog = ftqc.rewrite_program(raw_prog, self.steane_7bit)

        results = self.run_program(new_prog)
        for result in results:
            self.assertEqual(result[0], 1)

    def test_XXX_program(self):
        raw_prog = Program()
        ro = raw_prog.declare('ro', 'BIT', 1)
        raw_prog += gates.X(0)
        raw_prog += gates.X(0)
        raw_prog += gates.X(0)
        raw_prog += gates.MEASURE(0, ro[0])

        new_prog = ftqc.rewrite_program(raw_prog, self.steane_7bit)

        results = self.run_program(new_prog)
        for result in results:
            self.assertEqual(result[0], 1)

    def test_YZ_program(self):
        raw_prog = Program()
        ro = raw_prog.declare('ro', 'BIT', 1)
        raw_prog += gates.Y(0)
        raw_prog += gates.Z(0)
        raw_prog += gates.MEASURE(0, ro[0])

        new_prog = ftqc.rewrite_program(raw_prog, self.steane_7bit)

        results = self.run_program(new_prog)
        for result in results:
            self.assertEqual(result[0], 1)

    def test_HZH_program(self):
        raw_prog = Program()
        ro = raw_prog.declare('ro', 'BIT', 1)
        raw_prog += gates.H(0)
        raw_prog += gates.Z(0)
        raw_prog += gates.H(0)
        raw_prog += gates.MEASURE(0, ro[0])

        new_prog = ftqc.rewrite_program(raw_prog, self.steane_7bit)

        results = self.run_program(new_prog)
        for result in results:
            self.assertEqual(result[0], 1)

    def test_multiple_measurements_program(self):
        raw_prog = Program()
        ro = raw_prog.declare('ro', 'BIT', 2)
        raw_prog += gates.H(0)
        raw_prog += gates.MEASURE(0, ro[0])
        raw_prog.if_then(ro[0], gates.X(0), Program())
        raw_prog += gates.MEASURE(0, ro[1])

        new_prog = ftqc.rewrite_program(raw_prog, self.steane_7bit)

        results = self.run_program(new_prog)
        for result in results:
            self.assertEqual(result[1], 0)

    @unittest.skip("2 qubits is too slow")
    def test_superdense_coding_program(self):
        self.superdense_coding_program(0, 0)
        self.superdense_coding_program(0, 1)
        self.superdense_coding_program(1, 0)
        self.superdense_coding_program(1, 1)

    def superdense_coding_program(self, bit0: int, bit1: int):
        raw_prog = Program()
        ro = raw_prog.declare('ro', 'BIT', 2)
        # Prepare Bell pair
        raw_prog += gates.H(0)
        raw_prog += gates.CNOT(0, 1)
        # Alice controls qubit 0 and Bob controls 1
        if bit0 == 0 and bit1 == 0:
            pass
        if bit0 == 0 and bit1 == 1:
            raw_prog += gates.X(0)
        if bit0 == 1 and bit1 == 0:
            raw_prog += gates.Z(0)
        if bit0 == 1 and bit1 == 1:
            raw_prog += gates.X(0)
            raw_prog += gates.Z(0)
        # Now Alice sends qubit 0 to Bob
        # Bob rotates from Bell basis to standard basis
        raw_prog += gates.CNOT(0, 1)
        raw_prog += gates.H(0)
        # Measure qubits into Bob's registers
        raw_prog += gates.MEASURE(0, ro[0])
        raw_prog += gates.MEASURE(1, ro[1])

        new_prog = ftqc.rewrite_program(raw_prog, self.steane_7bit)

        results = self.run_program(new_prog)
        for result in results:
            self.assertEqual(result[0], bit0)
            self.assertEqual(result[1], bit1)

    def run_program(self, prog: Program):
        n_qubits = len(prog.get_qubits())
        qvm = pyquil.get_qc("{}q-qvm".format(n_qubits))
        return qvm.run(prog)
