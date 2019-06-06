import numpy as np
import pyquil
import pyquil.gates as gates
from pyquil.quil import Program
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

    def test_single_x_program(self):
        raw_prog = Program()
        ro = raw_prog.declare('ro', 'BIT', 1)
        raw_prog += (gates.I(0) for _ in range(3))
        raw_prog += gates.X(0)
        raw_prog += gates.MEASURE(0, ro[0])

        new_prog = ftqc.rewrite_program(raw_prog, self.steane_7bit, correction_interval=1)

        qvm = pyquil.get_qc("21q-qvm")
        results = qvm.run(new_prog)[0]
        assert results == [1]
