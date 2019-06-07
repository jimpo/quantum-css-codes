import numpy as np
import pyquil
import pyquil.gates as gates
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
import time
from typing import Callable, List
import unittest

from css_code import CSSCode
import ftqc


class TestFidelity(unittest.TestCase):
    def setUp(self):
        # Parity check matrix for the [7, 4, 3] Hamming code.
        h = np.array([
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ])
        self.steane_7bit = CSSCode(h, h)

    def test_X_fidelity(self):
        raw_prog = Program()
        ro = raw_prog.declare('ro', 'BIT', 1)
        raw_prog += gates.X(0)
        raw_prog += gates.X(0)
        raw_prog += gates.X(0)
        raw_prog += gates.MEASURE(0, ro[0])

        ft_prog = ftqc.rewrite_program(raw_prog, self.steane_7bit)

        is_correct = lambda result: result[0] == 1

        trials = 1000
        correct, elapsed = self.run_and_benchmark_program(raw_prog, trials, is_correct,
                                                          separate=False)
        print(correct, trials, elapsed)

        trials = 10
        correct, elapsed = self.run_and_benchmark_program(ft_prog, trials, is_correct,
                                                          separate=True)
        print(correct, trials, elapsed)

    def run_and_benchmark_program(self, prog: Program, trials: int,
                                  is_correct: Callable[[List[int]], bool], separate: bool) -> int:
        n_qubits = len(prog.get_qubits())
        qvm = pyquil.get_qc("{}q-qvm".format(n_qubits), noisy=True)

        if separate:
            elapsed = []
            results = []
            for i in range(trials):
                start_time = time.time()
                trial_results = qvm.run(prog)
                end_time = time.time()

                elapsed.append(end_time - start_time)
                results.extend(trial_results)
        else:
            prog.wrap_in_numshots_loop(trials)
            start_time = time.time()
            results = qvm.run(prog)
            end_time = time.time()
            elapsed = end_time - start_time

        correct = sum(is_correct(result) for result in results)
        return correct, elapsed
