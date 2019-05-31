import numpy as np
from pyquil.paulis import PauliTerm, ID, sX, sY, sZ
import unittest

import css_code
from css_code import CSSCode

class TestCSSCode(unittest.TestCase):
    def setUp(self):
        # Parity check matrix for the [7, 4, 3] Hamming code.
        h = np.array([
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ])
        self.steane_7bit = CSSCode(h, h)

    def test_steane(self):
        self.check_normalized(self.steane_7bit.parity_check_c1, 0)
        self.check_normalized(self.steane_7bit.parity_check_c2, 3)

    def check_normalized(self, parity_check, offset):
        r, n = parity_check.shape
        self.assertTrue(np.array_equal(parity_check[:, offset:offset+r], np.identity(r)))

    def test_stabilisers(self):
        expected = [
            sX(0) * sX(3) * sX(4) * sX(5),
            sX(1) * sX(3) * sX(5) * sX(6),
            sX(2) * sX(4) * sX(5) * sX(6),
            sZ(0) * sZ(2) * sZ(3) * sZ(6),
            sZ(0) * sZ(1) * sZ(4) * sZ(6),
            sZ(0) * sZ(1) * sZ(2) * sZ(5),
        ]
        self.assertEqual(self.steane_7bit.stabilisers(), expected)

    def test_z_operators(self):
        expected = [
            sZ(1) * sZ(2) * sZ(6),
        ]
        self.assertEqual(self.steane_7bit.z_operators(), expected)

    def test_x_operators(self):
        expected = [
            sZ(1) * sZ(2) * sZ(6),
        ]
        self.assertEqual(self.steane_7bit.z_operators(), expected)

    def test_encode(self):
        n = 7
        prog = self.steane_7bit.encode(range(n))

        # Matrix stabilised by Z_1, ..., Z_n
        mat = np.concatenate(
            (np.zeros((n, n), dtype='int'), np.identity(n, dtype='int')),
            axis=1
        )
        css_code.transform_stabilisers(mat, prog)

        expected_mat = np.zeros((n, 2 * n), dtype='int')
        expected_mat[0:3, 0:7] = self.steane_7bit.parity_check_c1
        expected_mat[3:6, 7:14] = self.steane_7bit.parity_check_c2
        expected_mat[6, 7:10] = np.transpose(self.steane_7bit.parity_check_c1[:, 6:7])
        expected_mat[6, 13:14] = np.identity(1, dtype='int')

        self.assertTrue(np.array_equal(mat, expected_mat))