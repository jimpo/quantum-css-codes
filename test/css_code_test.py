import numpy as np
import unittest

from css_code import CSSCode

class TestCSSCode(unittest.TestCase):
    def test_steane(self):
        # Parity check matrix for the [7, 4, 3] Hamming code.
        h = np.array([
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ])

        steane = CSSCode(h, h)
        self.check_normalized(steane.parity_check_c1, 0)
        self.check_normalized(steane.parity_check_c2, 3)

    def check_normalized(self, parity_check, offset):
        r, n = parity_check.shape
        self.assertTrue(np.array_equal(parity_check[:, offset:offset+r], np.identity(r)))
