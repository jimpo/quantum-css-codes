import numpy as np
import unittest

import bin_matrix


class TestBinaryMatrix(unittest.TestCase):
    def test_reduced_row_echelon_form(self):
        mat = np.array([
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ], dtype='int')
        expected = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ], dtype='int')
        actual = bin_matrix.reduced_row_echelon_form(mat)
        self.assertTrue(np.array_equal(actual, expected))

    def test_vec_to_int(self):
        vec = np.array([0, 1, 0, 1, 1])
        self.assertEqual(bin_matrix.vec_to_int(vec), 11)

    def test_int_to_vec(self):
        expected = np.array([0, 1, 0, 1, 1])
        self.assertTrue(np.array_equal(bin_matrix.int_to_vec(11, 5), expected))

        with self.assertRaises(ValueError):
            bin_matrix.int_to_vec(11, 3)
