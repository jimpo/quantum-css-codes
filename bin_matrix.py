"""
Utilities for binary numpy matrices.
"""

import numpy as np


def reduced_row_echelon_form(mat):
    """
    Returns a new copy of a binary matrix in reduced row echelon form.
    """
    mat = np.copy(mat)
    m, n = mat.shape

    r = 0
    for c in range(n):
        # Find a row after the first r with a 1 in the c'th column.
        row = next((i for i in range(r, m) if mat[i, c] % 2 == 1), None)
        if row is None:
            continue

        # Ensure row r has a 1 in the c'th column.
        if mat[r, c] % 2 == 0:
            mat[r, :] += mat[row, :]

        # Ensure all other rows have a 0 in the i'th column.
        for i in range(m):
            if i != r and mat[i, c] % 2 == 1:
                mat[i, :] += mat[r, :]

        # Move onto the next row.
        r += 1

    return np.mod(mat, 2)

def vec_to_int(vec):
    """
    Convert a big-endian bit vector to an integer.
    """
    result = 0
    for i in range(vec.size):
        result = (result << 1) + vec[i]
    return result

def int_to_vec(int_repr, n):
    """
    Convert a int to its big-endian bit vector representation.
    """
    vec = np.zeros(n, dtype='int')
    for i in reversed(range(n)):
        vec[i] = int_repr & 1
        int_repr = (int_repr >> 1)
    if int_repr != 0:
        raise ValueError("n is too small")
    return vec

def weight_w_vectors(n, w):
    """
    Generate a sequence of all length n binary vectors with Hamming weight w.
    """
    def helper(vec, w, start):
        n = vec.size
        if w == 0:
            yield np.copy(vec)
        else:
            for i in range(start, n):
                vec[i] = 1
                yield from helper(vec, w - 1, i + 1)
                vec[i] = 0

    vec = np.zeros(n, dtype='int')
    yield from helper(vec, w, 0)
