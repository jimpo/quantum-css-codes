import numpy as np

from errors import InvalidCodeError


class CSSCode(object):
    """
    A Calderbank-Steane-Shor (CSS) code is defined by two binary linear codes C_1, C_2 such that the
    dual code of C_2 is a subspace of C_1. If C_1 is a [n, k_1, d_1] code and C_2 is a [n, k_2, d_2]
    code, then the logical Hilbert space of the CSS code has dimension k_1 + k_2 - n and minimum
    distance min(d_1, d_2).
    """
    def __init__(self, parity_check_c1, parity_check_c2):
        # Validate input codes
        r_1, n_1 = parity_check_c1.shape
        r_2, n_2 = parity_check_c2.shape
        if n_1 != n_2:
            raise ValueError("C_1 and C_2 must have the same code word length")

        h_1 = np.mod(np.array(parity_check_c1, dtype='int'), 2)
        h_2 = np.mod(np.array(parity_check_c2, dtype='int'), 2)
        if not np.array_equal(h_1, parity_check_c1):
            raise ValueError("C_1 parity check matrix must be binary")
        if not np.array_equal(h_2, parity_check_c2):
            raise ValueError("C_2 parity check matrix must be binary")

        # Check that the C_2 dual code is a subspace of C_1.
        prod = np.mod(np.matmul(h_1, np.transpose(h_2)), 2)
        if np.any(prod):
            raise ValueError("C_2 dual code must be a subspace of C_1")

        # Put H_1 and H_2 into standard form. In standard form, H_1 is represented as [I A_1 A_2]
        # where I has width r_1, A_1 has width r_2, and A_2 has width n - r_1 - r_2. H_2 is
        # represented as [D I E], where D has width r_1, I has width r_2, and E has width
        # n - r_1 - r_2.
        h_1, qubit_swaps = normalize_parity_check(h_1, offset=0)
        for indices in qubit_swaps:
            swap_columns(h_2, indices)

        h_2, qubit_swaps = normalize_parity_check(h_2, offset=r_1)
        for indices in qubit_swaps:
            swap_columns(h_1, indices)

        self.parity_check_c1 = h_1
        self.parity_check_c2 = h_2

def swap_columns(mat, indices):
    i, j = indices
    mat[:,i], mat[:,j] = np.array(mat[:,j]), np.array(mat[:,i])

def normalize_parity_check(h, offset):
    r, n = h.shape
    if n < offset + r:
        raise ValueError("not enough columns")

    qubit_swaps = []
    for i in range(r):
        # Find a row after the first i-1 with a 1 in the i'th column past the offset.
        row = next((j for j in range(i, r) if h[j, i + offset] % 2 == 1), None)
        if row is not None:
            # Ensure row j has a 1 in the i'th column.
            if h[i, i + offset] % 2 == 0:
                h[i, :] += h[row, :]
        else:
            # If no remaining rows have 1 in the i'th column path the offset, then swap qubits.
            col = next((j for j in range(i + offset, n) if h[i, j] % 2 == 1), None)
            if col is None:
                raise InvalidCodeError("rows are not independent")

            qubit_swaps.append((i + offset, col))
            swap_columns(h, qubit_swaps[-1])

        # Ensure all other rows have a 0 in the i'th column.
        for j in range(r):
            if i != j and h[j, i + offset] % 2 == 1:
                h[j, :] += h[i, :]

    return np.mod(h, 2), qubit_swaps
