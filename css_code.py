import itertools
import numpy as np
from pyquil import Program
import pyquil.gates as gates
from pyquil.paulis import PauliTerm, ID, sX, sY, sZ
from pyquil.quilatom import Qubit, QubitPlaceholder
from pyquil.quilbase import Gate
from typing import List, Union

from errors import InvalidCodeError
import operations


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

        self.n = n_1
        self.r_1 = r_1
        self.r_2 = r_2
        self.parity_check_c1 = h_1
        self.parity_check_c2 = h_2
        t_1, self._c1_syndromes = syndrome_table(h_1)
        t_2, self._c2_syndromes = syndrome_table(h_2)
        self.t = min(t_1, t_2)
        self._transversal_gates = self._determine_transversal_gates(h_1, h_2)

    def stabilisers(self) -> List[PauliTerm]:
        zeros = np.zeros(self.n, dtype='int')
        x_stabilisers = (
            pauli_term_for_row(self.parity_check_c1[i, :], zeros)
            for i in range(self.r_1)
        )
        z_stabilisers = (
            pauli_term_for_row(zeros, self.parity_check_c2[i, :])
            for i in range(self.r_2)
        )
        return list(itertools.chain(x_stabilisers, z_stabilisers))

    def z_operators(self) -> List[PauliTerm]:
        """
        Returns the logical Z operators chosen for this CSS code.

        See Nielsen & Chuang section 10.5.7 for the method of choosing operators.
        """
        n, r_1, r_2 = self.n, self.r_1, self.r_2
        s = n - r_1 - r_2

        # Use the row vector [ 0 0 0 | A2^T 0 I ], which commutes with the check matrix.
        check_mat = np.zeros((s, 2 * n), dtype='int')
        check_mat[:, n:(n + r_1)] = np.transpose(self.parity_check_c1[:, (r_1 + r_2):n])
        check_mat[:, (n + r_1 + r_2):(2 * n)] = np.identity(s)
        return [
            pauli_term_for_row(check_mat[i, :n], check_mat[i, n:])
            for i in range(s)
        ]

    def x_operators(self) -> List[PauliTerm]:
        """
        Returns the logical X operators chosen for this CSS code.

        See Nielsen & Chuang section 10.5.7 for the method of choosing operators.
        """
        n, r_1, r_2 = self.n, self.r_1, self.r_2
        s = n - r_1 - r_2

        # Use the row vector [ 0 E^T I | 0 0 0 ], which commutes with the check matrix.
        check_mat = np.zeros((s, 2 * n), dtype='int')
        check_mat[:, r_1:(r_1 + r_2)] = np.transpose(self.parity_check_c2[:, (r_1 + r_2):n])
        check_mat[:, (r_1 + r_2):n] = np.identity(s)
        return [
            pauli_term_for_row(check_mat[i, :n], check_mat[i, n:])
            for i in range(s)
        ]

    def is_transversal(self, gate_name: str) -> bool:
        """
        Determines whether a quantum gates is known to be fault tolerant with transversal
        application. Transversal application is when the logical gate can by implemented by
        application of the physical gate to each individual qubit.
        """
        return gate_name in self._transversal_gates

    def _determine_transversal_gates(self, parity_check_c1, parity_check_c2):
        # See "Efficient fault-tolerant quantum computing" by Andrew M. Steane for rationales.
        gates = []

        # The identity is clearly transversal.
        gates.append('I')

        # CNOT is always transversal for CSS codes. (Lemma 2, Steane 1998)
        gates.append('CNOT')

        # If C_1 = C_2, then H and CZ are transversal. (Lemma 3, Steane 1998)
        if codes_equal(parity_check_c1, parity_check_c2):
            gates.append('H')
            gates.append('CZ')

            # If C_1 = C_2 and is doubly even, then P transversal. (Lemma 3, Steane 1998)
            if is_doubly_even(parity_check_c1):
                gates.append('PHASE')

        return frozenset(gates)

    def encode(self, qubits: List[Union[QubitPlaceholder, int]]) -> Program:
        n, r_1, r_2 = self.n, self.r_1, self.r_2

        # We are starting with all qubits in the |0> state, meaning they are stabilised by
        # Z_1, Z_2, ..., Z_n. We want to do a unitary transformation to a state stabilised by the
        # code stabilisers along with the stabilisers for the logical 0 state. In general, if a
        # state |ᴪ> is stabilised by S, then U|ᴪ> is stabilised by USU†. We can perform Clifford
        # gate operations to transform the stabiliser set. For details see Nielsen & Chuang
        # section 10.5.2 and Problem 10.3. Also, see Appendix A of "Fault-tolerant Preparation of
        # Stabilizer States for Quantum CSS Codes by ClassicalError-Correcting Codes."
        stabilisers = self.stabilisers() + self.z_operators()

        # The idea is that we want to transform the parity check matrix from
        #
        # [[ 0 0 0 | I1  0  0 ],     [[ I1 A1 A2 | 0    0  0 ],
        #  [ 0 0 0 | 0  I2  0 ],  =>  [  0  0  0 | D   I2  E ],
        #  [ 0 0 0 | 0   0 I3 ]]      [  0  0  0 | A2T  0 I3 ]]
        #
        # Transformations to manipulate the parity check are derived from Figure 10.7 in
        # Nielsen & Chuang which shows how Pauli operators behave under conjugation by Clifford
        # operators.

        # The program accumulates the actual operations on the qubits.
        prog = Program()

        # Step 1: Apply Hadamards to move I1 to the X side. Post-state:
        #
        # [[ I1 0 0 | 0  0  0 ],
        #  [  0 0 0 | 0 I2  0 ],
        #  [  0 0 0 | 0  0 I3 ]]
        for i in range(r_1):
            prog += gates.H(qubits[i])

        # Step 2: Copy Z's from I2 to E. Post-state:
        #
        # [[ I1 0 0 | 0  0  0 ],
        #  [  0 0 0 | 0 I2  E ],
        #  [  0 0 0 | 0  0 I3 ]]
        for i in range(r_1, r_1 + r_2):
            for j in range(r_1 + r_2, n):
                if self.parity_check_c2[i - r_1, j] == 1:
                    prog += gates.CNOT(qubits[j], qubits[i])

        # Step 3: Copy X's from I1 to A and B. This has the side effect of constructing D and A2T.
        # Post-state:
        #
        # [[ I1 A B |   0  0  0 ],
        #  [  0 0 0 |   D I2  E ],
        #  [  0 0 0 | A2T  0 I3 ]]
        for i in range(r_1):
            for j in range(r_1, n):
                if self.parity_check_c1[i, j] == 1:
                    prog += gates.CNOT(qubits[i], qubits[j])

        return prog


def measure(css: CSSCode, data: List[QubitPlaceholder],
            ancilla_x: List[QubitPlaceholder], ancilla_z: List[QubitPlaceholder]) -> Program:
    pass

def syndrome_table(parity_check):
    """
    Given a parity check matrix of a binary linear code, determine the unique decoding threshold t
    and return it along with a lookup table of syndromes to error terms of weight at most t. This
    lookup table can be used to decode noisy codewords.
    """
    _, n = parity_check.shape

    table = {}
    for w in range(n + 1):
        # t_table is a table of syndromes produced by weight-t errors.
        t_table = {}
        for e in weight_w_vectors(n, w):
            syndrome = np.mod(np.matmul(parity_check, e), 2)
            syndrome_int = vec_to_int(syndrome)
            if syndrome_int in table or syndrome_int in t_table:
                return w - 1, table
            t_table[syndrome_int] = e
        # Merge t_table into table
        table = {**table, **t_table}
    return n, table

def vec_to_int(vec):
    """
    Convert a vector of bits to its big-endian int representation.
    """
    result = 0
    for i in range(vec.size):
        result = (result << 1) + vec[i]
    return result

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

def transform_stabilisers(mat, prog):
    _, n = mat.shape

    for inst in prog.instructions:
        if not isinstance(inst, Gate):
            raise ValueError("program must only contain gates")
        if any(not isinstance(qubit, Qubit) for qubit in inst.qubits):
            raise ValueError("gate cannot have placeholders")

        qubits = [qubit.index for qubit in inst.qubits]
        if any(qubit >= n for qubit in qubits):
            raise ValueError("qubit index must be within [0, n)")

        if inst.name == 'H':
            conjugate_h_with_check_mat(mat, *qubits)
        elif inst.name == 'CNOT':
            conjugate_cnot_with_check_mat(mat, *qubits)
        else:
            raise ValueError("cannot conjugate gate {}".format(inst.name))


def conjugate_h_with_check_mat(mat, qubit):
    k, cols = mat.shape
    n = cols // 2
    q = qubit

    for i in range(k):
        if mat[i, q] == 1 and mat[i, n + q] == 1:
            raise NotImplementedError("only handles CSS codes")
        else:
            # H switches X and Z paulis
            mat[i, q], mat[i, n + q] = mat[i, n + q], mat[i, q]

def conjugate_cnot_with_check_mat(mat, control, target):
    k, cols = mat.shape
    n = cols // 2
    c, t = control, target

    for i in range(k):
        # CNOT propagates X paulis from control to target
        if mat[i, c] == 1:
            mat[i, t] = (mat[i, t] + 1) % 2

        # CNOT propagates Z paulis from target to control
        if mat[i, n + t] == 1:
            mat[i, n + c] = (mat[i, n + c] + 1) % 2

def swap_columns(mat, indices):
    i, j = indices
    mat[:,i], mat[:,j] = np.array(mat[:,j]), np.array(mat[:,i])

def pauli_term_for_row(x_check, z_check) -> PauliTerm:
    """
    Determine the Pauli operator from a row in the check matrix.

    See Nielsen & Chuang 10.5.1 for details.
    """
    n = x_check.size
    if not x_check.shape == (n,):
        raise ValueError("x_check has the wrong dimensions")
    if not z_check.shape == (n,):
        raise ValueError("z_check has the wrong dimensions")

    result = ID()
    for i in range(n):
        if x_check[i] and z_check[i]:
            result *= sY(i)
        elif x_check[i]:
            result *= sX(i)
        elif z_check[i]:
            result *= sZ(i)
    return result

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

def codes_equal(parity_check_1, parity_check_2) -> bool:
    if parity_check_1.shape != parity_check_2.shape:
        return False
    return np.array_equal(
        reduced_row_echelon_form(parity_check_1),
        reduced_row_echelon_form(parity_check_2)
    )

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

def is_doubly_even(mat):
    """
    Returns whether even row in the parity check matrix of a binary code is a multiple of 4.
    """
    return not np.any(np.mod(np.sum(mat, axis=1), 4))
