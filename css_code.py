import itertools
import numpy as np
import pyquil.gates as gates
from pyquil.paulis import PauliTerm, ID, sX, sY, sZ
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, Qubit, QubitPlaceholder
from pyquil.quilbase import Gate
from typing import List, Union

import bin_matrix
from qecc import CodeBlock, QECC
import quil_classical
from quil_classical import MemoryChunk
from errors import InvalidCodeError, UnsupportedGateError


class CSSCode(QECC):
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

        self._n = n_1
        self._k = n_1 - r_1 - r_2
        self.r_1 = r_1
        self.r_2 = r_2
        self.parity_check_c1 = h_1
        self.parity_check_c2 = h_2
        t_1, self._c1_syndromes = syndrome_table(h_1)
        t_2, self._c2_syndromes = syndrome_table(h_2)
        self._t = min(t_1, t_2)
        self._transversal_gates = self._determine_transversal_gates(h_1, h_2)

        if self.k != 1:
            raise InvalidCodeError("currently only supports CSS codes for a single logical qubit")

    @property
    def n(self):
        return self._n

    @property
    def k(self):
        return self._k

    @property
    def t(self):
        return self._t

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
        Returns the logical Z operators chosen for this CSS code as Pauli terms.
        """
        z_matrix = self.z_operator_matrix()
        zeros = np.zeros_like(z_matrix, dtype='int')
        return [
            pauli_term_for_row(zeros[i, :], z_matrix[i, :])
            for i in range(self.k)
        ]

    def z_operator_matrix(self):
        """
        Returns a check matrix for the logical Z operators chosen for this CSS code.

        See Nielsen & Chuang section 10.5.7 for the method of choosing operators.
        """
        n, r_1, r_2, k = self.n, self.r_1, self.r_2, self.k

        # Use the row vector [ A2^T 0 I ], which commutes with the check matrix.
        check_mat = np.zeros((k, n), dtype='int')
        check_mat[:, 0:r_1] = np.transpose(self.parity_check_c1[:, (r_1 + r_2):n])
        check_mat[:, (r_1 + r_2):n] = np.identity(k)
        return check_mat

    def x_operators(self) -> List[PauliTerm]:
        """
        Returns the logical X operators chosen for this CSS code as Pauli terms.
        """
        x_matrix = self.x_operator_matrix()
        zeros = np.zeros_like(x_matrix, dtype='int')
        return [
            pauli_term_for_row(x_matrix[i, :], zeros[i, :])
            for i in range(self.k)
        ]

    def x_operator_matrix(self):
        """
        Returns a check matrix for the logical X operators chosen for this CSS code.

        See Nielsen & Chuang section 10.5.7 for the method of choosing operators.
        """
        n, r_1, r_2, k = self.n, self.r_1, self.r_2, self.k

        # Use the row vector [ 0 E^T I | 0 0 0 ], which commutes with the check matrix.
        check_mat = np.zeros((k, n), dtype='int')
        check_mat[:, r_1:(r_1 + r_2)] = np.transpose(self.parity_check_c2[:, (r_1 + r_2):n])
        check_mat[:, (r_1 + r_2):n] = np.identity(k)
        return check_mat

    def y_operators(self) -> List[PauliTerm]:
        """
        Returns the logical Y operators chosen for this CSS code as Pauli terms.
        """
        # Y = iXZ
        y_operators = [1j * x_op * z_op
                       for x_op, z_op in zip(self.x_operators(), self.z_operators())]
        for y_op in y_operators:
            assert y_op.coefficient == 1
        return y_operators

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

        # I is always transversal for stabiliser codes.
        gates.append('I')

        # CNOT is always transversal for CSS codes. (Lemma 2, Steane 1998)
        gates.append('CNOT')

        # If C_1 = C_2, then H and CZ are transversal. (Lemma 3, Steane 1998)
        if codes_equal(parity_check_c1, parity_check_c2):
            gates.append('H')
            gates.append('CZ')

            # If C_1 = C_2 and is doubly even, then P transversal. (Lemma 3, Steane 1998)
            if is_doubly_even(parity_check_c1):
                gates.append('S')

        return frozenset(gates)

    def noisy_encode_zero(self, qubits: List[Union[QubitPlaceholder, int]]) -> Program:
        """
        Construct a program preparing a new logical qubit in the |0> state (+1 eigenstate of \hat Z).
        The qubits must all be reset to the physical |0> state at the beginning of this program. The
        preparation is not fault tolerant and any physical errors that occur during preparation may
        create many correlated errors in the code block.
        """
        n, r_1, r_2 = self.n, self.r_1, self.r_2

        # We are starting with all qubits in the |0> state, meaning they are stabilised by
        # Z_1, Z_2, ..., Z_n. We want to do a unitary transformation to a state stabilised by the
        # code stabilisers along with the stabilisers for the logical 0 state. In general, if a
        # state |ᴪ> is stabilised by S, then U|ᴪ> is stabilised by USU†. We can perform Clifford
        # gate operations to transform the stabiliser set. For details see Nielsen & Chuang
        # section 10.5.2 and Problem 10.3. Also, see Appendix A of "Fault-tolerant Preparation of
        # Stabilizer States for Quantum CSS Codes by ClassicalError-Correcting Codes."
        #
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

        # Step 0: Copy Z's from I3 to E. Post-state:
        #
        # [[ I1 0 0 | 0  0  0 ],
        #  [  0 0 0 | 0 I2  E ],
        #  [  0 0 0 | 0  0 I3 ]]
        #
        # This is just a multiplication of stabilisers and does not require any instructions.

        # Step 1: Apply Hadamards to move I1 to the X side. Post-state:
        #
        # [[ I1 0 0 | 0  0  0 ],
        #  [  0 0 0 | 0 I2  E ],
        #  [  0 0 0 | 0  0 I3 ]]
        for i in range(r_1):
            prog += gates.H(qubits[i])

        # Step 2: Copy X's from I1 to A1 and A2. This has the side effect of constructing D and A2T.
        # Post-state:
        #
        # [[ I1 A1 A2 |   0  0  0 ],
        #  [  0  0  0 |   D I2  E ],
        #  [  0  0  0 | A2T  0 I3 ]]
        for i in range(r_1):
            for j in range(r_1, n):
                if self.parity_check_c1[i, j] == 1:
                    prog += gates.CNOT(qubits[i], qubits[j])

        return prog

    def noisy_encode_plus(self, qubits: List[Union[QubitPlaceholder, int]]) -> Program:
        """
        Construct a program preparing a new logical qubit in the |+> state (+1 eigenstate of \hat X).
        The qubits must all be reset to the physical |0> state at the beginning of this program. The
        preparation is not fault tolerant and any physical errors that occur during preparation may
        create many correlated errors in the code block.
        """
        # See implementation of noisy_encode_zero for more detailed comments.
        n, r_1, r_2 = self.n, self.r_1, self.r_2

        # The idea is that we want to transform the parity check matrix from
        #
        # [[ 0 0 0 | I1  0  0 ],     [[ I1 A1 A2 | 0  0  0 ],
        #  [ 0 0 0 | 0  I2  0 ],  =>  [  0  0  0 | D I2  E ],
        #  [ 0 0 0 | 0   0 I3 ]]      [  0 ET I3 | 0  0  0 ]]

        # The program accumulates the actual operations on the qubits.
        prog = Program()

        # Step 1: Apply Hadamards to move I1 and I3 to the X side. Post-state:
        #
        # [[ I1 0  0 | 0  0 0 ],
        #  [  0 0  0 | 0 I2 0 ],
        #  [  0 0 I3 | 0  0 0 ]]
        for i in range(r_1):
            prog += gates.H(qubits[i])
        for i in range(r_1 + r_2, n):
            prog += gates.H(qubits[i])

        # Step 2: Copy Z's from I2 to E. This has the side effect of constructing ET. Post-state:
        #
        # [[ I1  0  0 | 0  0 0 ],
        #  [  0  0  0 | 0 I2 E ],
        #  [  0 ET I3 | 0  0 0 ]]
        for i in range(r_1, r_1 + r_2):
            for j in range(r_1 + r_2, n):
                if self.parity_check_c2[i - r_1, j] == 1:
                    prog += gates.CNOT(qubits[j], qubits[i])

        # Step 3: Copy X's from I1 to A1 and A2. This has the side effect of constructing D.
        # Post-state:
        #
        # [[ I1 A1 A2 | 0  0 0 ],
        #  [  0  0  0 | D I2 E ],
        #  [  0 ET I3 | 0  0 0 ]]
        for i in range(r_1):
            for j in range(r_1, n):
                if self.parity_check_c1[i, j] == 1:
                    prog += gates.CNOT(qubits[i], qubits[j])

        return prog

    def encode_zero(self, prog: Program, block: CodeBlock):
        block.reset(prog)
        # TODO: Make fault tolerant
        prog += self.noisy_encode_zero(block.qubits)

    def encode_plus(self, prog: Program, block: CodeBlock):
        block.reset(prog)
        # TODO: Make fault tolerant
        prog += self.noisy_encode_plus(block.qubits)

    def apply_gate(self, prog: Program, gate_name: str, *blocks: CodeBlock):
        pauli_prog = self._apply_pauli(gate_name, *blocks)
        if pauli_prog is not None:
            prog += pauli_prog
            return

        transversal_prog = self._apply_transversal(gate_name, *blocks)
        if transversal_prog is not None:
            prog += transversal_prog
            return

        universal_prog = self._apply_universal(gate_name, *blocks)
        if universal_prog is not None:
            prog += universal_prog
            return

        raise UnsupportedGateError("logical gate {} not implemented".format(gate_name))

    def _apply_pauli(self, gate_name: str, *blocks: CodeBlock) -> Program:
        if gate_name == 'I':
            return Program()

        operators = None
        if gate_name == 'X':
            operators = self.x_operators()
        if gate_name == 'Y':
            operators = self.y_operators()
        if gate_name == 'Z':
            operators = self.z_operators()

        if operators is None:
            return None

        assert len(blocks) == 1
        block = blocks[0]

        # Only one logical qubit per code block.
        pauli_term = operators[0]
        assert pauli_term.coefficient == 1
        return Program(gates.QUANTUM_GATES[pauli](block.qubits[q]) for q, pauli in pauli_term)

    def _apply_transversal(self, gate_name: str, *blocks: CodeBlock) -> Program:
        """
        Attempt to construct a program implementing the given gate transversally. If that cannot be
        done with this type of code, then return None.
        """
        if not self.is_transversal(gate_name):
            return None

        qubits = (block.qubits for block in blocks)

        if gate_name == 'I':
            return apply_transversally(gates.I, *qubits)
        if gate_name == 'CNOT':
            return apply_transversally(gates.CNOT, *qubits)
        if gate_name == 'H':
            return apply_transversally(gates.H, *qubits)
        if gate_name == 'CZ':
            return apply_transversally(gates.CZ, *qubits)
        if gate_name == 'S':
            return apply_transversally(
                lambda qubit: (gates.Z(qubit), gates.S(qubit)), *qubits
            )
        raise NotImplementedError("transversal {} not implemented".format(gate_name))

    def _apply_universal(self, gate_name, *blocks: List[QubitPlaceholder]) -> Program:
        return None

    def error_correct(self, prog: Program, data: CodeBlock,
                      ancilla_x: List[QubitPlaceholder], ancilla_z: List[QubitPlaceholder],
                      scratch: MemoryChunk):
        """
        Extend a Quil program to perform error correction.
        """
        if data.n != self.n:
            raise ValueError("data code word is of incorrect size")
        if len(ancilla_x) != self.n:
            raise ValueError("ancilla_x code word is of incorrect size")
        if len(ancilla_z) != self.n:
            raise ValueError("ancilla_z code word is of incorrect size")
        if len(scratch) < self.error_correct_scratch_size:
            raise ValueError("scratch buffer is too small")

        # Split up the scratch buffer.
        mem_x = scratch[:self.n]
        scratch_x = scratch[mem_x.end:(mem_x.end + self.r_1 + 2)]
        mem_z = scratch[scratch_x.end:(scratch_x.end + self.n)]
        scratch_z = scratch[mem_z.end:(mem_z.end + self.r_2 + 2)]

        # See section 4.4 of "An Introduction to Quantum Error Correction and Fault-Tolerant Quantum
        # Computation" by Daniel Gottesman for correction circuit.

        # Propagate X errors from data block to ancilla_x block, then measure in the Z basis.
        prog += apply_transversally(gates.CNOT, data.qubits, ancilla_x)

        # Propagate Z errors from data block to ancilla_z block, then measure in the X basis.
        prog += apply_transversally(gates.CNOT, ancilla_z, data.qubits)
        prog += apply_transversally(gates.H, ancilla_z)

        # Perform measurements.
        prog += (gates.MEASURE(ancilla_x[i], mem_x[i]) for i in range(self.n))
        prog += (gates.MEASURE(ancilla_z[i], mem_z[i]) for i in range(self.n))

        # Correct the measured classical codewords.
        quil_classical_correct(prog, mem_x, data.x_errors, scratch_x,
                               self.parity_check_c1, self._c1_syndromes)
        quil_classical_correct(prog, mem_z, data.z_errors, scratch_z,
                               self.parity_check_c2, self._c2_syndromes)

    @property
    def error_correct_scratch_size(self) -> int:
        """
        Returns the minimum size of the scratch buffer required by error_correct.
        """
        return (self.n + self.r_1 + 2) + (self.n + self.r_2 + 2)

    def measure(self, prog: Program, data: CodeBlock, index: int, outcome: MemoryReference,
                ancilla: CodeBlock, scratch: MemoryChunk, scratch_int: MemoryChunk):
        """
        Extend a Quil program to measure the logical qubit in the Z basis. Ancilla must be in a
        logical |0> state.

        Index is the index of the logical qubit within the code block. Currently must be 0.

        This measurement is made fault tolerant by repeating a noisy measurement 2t + 1 times and
        returning a majority vote.

        This yields control after each fault tolerant operation so that a round of error correction
        may be performed globally if required.
        """
        if index != 0:
            raise ValueError("only one logical qubit per code block")
        if data.n != self.n:
            raise ValueError("data code word is of incorrect size")
        if ancilla.n != self.n:
            raise ValueError("ancilla code word is of incorrect size")
        if len(scratch) < self.measure_scratch_size:
            raise ValueError("scratch buffer is too small")
        if len(scratch_int) < 1:
            raise ValueError("scratch_int buffer is too small")

        trials = 2 * self.t + 1

        # Split up the scratch buffer.
        noisy_outcomes = scratch[:trials]
        noisy_scratch = scratch[trials:]

        for i in range(trials):
            self.noisy_measure(prog, data, index, noisy_outcomes[i], ancilla, noisy_scratch)
            yield

        # Because of the stupid thing where the QVM relies on MEASURE to initialize classical
        # registers, do a superfluous measure here of the already trashed ancilla.
        prog += gates.MEASURE(ancilla.qubits[0], outcome)

        quil_classical.majority_vote(prog, noisy_outcomes, outcome, scratch_int)

    @property
    def measure_scratch_size(self) -> int:
        return self.n + self.r_2 + 2 + 2 * self.t + 1

    def noisy_measure(self, prog: Program, data: CodeBlock, index: int, outcome: MemoryReference,
                      ancilla: CodeBlock, scratch: MemoryChunk):
        """
        Extend a Quil program to measure the logical qubit in the Z basis. Ancilla must be in a
        logical |0> state.

        Index is the index of the logical qubit within the code block. Currently must be 0.

        This measurement is not fault tolerant and may fail if any single operation fails.
        """
        n, r_2 = self.n, self.r_2

        if index != 0:
            raise ValueError("only one logical qubit per code block")
        if data.n != n:
            raise ValueError("data code word is of incorrect size")
        if ancilla.n != n:
            raise ValueError("ancilla code word is of incorrect size")
        if len(scratch) < n + r_2 + 2:
            raise ValueError("scratch buffer is too small")

        # Split up the scratch buffer.
        mem = scratch[:n]
        scratch = scratch[n:(n + r_2 + 2)]

        # Reset the ancilla to |0>.
        self.encode_zero(prog, ancilla)

        # Propagate each Z in the operator from data block to ancilla block for each, then measure
        # in the Z basis.
        #
        # This implements the technique from section 3 of
        # "Efficient fault-tolerant quantum computing" by Andrew M. Steane.
        prog += apply_transversally(gates.CNOT, data.qubits, ancilla.qubits)
        prog += (gates.MEASURE(ancilla.qubits[i], mem[i]) for i in range(self.n))

        # Opportunistically correct any X errors.
        quil_classical_correct(prog, mem, data.x_errors, scratch,
                               self.parity_check_c2, self._c2_syndromes)

        # Finally, compute the measurement outcome.
        z_operator = self.z_operator_matrix()[index:(index + 1), :]
        outcome_chunk = MemoryChunk(
            MemoryReference(outcome.name), outcome.offset, outcome.offset + 1
        )
        quil_classical.matmul(prog, z_operator, mem, outcome_chunk, scratch)


def quil_classical_correct(prog: Program, codeword: MemoryChunk, errors: MemoryChunk,
                           scratch: MemoryChunk, parity_check, syndromes):
    """
    Extend a Quil program with instructions to correct a noisy classical codeword. Given a noisy
    codeword and a vector of known existing errors, this attempts to identify the valid codeword
    it decodes to. If such a correction is possible, this program updates the error vector with the
    additional corrections required. If no such correction is possible because the number of errors
    exceeds the unique decoding threshold, the errors vector is left unchanged.
    """
    m, n = parity_check.shape
    if len(codeword) != n:
        raise ValueError("codeword is of incorrect size")
    if len(errors) != n:
        raise ValueError("errors is of incorrect size")
    if len(scratch) < m + 2:
        raise ValueError("scratch buffer is too small")

    # Add in existing known errors to the noisy codeword.
    prog += (gates.XOR(codeword[i], errors[i]) for i in range(n))

    # Compute the syndrome by multiplying by the parity check matrix.
    syndrome = scratch[2:m+2]
    quil_classical.matmul(prog, parity_check, codeword, syndrome, scratch[:2])

    # Revert codeword back to the state without error corrections.
    prog += (gates.XOR(codeword[i], errors[i]) for i in range(n))

    # Find the matching syndrome in the syndrome table and apply the correction.
    for match_syndrome_key, correction in syndromes.items():
        match_syndrome = bin_matrix.int_to_vec(match_syndrome_key, m)

        matches = scratch[1:2]
        quil_classical.string_match(prog, syndrome, match_syndrome, matches, scratch[:1])
        quil_classical.conditional_xor(prog, errors, correction, matches, scratch[:1])

    # Now correct codeword with updated corrections.
    prog += (gates.XOR(codeword[i], errors[i]) for i in range(n))

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
        for e in bin_matrix.weight_w_vectors(n, w):
            syndrome = np.mod(np.matmul(parity_check, e), 2)
            syndrome_int = bin_matrix.vec_to_int(syndrome)
            if syndrome_int in table or syndrome_int in t_table:
                return w - 1, table
            t_table[syndrome_int] = e
        # Merge t_table into table
        table = {**table, **t_table}
    return n, table

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
        bin_matrix.reduced_row_echelon_form(parity_check_1),
        bin_matrix.reduced_row_echelon_form(parity_check_2)
    )

def is_doubly_even(mat):
    """
    Returns whether even row in the parity check matrix of a binary code is a multiple of 4.
    """
    return not np.any(np.mod(np.sum(mat, axis=1), 4))

def apply_transversally(gate, *blocks) -> Program:
    return Program(gate(*qubits) for qubits in zip(*blocks))
