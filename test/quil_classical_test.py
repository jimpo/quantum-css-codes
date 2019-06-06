import numpy as np
import pyquil
from pyquil import Program
import pyquil.gates as gates
from pyquil.quilatom import MemoryReference
import unittest

import quil_classical
from quil_classical import MemoryChunk

class TestQuilClassical(unittest.TestCase):
    def setUp(self):
        self.qvm = pyquil.get_qc("1q-qvm")

    def test_matmul(self):
        m, n = 20, 10
        mat = np.random.randint(0, 2, size=(m, n), dtype='int')
        vec = np.random.randint(0, 2, size=n, dtype='int')

        prog = Program()
        raw_mem = prog.declare('ro', 'BIT', n + m + 1)
        self.initialize_memory(prog, raw_mem)

        mem = MemoryChunk(raw_mem, 0, n + m + 1)
        vec_in = mem[0:n]
        vec_out = mem[n:(n + m)]
        scratch = mem[(n + m):(n + m + 1)]

        # Copy data from vec into program memory.
        for i in range(n):
            prog += gates.MOVE(vec_in[i], int(vec[i]))

        quil_classical.matmul(prog, mat, vec_in, vec_out, scratch)

        results = self.qvm.run(prog)[0]

        actual = results[n:(n+m)]
        expected = np.mod(np.matmul(mat, vec), 2)
        for i in range(m):
            self.assertEqual(actual[i], int(expected[i]))

    def test_string_match(self):
        test_cases = [
            ([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], True),
            ([0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], True),
            ([0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1], True),
            ([0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], False),
            ([0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1], False),
            ([0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1], False),
        ]

        for vec1, vec2, expected_match in test_cases:
            n = len(vec1)
            prog = Program()
            raw_mem = prog.declare('ro', 'BIT', n + 2)
            self.initialize_memory(prog, raw_mem)

            mem = MemoryChunk(raw_mem, 0, n + 2)
            vec = mem[0:n]
            output = mem[n:(n + 1)]
            scratch = mem[(n + 1):(n + 2)]

            # Copy data from vec2 into vec.
            for i in range(n):
                prog += gates.MOVE(vec[i], vec2[i])

            match_vec = np.array(vec1, dtype='int')
            quil_classical.string_match(prog, vec, match_vec, output, scratch)

            results = self.qvm.run(prog)[0]
            self.assertEqual(results[n] == 1, expected_match)

    def test_majority_vote(self):
        test_cases = [
            ([0, 0, 0], 0),
            ([0, 0, 1], 0),
            ([0, 1, 0], 0),
            ([1, 0, 0], 0),
            ([0, 1, 1], 1),
            ([1, 0, 1], 1),
            ([1, 1, 0], 1),
            ([1, 1, 1], 1),
            ([0, 1, 0, 1, 0], 0),
            ([1, 0, 1, 0, 1], 1),
        ]

        for inputs, expected_output in test_cases:
            prog = Program()
            raw_mem = prog.declare('ro', 'BIT', len(inputs) + 1)
            raw_scratch_int = prog.declare('scratch_int', 'INTEGER', 2)
            self.initialize_memory(prog, raw_mem)
            self.initialize_memory(prog, raw_scratch_int)

            mem = MemoryChunk(raw_mem, 0, raw_mem.declared_size)
            output_mem = mem[0]
            inputs_mem = mem[1:]

            scratch_int = MemoryChunk(raw_scratch_int, 0, raw_scratch_int.declared_size)

            # Copy data from inputs into mem.
            prog += (gates.MOVE(inputs_mem[i], inputs[i]) for i in range(len(inputs)))

            quil_classical.majority_vote(prog, inputs_mem, output_mem, scratch_int)

            results = self.qvm.run(prog)[0]
            self.assertEqual(results[0], expected_output)

    def initialize_memory(self, prog, mem):
        # Need to measure a qubit to initialize memory for some reason.
        for i in range(mem.declared_size):
            prog += gates.MEASURE(0, mem[i])
            prog += gates.MOVE(mem[i], 0)


class TestMemoryChunk(unittest.TestCase):
    def setUp(self):
        self.mem = MemoryReference("test", 0, 20)

    def test_constructor(self):
        chunk = MemoryChunk(self.mem, 10, 20)
        self.assertEqual(chunk.start, 10)
        self.assertEqual(chunk.end, 20)

        with self.assertRaises(IndexError):
            MemoryChunk(self.mem, 0, 21)

    def test_len(self):
        chunk = MemoryChunk(self.mem, 1, 10)
        self.assertEqual(len(chunk), 9)

    def test_getitem_single(self):
        chunk = MemoryChunk(self.mem, 10, 20)

        item = chunk[5]
        self.assertIsInstance(item, MemoryReference)
        self.assertEqual(item, self.mem[15])

    def test_getitem_slice(self):
        chunk = MemoryChunk(self.mem, 10, 20)

        sub_chunk = chunk[2:9]
        self.assertIsInstance(sub_chunk, MemoryChunk)
        self.assertEqual(sub_chunk.start, 12)
        self.assertEqual(sub_chunk.end, 19)

        sub_chunk = chunk[:9]
        self.assertIsInstance(sub_chunk, MemoryChunk)
        self.assertEqual(sub_chunk.start, 10)
        self.assertEqual(sub_chunk.end, 19)

        sub_chunk = chunk[2:]
        self.assertIsInstance(sub_chunk, MemoryChunk)
        self.assertEqual(sub_chunk.start, 12)
        self.assertEqual(sub_chunk.end, 20)
