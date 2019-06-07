"""
Utilities for dealing with and operating on classical memory registers in Quil.
"""

from pyquil import Program
import pyquil.gates as gates
from pyquil.quilatom import MemoryReference


class MemoryChunk(object):
    """
    A MemoryChunk represents a slice of Quil classical memory. This class wraps MemoryReference in
    pyQuil with the ability to split up a memory reference into multiple sized chunks.
    """
    def __init__(self, mem: MemoryReference, start: int, end: int):
        if mem.declared_size is not None and mem.declared_size < end:
            raise IndexError("bounds would exceed declared size of memory reference")

        self.mem = mem
        self.start = start
        self.end = end

    def __str__(self):
        return "{}[{}:{}]".format(
            self.mem.name, self.start + self.mem.offset, self.end + self.mem.offset
        )

    def __repr__(self):
        return "<MChunk {}[{}:{}]>".format(
            self.mem.name, self.start + self.mem.offset, self.end + self.mem.offset
        )

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start
            end = index.stop
            if start is None:
                start = 0
            if end is None:
                end = len(self)
            start += self.start
            end += self.start

            if start < self.start or end > self.end:
                raise IndexError("out of bounds")
            return MemoryChunk(self.mem, start, end)

        if index < 0 or index >= len(self):
            raise IndexError("out of bounds")
        return self.mem[self.start + index]

    def __iter__(self):
        for index in range(self.start, self.end):
            yield self.mem[index]


def matmul(prog: Program, mat, vec: MemoryChunk, result: MemoryChunk,
           scratch: MemoryChunk):
    """
    Extend a Quil program with instructions to perform a classical matrix multiplication of a fixed
    binary matrix with a vector of bits stored in classical memory.
    """
    m, n = mat.shape
    if len(vec) != n:
        raise ValueError("mat and vec are of incompatible sizes")
    if len(result) != m:
        raise ValueError("mat and result are of incompatible sizes")
    if len(scratch) < 1:
        raise ValueError("scratch buffer is too small")

    for i in range(m):
        prog += gates.MOVE(result[i], 0)
        for j in range(n):
            prog += gates.MOVE(scratch[0], vec[j])
            prog += gates.AND(scratch[0], int(mat[i][j]))
            prog += gates.XOR(result[i], scratch[0])

def string_match(prog: Program, mem: MemoryChunk, vec, output: MemoryChunk, scratch: MemoryChunk):
    """
    Compares a bit string in Quil classical memory to a constant vector. If they are equal, the
    function assigns output to 1, otherwise 0.
    """
    n = len(mem)
    if vec.size != n:
        raise ValueError("length of mem and vec do not match")
    if len(scratch) < 1:
        raise ValueError("scratch buffer is too small")

    prog += gates.MOVE(output[0], 0)
    for i in range(n):
        prog += gates.MOVE(scratch[0], mem[i])
        prog += gates.XOR(scratch[0], int(vec[i]))
        prog += gates.IOR(output[0], scratch[0])
    prog += gates.NOT(output[0])

def conditional_xor(prog: Program, mem: MemoryChunk, vec, flag: MemoryChunk, scratch: MemoryChunk):
    """
    Conditionally bitwise XORs a constant vector to a bit string in Quil classical memory if a
    flag bit is set. If the flag bit is not set, this does not modify the memory.
    """
    n = len(mem)
    if vec.size != n:
        raise ValueError("length of mem and vec do not match")

    for i in range(n):
        prog += gates.MOVE(scratch[0], flag[0])
        prog += gates.AND(scratch[0], int(vec[i]))
        prog += gates.XOR(mem[i], scratch[0])

def majority_vote(prog: Program, inputs: MemoryChunk, output: MemoryReference,
                  scratch_int: MemoryChunk):
    if len(scratch_int) < 2:
        raise ValueError("scratch_int buffer too small")
    if len(inputs) % 2 == 0:
        raise ValueError("inputs length must be odd")

    prog += gates.MOVE(scratch_int[0], 0)
    for bit in inputs:
        prog += gates.CONVERT(scratch_int[1], bit)
        prog += gates.ADD(scratch_int[0], scratch_int[1])

    threshold = (len(inputs) + 1) // 2
    prog += gates.MOVE(scratch_int[1], threshold)
    prog += gates.GE(output, scratch_int[0], scratch_int[1])
