from pyquil import Program
import pyquil.gates as gates

from memory import MemoryChunk


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
        prog += gates.MOVE(scratch[0], flag)
        prog += gates.AND(scratch[0], vec[i])
        prog += gates.XOR(mem[i], scratch[0])

