from pyquil.quilatom import MemoryReference
import unittest

from memory import MemoryChunk

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
