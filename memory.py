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
        return "{}[{}:{}]".format(self.mem.name, self.start, self.end)

    def __repr__(self):
        return "<MChunk {}[{}:{}]>".format(self.mem.name, self.start, self.end)

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
