from ReplayTables._utils.MinMaxHeap import MinMaxHeap

h = MinMaxHeap()
h.add(3, 'a')
h.add(1, 'b')
h.add(2, 'c')
h.add(6, 'd')
h.add(4, 'e')
h.add(8, 'f')

print(h._heap)
print(h.max())
print(h.min())

print(h.pop_min())
print(h._heap)

print(h.pop_max())
print(h._heap)
