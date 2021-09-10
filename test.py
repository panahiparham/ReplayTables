import numpy as np
from ReplayTables.Table import Table, View

table = Table(6, [
    { 'name': 'obs', 'shape': tuple(), 'dtype': 'float64' },
])

view3 = View(table, 3)
view2 = View(table, 2)

for i in range(15):
    x = i
    table.addTuple((x, ))

    if i % 5 == 4:
        table.endTrajectory()

    s = view3.getAll()
    print('---------------------')
    print(i)
    print(s)
