import numpy as np
import matplotlib.pyplot as plt
from ReplayTables._utils.SumTree import SumTree

rng = np.random.RandomState(2)

t = SumTree(1000)
t.update(0, list(range(950)), np.linspace(-5, 5, num=950) ** 2)
x = t.sample(rng, 500000)

counts = np.unique(x, return_counts=True)
plt.bar(*counts, width=1)
plt.show()
