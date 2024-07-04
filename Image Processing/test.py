import numpy as np

cumulitive = np.cumsum([1, 2, 3, 4, 5])
print(cumulitive[-1] - cumulitive[1])