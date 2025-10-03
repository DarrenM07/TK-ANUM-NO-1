import numpy as np

def random_distinct_nodes(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=n)
    x.sort()
    # nudge nodes that are too close
    for i in range(1, n):
        if abs(x[i]-x[i-1]) < 1e-3:
            x[i] += (i+1)*1e-3
    return x
