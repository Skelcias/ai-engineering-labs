import numpy as np

N_IN = 784


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


# Initialisation de He pour avoir une variance des activations au fil des couches qui reste à peu près la même ni explosion ni exctinction. 2/sqrt(n_entrées)
rng = np.random.default_rng(seed=42)
a0 = np.linspace(0, 1, N_IN).reshape(N_IN, 1)
W1 = rng.normal(loc=0, scale=np.sqrt(2 / N_IN), size=(16, N_IN))
W2 = rng.normal(loc=0, scale=np.sqrt(2 / W1.shape[0]), size=(16, W1.shape[0]))
W3 = rng.normal(loc=0, scale=np.sqrt(2 / W2.shape[0]), size=(10, W2.shape[0]))
b1 = np.zeros(shape=(16, 1))
b2 = np.zeros(shape=(16, 1))
b3 = np.zeros(shape=(10, 1))

z1 = W1 @ a0 + b1
assert z1.shape == (16, 1)
a1 = ReLU(z1)

z2 = W2 @ a1 + b2
assert z2.shape == (16, 1)
a2 = ReLU(z2)
z3 = W3 @ a2 + b3
assert z3.shape == (10, 1)

assert W1.size + W2.size + W3.size + b1.size + b2.size + b3.size == 13002
print(z3.shape)
print(z3)
