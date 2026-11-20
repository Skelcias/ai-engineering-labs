import numpy as np


def ReLU(x):
    return np.maximum(0, x)


def softmax(z):
    e = np.exp(z - z.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def init_params(sizes, rng):  # sizes : [784, 16, 16, 10]
    return [
        (rng.normal(0, np.sqrt(2 / n_in), size=(n_out, n_in)), np.zeros((n_out, 1)))
        for n_in, n_out in zip(sizes[:-1], sizes[1:])
    ]


def forward(x, params):  # x : (n_in, batch)
    a = x
    for W, b in params[:-1]:
        a = ReLU(W @ a + b)
    W, b = params[-1]
    return softmax(W @ a + b)


# On met au propre la première brique d'initialisation d'un MLP via numPy avant de passer a la fonction coût.


# SSE = sum of squared errors, la fonction coût la plus basique, en Machine Learning on prefère MSE = mean square error, la même chose mais divisée par le nombre de valeurs
# le coût ne dépend alors plus de la taille du batch.
def cost(y_pred, y_true):
    return np.sum((y_pred - y_true) ** 2)


if __name__ == "__main__":
    params = init_params([784, 16, 16, 10], np.random.default_rng(seed=42))
    assert [W.shape for W, _ in params] == [(16, 784), (16, 16), (10, 16)]
    assert [b.shape for _, b in params] == [(16, 1), (16, 1), (10, 1)]
    assert sum(W.size + b.size for W, b in params) == 13002

    result = forward(np.linspace(0, 1, 784).reshape(784, 1), params)
    assert result.shape == (10, 1)
    assert np.isclose(result.sum(), 1)  # softmax : les probas somment à 1
    assert np.all(result >= 0)

    y_true = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
    cst = cost(result, y_true)
    assert np.isscalar(cst)
    assert 0 <= cst <= 2  # avec softmax + one-hot, la SSE est bornée par 2
    assert cost(y_true, y_true) == 0  # prédiction parfaite → coût nul
    print(cst)
