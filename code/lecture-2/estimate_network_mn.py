# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "typer",
# ]
# ///
"""
Run simple model using Michael Nielsen's code (with some tweaks).
Original code:
    https://github.com/mnielsen/neural-networks-and-deep-learning

Move to scripts folder somewhere (if you like).

Usage:
    uv run scripts/estimate_network_mn.py estimate
or move it into environment with packages and run:
    python scripts/estimate_network_mn.py estimate
or copy wherever.

Example run by:
    uv run scripts/estimate_network_mn.py example
"""

import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import typer
from numpy.typing import NDArray
from sklearn.datasets import fetch_openml

parent = Path(__file__).parent
fp_base = parent.parent if parent.name == "scripts" else Path.cwd()
fp_data = fp_base.joinpath("data")
fp_data.mkdir(exist_ok=True)
mnist_file = fp_data / "mnist.npz"

Data = list[tuple[NDArray[np.float64], NDArray[np.uint8]]]


def onehot(y: NDArray[np.uint8], n_classes: int = 10) -> NDArray[np.uint8]:
    return np.eye(n_classes, dtype=np.uint8)[y]


def prepare_nielsen(
    X: NDArray[np.int64],
    y: NDArray[np.uint8],
    y_transform=lambda x: x.reshape(10, 1),
    one_hot: bool = True,
) -> list[tuple[NDArray[np.float64], NDArray[np.uint8]]]:
    """
    Massages data into the shape MN's code expects i.e. list of tuples with
    vectors (781, 1) and (10, 1).
    """
    return [
        (x.reshape(784, 1) / 256, y_transform(v))
        for x, v in zip(X, onehot(y) if one_hot else y)
    ]


def sigmoid(z: NDArray[np.float64]):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z: NDArray[np.float64]):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z: NDArray[np.float64], axis: int = 1) -> NDArray[np.float64]:
    z_shift = z - z.max(axis=axis, keepdims=True)  # numerical stability
    expz = np.exp(z_shift)
    return expz / expz.sum(axis=axis, keepdims=True)


class Network(object):
    """
    The Network class of Michael Nielsen.
    Most of the text in the docstrings are of Nielsen.
    I have replaced sigmoid activation in output layer with softmax
    and cross-entropy as the loss function.
    Also, I have added some type hints for readability.
    """

    def __init__(self, sizes: list[int]):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a: NDArray[np.float64]):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = w @ a + b
            a = sigmoid(z)
        # final layer softmax
        b, w = self.biases[-1], self.weights[-1]
        # softmax along rows while z is (10, 1) [i.e. a column vector]
        a = softmax(w @ a + b, axis=0)
        return a

    def SGD(
        self,
        training_data: Data,
        epochs: int,
        mini_batch_size: int,
        eta: float,
        test_data: Data | None = None,
    ):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                n_test = len(test_data)
                print(
                    "Epoch {0}: {1} / {2}".format(
                        j,
                        self.evaluate(test_data),
                        n_test,
                    )
                )
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch: Data, eta: float):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(
        self, x: NDArray[np.float64], y: NDArray[np.uint8]
    ) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        """backprop implementation of MN.

        Returns tuple with derivatives for weights and biases
        for each layer.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = w @ activation + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # final layer softmax
        b, w = self.biases[-1], self.weights[-1]
        # softmax along rows while z is (10, 1) [i.e. a column vector]
        activation = softmax(z := w @ activation + b, axis=0)
        zs.append(z)
        activations.append(activation)

        # backward pass
        delta = self.loss_deriv(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = delta @ activations[-2].T
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = (self.weights[-l + 1].transpose() @ delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[-l - 1].T
        return (nabla_b, nabla_w)

    def evaluate(self, test_data: Data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def loss_deriv(self, yhat: NDArray[np.float64], y: NDArray[np.uint8]):
        """
        See e.g.
            https://ai-for-humanity-ucph.github.io/2025/slides/lecture-2/#/40.
        """
        return yhat - y


def save_model(net: Any, suffix: str) -> Path:
    outdir = fp_base.joinpath("models")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"model_{suffix}.pkl"
    with open(outfile, "wb") as f:
        pickle.dump(net, f)
    return outfile


def load_model(suffix: str):
    infile = fp_base.joinpath("models") / f"model_{suffix}.pkl"
    with open(infile, "rb") as f:
        return pickle.load(f)


def download_mnist():
    if mnist_file.exists():
        print("Mnist data already exist in data folder")
        return

    mnist = fetch_openml("mnist_784", as_frame=False)
    X = mnist["data"].astype(np.uint8)  # type: ignore
    y = mnist["target"].astype(np.uint8)  # type: ignore

    np.savez_compressed(mnist_file, X=X, y=y)
    print(f"Mnist data saved to {fp_data}")


def load_mnist() -> tuple[NDArray[np.int64], NDArray[np.uint8]]:
    if not mnist_file.exists():
        raise ValueError("Mnist data hasn't been downloaded")
    data = np.load(fp_data / "mnist.npz")
    X, y = data["X"], data["y"]
    return X, y


#  NOTE: Load data here
# Load data:
X, y = load_mnist()
X_train, y_train = X[:50_000], y[:50_000]
X_val, y_val = X[50_000:60_000], y[50_000:60_000]
X_test, y_test = X[60_000:], y[60_000:]  # not used

training_data = prepare_nielsen(X_train, y_train)
test_data = prepare_nielsen(X_test, y_test)
# strictly speaking, MN denotes validation data as test data;
# the `net.evaluate` compares argmax with actual label; hence no one-hot
# encoding.
val_data = prepare_nielsen(
    X_val,
    y_val,
    y_transform=lambda x: x.item(),
    one_hot=False,
)

# Nifty tool; see https://typer.tiangolo.com/
app = typer.Typer()


@app.command()
def example():
    """Forward and backward for single example from lecture.

    I went through it interactively in the shell.
    """

    net = Network([784, 30, 10])

    x_i, y_i = test_data[-1]

    print(x_i.shape, y_i.shape)
    print(f"y: {y_i.argmax()}")

    x = x_i
    y_onehot = y_i

    # Example forward:
    W_1, W_2 = net.weights
    b_1, b_2 = net.biases

    a0 = x  # (784,1)
    z1 = W_1 @ a0 + b_1
    a1 = sigmoid(z1)

    z2 = W_2 @ a1 + b_2
    yhat = softmax(z2, axis=0)

    with np.printoptions(precision=3, suppress=True):
        print(f"yhat:\n{yhat}")
        print(f"y pred label: {yhat.argmax()}")
        print(f"y true label: {y_i.argmax()}")

    # cross-entropy loss
    eps = 1e-9
    loss = -np.sum(y_onehot * np.log(yhat + eps)).item()
    print(f"loss: {loss:.2f}")

    # backward
    delta2 = yhat - y_onehot  # (10,1)
    dW2 = delta2 @ a1.T  # (10,30)
    db2 = delta2  # (10,1)
    delta1 = (W_2.T @ delta2) * sigmoid_prime(z1)  # (30,1)
    dW1 = delta1 @ a0.T  # (30,784)
    db1 = delta1  # (30,1)

    # inspect net.train_step to see full backprop
    (dbs, dWs) = net.backprop(x_i, y_i)

    # above computed equals the one returned by backprop function
    assert np.allclose(dbs[0], db1)
    assert np.allclose(dbs[1], db2)
    assert np.allclose(dWs[0], dW1)
    assert np.allclose(dWs[1], dW2)

    print("Gradients match")

    # Load already trained model and predict on the same

    try:
        trained_net: Network = load_model("nielsen-simple")
    except FileNotFoundError:
        print("Error: Model not estimated")
    else:
        yhat = trained_net.feedforward(x_i)

        with np.printoptions(precision=8, suppress=True):
            print(f"yhat:\n{yhat}")
            print(f"yhat_cls: {yhat.argmax()}")
            print(f"y: {y_i.argmax()}")

        loss = -np.sum(y_onehot * np.log(yhat + eps)).item()
        print(f"loss: {loss}")


@app.command()
def estimate(
    epochs: int = 15,
    mini_batch_size: int = 32,
    lr: float = 3.0,
    suffix: str = "nielsen-simple",
    save_network: bool = True,
):
    """Train network and optionally save model."""

    net = Network([784, 30, 10])

    print(
        f"Estimating basic network {epochs=}, {lr=}, {mini_batch_size=}, {save_network=}, {suffix=}"
    )

    net.SGD(
        training_data,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        eta=lr,
        test_data=val_data,
    )

    # Save model if requested
    if save_network:
        outfile = save_model(net, suffix)
        typer.echo(f"Saved model to {outfile}")


if __name__ == "__main__":
    app()
