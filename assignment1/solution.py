import pickle
import numpy as np
import gzip

class NN(object):
    def __init__(self,
                 hidden_dims=(784, 256),
                 datapath='mnist.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=500,
                 seed=None,
                 activation="relu",
                 init_method="glorot"
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        self.train, self.valid, self.test = self.load_mnist()

    def load_mnist(self):
        data_file = gzip.open("mnist.pkl.gz", "rb")
        train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
        data_file.close()

        train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
        train_results = [self.vectorized_result(y) for y in train_data[1]]
        train_data = np.array(train_inputs).reshape(-1, 784), np.array(train_results).reshape(-1, 10)

        val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
        val_results = [self.vectorized_result(y) for y in val_data[1]]
        val_data = np.array(val_inputs).reshape(-1, 784), np.array(val_results).reshape(-1, 10)

        test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
        test_data = list(zip(test_inputs, test_data[1]))

        return train_data, val_data, test_data

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            pass
        # WRITE CODE HERE
        pass
        return 0

    def sigmoid(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            pass
        # WRITE CODE HERE
        pass
        return 0

    def tanh(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            pass
        # WRITE CODE HERE
        pass
        return 0

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            # WRITE CODE HERE
            pass
        elif self.activation_str == "sigmoid":
            # WRITE CODE HERE
            pass
        elif self.activation_str == "tanh":
            # WRITE CODE HERE
            pass
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        # WRITE CODE HERE
        pass
        return 0

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        # WRITE CODE HERE
        pass
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        # WRITE CODE HERE
        pass
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            pass

    def one_hot(self, y, n_classes=None):
        n_classes = n_classes or self.n_classes
        return np.eye(n_classes)[y]

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        # WRITE CODE HERE
        pass
        return 0

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                # WRITE CODE HERE
                pass

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        # WRITE CODE HERE
        pass
        return 0
