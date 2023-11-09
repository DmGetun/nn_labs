import numpy as np
from csv import reader
from PIL import Image

class Perceptron():

    def load_dataset(self, x_train, y_train, x_test, y_test):
        self.x_train = self.__flatten(self.__normalize(x_train))
        self.y_train = self.__one_hot(y_train)
        self.x_test = self.__flatten(self.__normalize(x_test))
        self.y_test = self.__one_hot(y_test)
        self.weights = None
        self.weights2 = None
        self.log_weights = []

    def __normalize(self, pixel):
        return np.round(pixel / 255)

    def __one_hot(self, arr):
        y = np.zeros((arr.size, arr.max() + 1))
        y[np.arange(arr.size), arr] = 1
        return y.astype(np.uint8)
    
    def __flatten(self, df):
        arrays = []
        for arr in df:
            new_array = np.append(arr.flatten(), 1)
            arrays.append(new_array)
        return np.array(arrays)


    def init_weights(self, class_count = 2, hidden_size = 20, seed = 20):
        np.random.seed(seed)
        weights = np.random.rand(785, hidden_size) * 0.1
        self.weights = weights

        biases1 = np.zeros((1, hidden_size))
        self.biases1 = biases1

        weights2 = np.random.rand(hidden_size, class_count) * 0.1
        self.weights2 = weights2

        biases2 = np.zeros((1, class_count))
        self.biases2 = biases2

        return [weights, weights2]

    def __sigmoid(self, v):
        return 1 / (1 + np.exp(-v))
    
    def __sigmoid_derivative(self, v):
        return v * (1 - v)
    
    def __softmax(self, x):
        x = np.atleast_2d(x)  # Это преобразует x к 2D массиву, если он еще не является таковым
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    def __cross_entropy_loss(self, y_pred, y_true):
        n_samples = y_true.shape[0]
        loss = -np.sum(np.log(y_pred) * y_true) / n_samples
        return loss
    
    def get_log_weights(self):
        return self.log_weights

    def __fit(self, learning_rate=0.1, epoch=1):
        y_pred = np.array([])
        for i in range(len(self.x_train)):
        # Прямое распространение
            hidden_out = self.__sigmoid(np.dot(self.x_train[i], self.weights) + self.biases1)  # выход скрытого слоя
            last_out = self.__softmax(np.dot(hidden_out, self.weights2) + self.biases2)  # выход выходного слоя
            y_pred = np.append(y_pred, last_out)
            # Ошибка и градиенты
            last_error = last_out - self.y_train[i]  # ошибка выходного слоя
            hidden_error = np.dot(last_error, self.weights2.T)
            hidden_delta = self.__sigmoid_derivative(hidden_out) * hidden_error

            # Обновление весов и смещений
            self.weights -= learning_rate * np.outer(self.x_train[i], hidden_delta)
            self.biases1 -= learning_rate * hidden_delta
            self.weights2 -= learning_rate * np.outer(hidden_out, last_error) # веса для выходного слоя
            self.biases2 -= learning_rate * last_error

            cross_entropy = self.__cross_entropy_loss(last_out, self.y_train[i])

        cross_entropy = self.__cross_entropy_loss(last_out, self.y_train)
        print(f'Эпоха: {epoch + 1}, Entropy: {cross_entropy}')
        
        return [self.weights, self.weights2, cross_entropy]
    
    def fit(self, learning_rate=0.1, iterations_number = 30, cross_entopy_value=None):
        cross_entropy = 1
        epoch = 1
        self.log_weights.append([self.weights.copy(), self.weights2.copy()])
        if cross_entopy_value is None:
            for i in range(iterations_number):
                w1,w2, _ = self.__fit(learning_rate, i)
                self.log_weights.append([w1, w2])
        else:
            while cross_entopy_value < cross_entropy:
                w1, w2, cross_entropy = self.__fit(learning_rate, epoch)
                self.log_weights.append([w1, w2])
                epoch += 1

        return w1, w2
    
    def predict(self, class_count):
        confusion_matrix_test = np.zeros((class_count, class_count))
        correct_predictions = 0
        
        for i in range(len(self.x_test)):
            z1 = np.dot(self.x_test[i], self.weights) + self.biases1
            a1 = self.__sigmoid(z1)
            z2 = np.dot(a1, self.weights2) + self.biases2
            predicted_class = np.argmax(self.__softmax(z2))
            actual_class = np.argmax(self.y_test[i])

            if predicted_class == actual_class:
                correct_predictions += 1

            confusion_matrix_test[actual_class, predicted_class] += 1

        return (correct_predictions / len(self.x_test) * 100), confusion_matrix_test

