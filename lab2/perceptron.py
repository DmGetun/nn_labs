import numpy as np

class Perceptron():

    def __init__(self, seed):
        self.seed = seed

    def load_dataset(self, x_train, y_train, x_test, y_test):
        self.x_train = self.__flatten(self.__normalize(x_train))
        self.y_train = self.__one_hot(y_train)
        self.x_test = self.__flatten(self.__normalize(x_test))
        self.y_test = self.__one_hot(y_test)
        self.weights = None
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


    def init_weights(self, in_count=784, class_count = 2):
        np.random.seed(self.seed)
        weights = np.random.random((in_count, class_count)) * 0.1
        bs = np.random.rand(class_count)
        weights = np.vstack([weights, bs])
        self.weights = weights
        return weights

    def __sigmoid(self, v):
        return 1 / (1 + np.exp(-v))
    
    def __predict(self, arr, weights):
        return self.__sigmoid(np.dot(arr, weights))


    def get_log_weights(self):
        return self.log_weights
    
    def fit(self, learning_rate=0.1, iterations_number = None, max_accuracy=None):
        
        if iterations_number is None and max_accuracy is None:
            raise 'Укажите гиперпараметры для модели'
        
        accuracy = 0
        epoch = 1
        self.log_weights.append(self.weights.copy())
        if max_accuracy is None:
            for i in range(iterations_number):
                w1, _ = self.__fit(learning_rate, i)
                self.log_weights.append(w1.copy())
                epoch += 1
        else:
            while accuracy < max_accuracy:
                w1, cross_entropy = self.__fit(learning_rate, epoch)
                epoch += 1

        return w1, _

    def __fit(self, learning_rate=0.1, iterations_number = None, max_accuracy=None):
        accuracy = 0
        for i in range(len(self.x_train)):
            self.log_weights.append(self.weights)
            y_out = self.__sigmoid(self.x_train[i] @ self.weights)
            errors = self.y_train[i] - y_out
            self.weights += learning_rate * np.outer(self.x_train[i], errors)
        print(f'Эпоха: {iterations_number+1}')
        
        return self.weights, accuracy

    def predict(self, class_count):
        confusion_matrix_test = np.zeros((class_count, class_count))
        correct_predictions = 0
        for i in range(len(self.x_test)):
            test = self.x_test[i]
            predicted_class = np.argmax(self.__predict(test, self.weights))
            actual_class = np.argmax(self.y_test[i])

            if predicted_class == actual_class:
                correct_predictions += 1

            confusion_matrix_test[actual_class, predicted_class] += 1

        return round((correct_predictions / len(self.x_test) * 100), 2), confusion_matrix_test