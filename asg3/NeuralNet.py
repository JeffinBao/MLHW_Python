#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################

import numpy as np
import pandas as pd


class NeuralNet:
    def __init__(self, train, test, header=True, h1=4, h2=2, activation='sigmoid'):
        self.activation = activation
        self.test = test
        np.random.seed(1)

        # raw_input = pd.read_csv(train)
        # TODO: Remember to implement the preprocess method
        # train_dataset = self.preprocess(train) # train is the csv file url or local path
        # ncols = len(train_dataset.columns)
        # nrows = len(train_dataset.index)
        # self.X = train_dataset.iloc[:, 0 : (ncols - 1)].values.reshape(nrows, ncols - 1)
        # self.y = train_dataset.iloc[:, (ncols - 1)].values.reshape(nrows, 1)

        self.X, self.y = self.preprocess(train)

        input_layer_size = len(self.X[0])
        output_layer_size = len(self.y.columns)
        # if not isinstance(self.y[0], np.ndarray):
        #     output_layer_size = 1
        # else:
        #     output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((len(self.X), len(self.X[0])))

        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((len(self.X), h1))

        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((len(self.X), h2))

        self.deltaOut = np.zeros((len(self.X), output_layer_size))

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            return self.__sigmoid(x)
        elif activation == "tanh":
            return self.__tanh(x)
        elif activation == "relu":
            return self.__relu(x)

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            return self.__sigmoid_derivative(x)
        elif activation == "tanh":
            return self.__tanh_derivative(x)
        elif activation == "relu":
            return self.__relu_derivative(x)

    def __sigmoid(self, x):
        value = 1 / (1 + np.exp(-x))
        return value

    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __relu(self, x):
        # should return an array, therefore use np.maximum
        value = np.maximum(x, 0)
        return value

    # the input x should be sigmoid function value
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # the input x should be tanh function value
    def __tanh_derivative(self, x):
        return 1 - x ** 2

    def __relu_derivative(self, x):
        # should return an array
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #
    def preprocess(self, path):
        names = []
        data = {}  # must use {}, if use [], means it is list and the index can not be string
        if 'car' in path:
            names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'cls']
            data = pd.read_csv(path, names=names, header=None, na_values='?')
        elif 'iris' in path:
            names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'cls']
            data = pd.read_csv(path, names=names, header=None, na_values='?')
        elif 'adult' in path:
            names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                     'hours-per-week', 'native-country', 'cls']
            data = pd.read_csv(path, names=names, header=None, sep=',\s', na_values='?', engine='python')

        # convert NaN to most frequent attribute value in each column
        for attr in names[:len(names) - 1]:
            value = data[attr].value_counts().idxmax()
            data = data.fillna({attr: value})

        # convert categorical values into numerical values except the 'cls' column
        import sklearn.preprocessing as pp
        enc = {}  # must use {}, if use [], means it is list and the index can not be string
        for column in data.columns[: len(names) - 1]:
            if data.dtypes[column] == np.object:
                enc[column] = pp.LabelEncoder()
                data[column] = enc[column].fit_transform(data[column])

        # get dummy variable of target value(cls), convert categorical values to numerical
        data = pd.get_dummies(data, columns=['cls'], prefix=['cls'])

        # set new header names, since after get_dummies operation, more column are added
        new_names = list(data)
        col_names = new_names[0: len(names) - 1]
        X = data[col_names]
        cls_name = new_names[len(names) - 1:]
        y = data[cls_name]
        # split data into training and test dataset
        from sklearn.model_selection import train_test_split as tt_split
        X_train, X_test, y_train, y_test = tt_split(X, y)

        # standardizing and scaling
        from sklearn.preprocessing import StandardScaler as ss
        scaler = ss()
        scaler.fit(X_train)  # only fit the training data
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # convert y_test DataFrame to numpy.ndarray, and combine attributes and target values
        y_test = y_test.as_matrix()
        test_data = np.concatenate((X_test, y_test), axis=1)
        # ','.join(new_names) is to convert header name list to str and use ',' as delimiter
        self.__processed_data_to_csv(test_data, self.test, ','.join(new_names))

        return X_train, y_train

    # save processed test data into csv file and add header, header should be str type
    # use ',' as the delimiter
    def __processed_data_to_csv(self, array, path, str):
        np.savetxt(path, array, delimiter=',', header=str)

    # Below is the training function
    def train(self, max_iterations=1000, learning_rate=0.05):
        error = []
        for iteration in range(max_iterations):
            out = self.forward_pass(self.activation)
            error = 0.5 * np.power((out - self.y), 2)  # come from the error function, and error is a DataFrame
            self.backward_pass(out, self.activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)  # T means transpose
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            # TODO test len is necessary or not, don't add, since it will decrease the performance
            self.w23 += update_layer2 / len(self.X)
            self.w12 += update_layer1 / len(self.X)
            self.w01 += update_input / len(self.X)

        print("After " + str(max_iterations) + " iterations, the total error is " + str(error.values.sum() / len(self.X)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    # back propagation algorithm -- forward_pass
    def forward_pass(self, activation):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01)
        self.X12 = self.__activation(in1, activation)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__activation(in2, activation)
        in3 = np.dot(self.X23, self.w23)
        out = self.__activation(in3, activation)
        return out

    # back propagation algorithm -- backward_pass
    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            self.deltaOut = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            self.deltaOut = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "relu":
            self.deltaOut = (self.y - out) * (self.__relu_derivative(out))

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            self.delta23 = (self.deltaOut.dot(self.w23.T)) * self.__sigmoid_derivative(self.X23)
        elif activation == "tanh":
            self.delta23 = (self.deltaOut.dot(self.w23.T)) * self.__tanh_derivative(self.X23)
        elif activation == "relu":
            self.delta23 = (self.deltaOut.dot(self.w23.T)) * self.__relu_derivative(self.X23)

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            self.delta12 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            self.delta12 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "relu":
            self.delta12 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

    def compute_input_layer_delta(self, activation="sigmoid"):
        # TODO test whether can write delta_input_layer this way
        # delta_input_layer = np.zeros((len(self.X), len(self.delta01[0])))
        if activation == "sigmoid":
            self.delta01 = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            self.delta01 = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "relu":
            self.delta01 = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))

            # self.delta01 = delta_input_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function
    def predict(self):
        # read test data from file
        test_data = pd.read_csv(self.test)
        # split test data into attributes and target values according to column index
        X_test = test_data[test_data.columns[:len(self.X[0])]]
        y_test = test_data[test_data.columns[len(self.X[0]):]]

        # similar to the forward pass
        in1 = np.dot(X_test, self.w01)
        X_test12 = self.__activation(in1, self.activation)
        in2 = np.dot(X_test12, self.w12)
        X_test23 = self.__activation(in2, self.activation)
        in3 = np.dot(X_test23, self.w23)
        out = self.__activation(in3, self.activation)

        # convert DataFrame to numpy.ndarray and count the quantity of correct prediction
        y_test = y_test.as_matrix()
        count = 0
        for i in range(len(y_test)):
            if y_test[i].argmax() == out[i].argmax():
                count += 1

        accuracy = count / len(y_test)

        return 1 - accuracy


if __name__ == "__main__":
    # car = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
    # iris = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    # adult = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    car = ''
    iris = ''
    adult = ''

    car_test_path = ''
    iris_test_path = ''
    adult_test_path = ''

    neural_network = NeuralNet(adult, adult_test_path, activation='tanh', h1=20, h2=10)
    neural_network.train()
    error = neural_network.predict()
    print('output error for test data is:', error)
    # testError = neural_network.predict("test.csv")
