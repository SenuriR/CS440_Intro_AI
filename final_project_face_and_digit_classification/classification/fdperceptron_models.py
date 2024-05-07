import numpy as np


#     Face perceptron class for binary classification tasks, specifically
#     designed to recognize faces or non-faces in images. It uses a standard perceptron
#     learning algorithm with added regularization and early stopping functionality
#     to prevent over-fitting
class FDPerceptronFace:

    # Initializes the FacePerceptron instance with random weights and
    # a specified learning rate (step size alpha at each iteration to update the weights)
    def __init__(self, num_features, rate):
        self.lr = rate
        # Weights array has one additional element for the bias.
        self.weights = np.random.standard_normal(num_features + 1)

    # STEP FUNCTION
    def fdstep(self, x):
        return np.where(x > 0, 1, 0)

        # TRAINING WITH FUNCTIONS

    #  Function train, parameters:
    #         - training_data (array): The input features for training.
    #         - training_label (array): The expected labels (targets) for training data.
    #         - validation_input (array): The input features for validation.
    #         - validation_label (array): The expected labels for validation data.
    #         - iterations (int): The number of iterations to train over the training dataset.
    #         - fdlambda (float): The regularization strength to avoid over-fitting.
    #
    #   In general, regularization is most effective when the training data is limited or when the model
    #   has a high complexity, such as a deep neural network with many parameters. In these cases,
    #   the model is more likely to overfit, and regularization can help to prevent this by encouraging
    #   the model to learn only the most important patterns in the data.
    def train(self, training_data, training_label, validation_input, validation_label, iterations, fdlambda):

        best = 0
        max_iter = 10
        counter = 0  # iteration counter to reach 'max_iter'

        if training_data.ndim > 2:
            training_data = training_data.reshape(training_data.shape[0], -1)

        training_data = np.c_[training_data, np.ones((training_data.shape[0]))]

        for iterations in range(iterations):
            for (x, target) in zip(training_data, training_label):

                prediction = self.fdstep(np.dot(x, self.weights))

                if prediction != target:
                    error = prediction - target
                    self.weights[0] += error * 1
                    self.weights[1:] -= self.lr * (error * x[1:] + fdlambda * self.weights[1:])

            accuracy = self.accuracy(validation_input, validation_label)
            if accuracy > best:
                best = accuracy
                counter = 0
            else:
                counter += 1
            if counter >= max_iter:
                print("Stopping at iteration", iterations)
                break

    def predict(self, training_data):
        if training_data.ndim > 2:
            training_data = training_data.reshape(training_data.shape[0], -1)
        training_data = np.c_[training_data, np.ones((training_data.shape[0], 1))]
        predictions = np.dot(training_data, self.weights)
        return self.fdstep(predictions)

    def export_face_weights(self):
        with open("fd_perceptron_face_weights.txt", "w") as file:
            weights_str = '\n'.join(str(w) for w in self.weights)
            file.write(weights_str)

    def accuracy(self, training_data, training_label):
        predictions = self.predict(training_data)
        correct_predictions = np.sum(predictions == training_label)
        return correct_predictions / len(training_label) * 100


class FDPerceptronDigit:

    def __init__(self, num_features, rate, num_classes=10):  # learning rate, num_classes is 0-9
        self.lr = rate
        self.weights = {i: np.random.randn(num_features + 1) for i in
                        range(num_classes)}  # creates a dictionary w/ keys 0-9
        self.num_classes = num_classes

    # TRAINING WITH FUNCTIONS
    # training_data - training data
    # training_label - expected label
    # iterations - cycle over training dataset
    def train(self, training_data, training_label, iterations):

        if training_data.ndim > 2:
            training_data = training_data.reshape(training_data.shape[0], -1)

        training_data = np.c_[training_data, np.ones((training_data.shape[0]))]
        for iterations in range(iterations):
            for (x, target) in zip(training_data, training_label):
                x = x.flatten()
                predictions = {label: np.dot(x, self.weights[label]) for label in range(self.num_classes)}
                predicted_class = max(predictions, key=predictions.get)

                if predicted_class != target:
                    self.weights[predicted_class] -= self.lr * x
                    self.weights[target] += self.lr * x

    def predict(self, training_data):
        if training_data.ndim > 2:
            training_data = training_data.reshape(training_data.shape[0], -1)
        training_data = np.c_[training_data, np.ones((training_data.shape[0]))]
        predictions = []
        for x in training_data:
            class_scores = {label: np.dot(x, self.weights[label]) for label in range(self.num_classes)}
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return predictions

    def export_digit_weights(self):
        with open("fd_perceptron_digit_weights.txt", "w") as file:
            for label, weights in self.weights.items():
                weights_str = ' '.join(str(w) for w in weights)
                file.write(f"Label {label}: {weights_str}\n")

    def accuracy(self, training_data, training_label):
        predictions = self.predict(training_data)
        correct_predictions = np.sum(predictions == training_label)
        return correct_predictions / len(training_label) * 100
