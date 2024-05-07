import numpy as np
import scipy.optimize as opt

class NeuralNetworkClassifier():
    def __init__(self, legalLabels, numInput, numHidden, numOutput, numData, l):
        
        # basic intialization
        self.input = numInput
        self.hidden = numHidden
        self.output = numOutput
        self.numData = numData
        self.legalLabels = legalLabels
        self.numFeatures = numInput
        self.numClasses = numOutput
        self.l = l

        # Initialize weights and bias
        self.w1 = np.random.randn(self.input, self.hidden) / np.sqrt(self.input)
        self.b1 = np.zeros((1, self.hidden))
        self.w2 = np.random.randn(self.hidden, self.output) / np.sqrt(self.hidden)
        self.b2 = np.zeros((1, self.output))
    
    # (step 2 - implement forward propagation to get h_theta_Xi for any instance Xi)
    def feed_forward(self, x, w1, w2):
        
        # hidden
        self.z1 = x.dot(w1) + self.b1
        # Activation function
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = self.a1.dot(w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2 # a2 here is h_theta_Xi

    def backProp(self, x, y, w1, w2, alpha):
        shape = x.shape[0]

        # Gradients
        dz2 = self.a2 - y

        dW2 = np.dot(np.transpose(self.a1),dz2) / shape

        db2 = np.sum(dz2,axis=0,keepdims=True) / shape

        dz1 = np.dot(dz2, np.transpose(w2)) * self.derivSigmoid(self.a1)

        dW1 = np.dot(np.transpose(x), dz1) / shape

        db1 = np.sum(dz1, axis=0, keepdims=True) / shape
        
        # Update Weights
        self.w2 -= alpha * dW2
        self.b2 -= alpha * db2
        self.w1 -= alpha * dW1
        self.b1 -= alpha * db1

        return (self.w1,self.w2)

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.trainData = list(trainingData)
        features = []
        for data in self.trainData:
            feature = np.array(list(data.values()))
            feature = feature.reshape(1,self.numFeatures) # reshape = (num of arrays, size of each array) in a 2d arr
            features.append(feature)
        trainingSet = np.array(features, np.int32)
        self.trainLabels = np.array(trainingLabels)
        self.validData = list(validationData)
        self.validLabels = list(validationLabels)
        epoch = 10
        acc = []
        losses = []
        alpha = self.l
        alpha = 0.1 # learning rate
        for j in range(epoch):
            l = []
            for i in range(len(self.trainData)):
                out = self.feed_forward(trainingSet[i], self.w1, self.w2)
                # out is now our "temporary answer" for the current training data
                losses.append((self.costFunction(out, self.trainLabels[i]))) # compute the loss "how wrong are we" (for testing purposes)
                self.w1, self.w2 = self.backProp(trainingSet[i], self.trainLabels[i], self.w1, self.w2, alpha)
            print("epochs:", j + 1, "======== acc:", (1-(sum(losses)/len(trainingSet)))*100)
            acc.append((1-(sum(losses)/len(self.trainData)))*100)

    def classify(self, testData):
        self.testData = list(testData)
        print(self.testData[0].values()) # unusual -- the testData here is "[]", causes rest of program to not run
        features = []
        print(self.testData)
        for data in self.testData:
            feature = np.array(list(data.values()))
            feature = feature.reshape(1,self.numFeatures) # reshape = (num of arrays, size of each array) in a 2d arr
            print(feature)
            features.append(feature)
        testSet = np.array(features, np.int32)
        
        # organize classifications to present results
        guesses = []
        print(len(testSet))
        for i in range(len(testSet)):
            out = self.feed_forward(testSet[i], self.w1, self.w2)
            for j in range(len(out[0])):
                if out[0][j] == np.max(out):
                    guesses.append(out[0][j])
        return guesses
    
    def costFunction(self, input, output):
        return (-1/self.numData) * ((input * np.log(output)) + ((1 - input)*np.log(1 - output)))
    
    # activation function
    def sigmoid(self, x):
        g = 1.0 / (1.0 + np.exp(-x))
        return g

    def derivSigmoid(self, y):
        d = y * (1.0 - y)
        return d
