import numpy as np

class NeuralNetworkClassifier():
    def __init__(self, legalLabels, numInput, numHidden, numOutput, numData, l):
        self.numInput = numInput
        self.numHidden = numHidden
        self.numOutput = numOutput
        self.numData = numData
        self.legalLabels = legalLabels
        self.l = l

        inputActivationDims = (self.numInput + 1, numData)
        hiddenActivationDims = (self.numHidden + 1, numData)
        outputActivationDims = (self.numOutput, numData)

        self.inputActivation = np.ones(inputActivationDims)
        self.hiddenActivation = np.ones(hiddenActivationDims)
        self.outputActivation = np.ones(outputActivationDims)

        biasVectorDims = (1, numData)
        self.biasVector = np.ones(biasVectorDims)

        inputDeltaDims = (self.numHidden, self.numInput+1)
        outputDeltaDims = (self.numOutput, self.numHidden+1)
        self.inputDelta = np.zeros(inputDeltaDims)
        self.outputDelta = np.zeros(outputDeltaDims)

        # randomly assign weights - not sure if I did this right, need to check this.
        self.hiddenRandomFactor = np.sqrt(6.0 / (self.numInput + self.numHidden))
        self.outputRandomFactor = np.sqrt(6.0 / (self.numInput + self.numOutput))
        self.inputWeights = np.random.rand(self.numHidden, self.numInput + 1) * 2 * self.hiddenRandomFactor - self.hiddenRandomFactor
        self.outputweights = np.random.rand(self.numOutput, self.numHidden + 1) * 2 * self.outputRandomFactor - self.outputRandomFactor

        self.l = l

    def forwardProp(self):
        print("Implement forward propagation here")

    def backProp(self):
        print("Implement backpropagation here")

    def train(self, trainData, trainLabels, validData, validLabels):
        print("Implement actually training sequence here (after forward and back are complete).")

    def classification(self, testData):
        print("Implement final classification using test data here")


    # useful functions:
    def activationFunctionSigmoid(x):
        g = 1.0 / (1.0 + np.exp(-x))
        return g
    
    def derivActivationFunctionSigmoid(y):
        d = y * (1.0 - y)
        return d
