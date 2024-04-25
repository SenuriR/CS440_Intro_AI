import numpy as np
import scipy.optimize as opt

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

    
    '''
    backPropagation implements the back propagation step of neural network classification.
    First step: obtain input- and output-weights
    Second step: compute (lower)delta - y, and compute errors...
    Third step: compute (uppder)delta... given (lower)delta computations from second step
    Fourth step: compute average regularized gradient D
    Fifth step: return updated weight vector
    '''
    def backPropagation(self, theta):
        # First step
        inPreReshape = theta[0:self.numHidden * (self.numInput + 1)]
        outPreReshape = theta[-self.numOutput * (self.numHidden + 1):]
        self.inputWeights = inPreReshape.reshape((self.numHidden, self.numInput + 1))
        self.outputWeights = outPreReshape.reshape((self.numOutput, self.numHidden + 1))

        # Second step
        outputError = self.outputActivation - self.outputTruth
        hiddenError = self.outputWeights[:, :-1].T.dot(outputError) * derivActivationFunctionSigmoid(self.hiddenActivation[:-1:])

        # Third step
        self.outputChange = outputError.dot(self.hiddenActivation.T) * (1/self.numData)
        self.inputChange = hiddenError.dot(self.inputActivation.T) * (1/self.numData)

        # Fourth step
        regularizationOut = self.l * self.outputWeights[:, :-1]
        regularizationIn = self.l * self.inputWeights[:, :-1]
        self.outputChange[:, :-1].__add__(regularizationOut)
        self.inputChange[:, :-1].__add__(regularizationIn)

        # Fifth step
        return np.append(self.inputChange.ravel(), self.outputChange.ravel())

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.trainData = trainingData
        self.trainLabels = trainingLabels
        self.validData = validationData
        self.validLabels = validationLabels

        training_set = self.getTrainingSet()

        iteration = 100
        self.inputActivation[:-1, :] = training_set.transpose()
        self.outputTruth = self.getTruthMat(trainingLabels)

        theta = np.append(self.inputWeights.ravel(), self.outputWeights.ravel())
        theta = opt.fmin_cg(self.feedForward, theta, fprime=self.backPropagate, maxiter=iteration) # final theta (weights) AFTER both forward and backward
        inPreReshape = theta[0:self.numHidden * (self.numInput + 1)]
        outPreReshape = theta[-self.numOutput * (self.numHidden + 1):]
        self.inputWeights = inPreReshape.reshape((self.numHidden, self.numInput + 1))
        self.outputWeights = outPreReshape.reshape((self.numOutput, self.numHidden + 1))

    def classification(self, testData):
        print("Implement final classification using test data here")
    
#################### HELPER FUNCTIONS ####################

    def getTrainingSet(self):
        self.training_size = len(list(self.trainData))
        feat_train = []
        for data in self.trainData:
            feat = list(data.values())
            feat_train.append(feat)
        training_set = np.array(feat_train, np.int32)   
        return training_set

    def getTruthMat(self, trainingLabels):
        truth = np.zeros((self.numOutput, self.numData))
        for ithLabel in range(self.numData):
            label = trainingLabels[ithLabel]
            if self.numOutput != 1:
                truth[label, ithLabel] = 1
            else:
                truth[:,1] = label # if there is one output neuron
        return truth

# useful functions:
def activationFunctionSigmoid(x):
    g = 1.0 / (1.0 + np.exp(-x))
    return g

def derivActivationFunctionSigmoid(y):
    d = y * (1.0 - y)
    return d
