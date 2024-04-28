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
    backProp implements the back propagation step of neural network classification.
    First step: obtain input- and output-weights
    Second step: compute (lower case) delta - y, compute errors...
    Third step: compute (upper case) delta... given (lower case )delta computations from second step
    Fourth step: compute average regularized gradient D
    Fifth step: return updated weight vector
    '''
    def backProp(self, theta):
        # First step
        inPreReshape = theta[0:self.numHidden * (self.numInput + 1)]
        outPreReshape = theta[-self.numOutput * (self.numHidden + 1):]
        self.inputWeights = inPreReshape.reshape((self.numHidden, self.numInput + 1))
        self.outputWeights = outPreReshape.reshape((self.numOutput, self.numHidden + 1))

        # Second step
        outputError = self.outputActivation - self.trueOut
        hiddenError = self.outputWeights[:, :-1].T.dot(outputError) * derivActivationFunctionSigmoid(self.hiddenActivation[:-1:]) # FROM NOTES

        # Third step
        self.outputDelta = (1/self.numData) * outputError.dot(self.hiddenActivation.T)
        self.inputDelta = (1/self.numData) * hiddenError.dot(self.inputActivation.T)

        # Fourth step
        regularizationOut = self.l * self.outputWeights[:, :-1]
        regularizationIn = self.l * self.inputWeights[:, :-1]
        self.outputDelta[:, :-1].__add__(regularizationOut)
        self.inputDelta[:, :-1].__add__(regularizationIn)

        # Fifth step
        flatIn = self.inputDelta.ravel()
        flatOut = self.outputDelta.ravel()
        return np.append(flatIn, flatOut)

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.trainData = trainingData
        self.trainLabels = trainingLabels
        self.validData = validationData
        self.validLabels = validationLabels

        training_set = self.getTrainingSet()

        iteration = 100
        self.inputActivation[:-1, :] = training_set.transpose()
        self.trueOut = np.zeros((self.numOutput, self.numData))
        for ithLabel in range(self.numData):
            label = trainingLabels[ithLabel]
            if self.numOutput != 1:
                self.trueOut[label, ithLabel] = 1
            else:
                self.trueOut[:,1] = label # if there is one output neuron
    

        flatInWeights = self.inputWeights.ravel()
        flatOutWeights = self.outputWeights.ravel()
        theta = np.append(flatInWeights, flatOutWeights)
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

# useful functions:
def activationFunctionSigmoid(x):
    g = 1.0 / (1.0 + np.exp(-x))
    return g

def derivActivationFunctionSigmoid(y):
    d = y * (1.0 - y)
    return d
