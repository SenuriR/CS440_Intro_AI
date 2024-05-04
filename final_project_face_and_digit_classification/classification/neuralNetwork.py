import numpy as np
import scipy.optimize as opt

class NeuralNetworkClassifier():
    def __init__(self, legalLabels, numInput, numHidden, numOutput, numData, l):
        
        # basic intialization
        self.numInput = numInput
        self.numHidden = numHidden
        self.numOutput = numOutput
        self.numData = numData
        self.legalLabels = legalLabels
        self.numFeatures = numInput
        self.numClasses = numOutput
        self.l = l

        # step 1 - randomly initialize weights
        self.inputWeights = np.random.rand(self.numHidden, self.numInput + 1)
        self.outputweights = np.random.rand(self.numOutput, self.numHidden + 1)

        # initialize input, hidden, and output activation functions
        inputActivationDims = (self.numInput + 1, numData) # +1 for bias
        hiddenActivationDims = (self.numHidden + 1, numData)
        outputActivationDims = (self.numOutput, numData)
        self.inputActivation = np.ones(inputActivationDims)
        self.hiddenActivation = np.ones(hiddenActivationDims)
        self.outputActivation = np.ones(outputActivationDims)

        # initialize input and output delta matrices
        inputDeltaDims = (self.numHidden, self.numInput+1)
        outputDeltaDims = (self.numOutput, self.numHidden+1)
        self.inputDelta = np.zeros(inputDeltaDims)
        self.outputDelta = np.zeros(outputDeltaDims)

        # also bias vector
        biasVectorDims = (1, numData)
        self.biasVector = np.ones(biasVectorDims)

    # (step 2 - implement forward propagation to get h_theta_Xi for any instance Xi)
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
        # theta is currently a vector that contains all the (updated) weights involved with our NN after going through the feed forward process
        # we want to split this vector up into inputWeights and outputWeights
        weights_in_hidden = theta[0:self.numHidden * (self.numInput + 1)]
        weights_hidden_out = theta[-self.numOutput * (self.numHidden + 1):]
        dim_hidden_in = (self.numHidden, self.numInput + 1)
        dim_out_hidden = (self.numOutput, self.numHidden + 1)
        self.inputWeights = weights_in_hidden.reshape(dim_hidden_in)
        self.outputWeights = weights_hidden_out.reshape(dim_out_hidden)

        # calculate lower case delta
        finalOutputError = self.outputActivation - self.trueOut
        # by hiddenError I mean all the errors inside the net that we have to use the formula in notes for (slide 41 lecture NN)
        output_exclude_final_vector = self.outputWeights[:, :-1]
        weight_tranpose_error = output_exclude_final_vector.T.dot(finalOutputError)
        gprime_activation = derivActivationFunctionSigmoid(self.hiddenActivation[:-1:])
        hiddenError = weight_tranpose_error * gprime_activation

        # calculate upper case delta and add regulations
        self.outputDelta = (1/self.numData) * finalOutputError.dot(self.hiddenActivation.T)
        self.inputDelta = (1/self.numData) * hiddenError.dot(self.inputActivation.T)
        regularizationOut = self.l * self.outputWeights[:, :-1]
        regularizationIn = self.l * self.inputWeights[:, :-1]
        self.outputDelta[:, :-1].__add__(regularizationOut)
        self.inputDelta[:, :-1].__add__(regularizationIn)

        # return updated weight vector
        flatIn = self.inputDelta.ravel()
        flatOut = self.outputDelta.ravel()
        return np.append(flatIn, flatOut)
    
    def gradientChecking(self):
        print("sen")

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        # basic intiialization
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
        "input activation"
        "for classify in case the difference of size between trainData and testData "
        self.size_test = len(list(testData))
        features_test = [];
        for datum in testData:
            feature = list(datum.values())
            features_test.append(feature)
        test_set = np.array(features_test, np.int32)
        feature_test_set = test_set.transpose()

        if feature_test_set.shape[1] != self.inputActivation.shape[1]:
            self.inputActivation = np.ones((self.input + 1, feature_test_set.shape[1]))
            self.hiddenActivation = np.ones((self.hidden + 1, feature_test_set.shape[1]))
            self.outputActivation = np.ones((self.output + 1, feature_test_set.shape[1]))
        self.inputActivation[:-1, :] = feature_test_set

        "hidden activation"
        hiddenZ = self.inputWeights.dot(self.inputActivation)
        self.hiddenActivation[:-1, :] = sigmoid(hiddenZ)

        "output activation"
        outputZ = self.outputWeights.dot(self.hiddenActivation)
        self.outputActivation = sigmoid(outputZ)
        if self.output > 1:
            return np.argmax(self.outputActivation, axis=0).tolist()
        else:
            return (self.outputActivation>0.5).ravel()

    def costFunction(self, input, output):
        return -1 * ((input * np.log(output)) + ((1 - input)*np.log(1 - output)))

    
#################### HELPER FUNCTIONS ####################

    def getTrainingSet(self):
        # the training size will be the size of the list of the training data given
        self.training_size = len(list(self.trainData))
        feat_train = []
        # for every data input in the training data
        for data in self.trainData:
            # the feature will just be the list of data values
            feat = list(data.values())
            # add the new feat to the features we want to use to train nn
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
