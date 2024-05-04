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
    def forwardProp(self, inputData):
        """
        inputData: A numpy array of shape (numFeatures, numData) representing the input data.

        Step 1: Ensure an explicit bias term that nn can work from
        Step 2: Calculate weighed sum of inputs in hidden layer
        Step 3: Apply activation function (signmoid) to hidden layer
        Step 4: Calculate weighed sum of inputs for output layer
        Step 5: Activation function to output layer, softmax for number classification, sigmoid for face detection

        Returns:
            A numpy array of shape (numOutput, numData) representing the network's output.
        """

        # Step 1
        self.inputActivation[0:, :] = np.concatenate((inputData), axis=0)

        # Step 2
        self.hiddenActivation[1:, :] = np.dot(self.inputWeights, self.inputActivation)

        # Step 3
        self.hiddenActivation[1:, :] = activationFunctionSigmoid(self.hiddenActivation[1:, :])

        # Step 4 
        self.outputActivation[1:, :] = np.dot(self.outputweights, self.hiddenActivation)

        # Step 5
        if self.numClasses > 2: # Numbers
            self.outputActivation[1:, :] = softmax(self.outputActivation[1:, :])
        else: # Gace detection
            self.outputActivation[1:, :] = activationFunctionSigmoid(self.outputActivation[1:, :])

        return self.outputActivation[1:, :]  # Output w/o bias



    
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
        self.testData = testData
        feat_test_set = self.getTestSet()

        if feat_test_set.shape[1] != self.inputActivation.shape[1]:
            self.inputActivation = np.ones((self.numInput + 1, feat_test_set.shape[1]))
            self.hiddenActivation = np.ones((self.numHidden + 1, feat_test_set.shape[1]))
            self.outputActivation = np.ones((self.numOutput + 1, feat_test_set.shape[1]))
        self.inputActivation[:-1, :] = feat_test_set

        hiddenZ = self.inputWeights.dot(self.inputActivation)
        self.hiddenActivation[:-1, :] = activationFunctionSigmoid(hiddenZ)

        outputZ = self.outputWeights.dot(self.hiddenActivation)
        self.outputActivation = activationFunctionSigmoid(outputZ)
        
        if self.output > 1:
            return np.argmax(self.outputActivation, axis=0).tolist()
        else:
            return (self.outputActivation>0.5).ravel()

    def costFunction(self, input, output):
        return -1 * ((input * np.log(output)) + ((1 - input)*np.log(1 - output)))

    
#################### HELPER FUNCTIONS ####################

    def getTestSet(self):
        self.size_test = len(list(self.testData))
        feat_test = []
        for data in self.testData:
            feat = list(data.values())
            feat_test.append(feat)
        test_set = np.array(feat_test, np.int32)
        feat_test_set = test_set.transpose()
        return feat_test_set
    
    def getTrainingSet(self):
        # the training size will be the size of the list of the training data given
        self.training_size = len(list(self.trainData))
        feat_train = []
        # for every data input in the training datatestData
        for data in self.trainData:
            # the feature will just be the list of data values
            feat = list(data.values())
            # add the new feat to the features we want to use to train nn
            feat_train.append(feat)
        
        training_set = np.array(feat_train, np.int32) # found this np.int32 online, should be helpful here
        return training_set

# useful functions:
def activationFunctionSigmoid(x):
    g = 1.0 / (1.0 + np.exp(-x))
    return g

def derivActivationFunctionSigmoid(y):
    d = y * (1.0 - y)
    return d


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
