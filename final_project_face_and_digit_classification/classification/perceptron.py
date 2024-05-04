# Perceptron implementation
import util
import numpy as np
PRINT = True

class Perceptron:
	def __init__(self, N, alpha=0.1):
		# initialize the weight matrix and store the learning rate
		self.W = np.random.randn(N + 1) / np.sqrt(N)
		self.alpha = alpha
  # N = # of columns in input feature vectors, set N equal to 2, since there are 2 inputs
  # alpha: Set value to 0.1 by default, can be 0.1,0.01, 0.001
  
  # step function- if x is positive return 1, otherwise return 0.
	def step(self, x):
		# apply the step function
		return 1 if x > 0 else 0

#fit function to fit the model to the data
	def fit(self, X, y, epochs=10):
		# insert a column of 1's as the last entry in the feature matrix -- this little trick allows us to treat the bias as a trainable parameter within the weight matrix
		X = np.c_[X, np.ones((X.shape[0]))]

  		# loop over the desired number of epochs
		for epoch in np.arange(0, epochs):      # loop over desired # of epocgs, for each epock loop over individual data point x & output target class label
			# loop over each individual data point
			for (x, target) in zip(X, y):  
				# take the dot product between the input features and the weight matrix, then pass this value through the step function to obtain the prediction by the perceptron
				p = self.step(np.dot(x, self.W))
				# only perform a weight update if our prediction does not match the target
				if p != target:
					# determine the error
					error = p - target
					# update the weight matrix
                # MAKE SURE TO CHANGE THE WEIGHT FOR BIAS W0 TO BE + 0R - 1 NOT ALPHA * ERRPR * X
					self.W += -self.alpha * error * x
     
     
  # predict the class labels for a given set of input data
	def predict(self, X, addBias=True):
		# ensure our input is a matrix
		X = np.atleast_2d(X)
		# check to see if the bias column should be added
		if addBias:
			# insert a column of 1's as the last entry in the feature
			# matrix (bias)
			X = np.c_[X, np.ones((X.shape[0]))]
		# take the dot product between the input features and the
		# weight matrix, then pass the value through the step
		# function
		return self.step(np.dot(X, self.W))
  
  
  # perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

  
  
  
  
  
  
# class PerceptronClassifier:
#   """
#   Perceptron classifier.
  
#   Note that the variable 'datum' in this code refers to a counter of features
#   (not to a raw samples.Datum).
#   """
#   def __init__( self, legalLabels, max_iterations):
#     self.legalLabels = legalLabels
#     self.type = "perceptron"
#     self.max_iterations = max_iterations
#     self.weights = {}
    
#     for label in legalLabels:
#       self.weights[label] = util.Counter() # this is the data-structure you should use

#   def setWeights(self, weights):
#     assert len(weights) == len(self.legalLabels);
#     self.weights == weights;
      
#   def train( self, trainingData, trainingLabels, validationData, validationLabels ):
#     """
#     The training loop for the perceptron passes through the training data several
#     times and updates the weight vector for each label based on classification errors.
#     See the project description for details. 
    
#     Use the provided self.weights[label] data structure so that 
#     the classify method works correctly. Also, recall that a
#     datum is a counter from features to values for those features
#     (and thus represents a vector a values).
#     """
    
#     self.features = trainingData[0].keys() # could be useful later
#     # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
#     # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
    
#     for iteration in range(self.max_iterations):
#       print ("Starting iteration " + iteration + "...")
#       for i in range(len(trainingData)):
#           "*** YOUR CODE HERE ***"
#           util.raiseNotDefined()
    
#   def classify(self, data ):
#     """
#     Classifies each datum as the label that most closely matches the prototype vector
#     for that label.  See the project description for details.
    
#     Recall that a datum is a util.counter... 
#     """
#     guesses = []
#     for datum in data:
#       vectors = util.Counter()
#       for l in self.legalLabels:
#         vectors[l] = self.weights[l] * datum
#       guesses.append(vectors.argMax())
#     return guesses

  
#   def findHighWeightFeatures(self, label):
#     """
#     Returns a list of the 100 features with the greatest weight for some label
#     """
#     featuresWeights = []

#     "*** YOUR CODE HERE ***"
#     util.raiseNotDefined()

#     return featuresWeights

