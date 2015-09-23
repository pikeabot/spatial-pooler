# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""A simple program that demonstrates the working of the spatial pooler"""

import numpy
from random import randrange, random
#from nupic.bindings.algorithms import SpatialPooler
from nupic.research.spatial_pooler import SpatialPooler
import Image
from nupic.bindings.math import (SM32 as SparseMatrix,
                                 SM_01_32_32 as SparseBinaryMatrix,
                                 GetNTAReal,
                                 Random as NupicRandom)



realDType = GetNTAReal()
VERSION=2
class AugmentedSpatialPooler(SpatialPooler):
	def __init__(self, inputDimensions, columnDimensions):
	    SpatialPooler.__init__(self, inputDimensions, columnDimensions)
        '''
	    inputDimensions = numpy.array((16,16), ndmin=1)
	    columnDimensions = numpy.array((16,16), ndmin=1)

	    numInputs = inputDimensions.prod()
	    numColumns = columnDimensions.prod()
	    self._numInputs = int(numInputs)
	    self._numColumns = int(numColumns)
	    self._columnDimensions = columnDimensions
	    self._inputDimensions = inputDimensions	    
	    #self._numColumns = numColumns	
	    #self._numInputs = numInputs
	    self._connectedSynapses = SparseBinaryMatrix(numInputs)
	    self._connectedSynapses.resize(numColumns, numInputs)
	    self._boostFactors = numpy.ones(numColumns, dtype=realDType)
	    self._inhibitionRadius = 0
	    self._updateInhibitionRadius()

	    self._potentialPools = SparseBinaryMatrix(numInputs)
	    self._potentialPools.resize(numColumns, numInputs)
	    self._synPermTrimThreshold = .05 / 2.0
	    self._potentialRadius = int(min(16, numInputs))
	    self._potentialPct = .5
	    self._globalInhibition = False
	    self._numActiveColumnsPerInhArea = int(10.0)
	    self._localAreaDensity = -1.0
	    self._stimulusThreshold = 0
	    self._synPermInactiveDec = 0.008
	    self._synPermActiveInc = 0.05
	    self._synPermBelowStimulusInc = 0.10 / 10.0
	    self._synPermConnected = 0.10
	    self._minPctOverlapDutyCycles = 0.001
	    self._minPctActiveDutyCycles = 0.001
	    self._dutyCyclePeriod = 1000
	    self._maxBoost = 10.0
	    self._spVerbosity = 0
	    self._wrapAround = True
	    self._synPermMin = 0.0
	    self._synPermMax = 1.0

	    #self._updateInhibitionRadius()
	    self._tieBreaker = 0.01*numpy.array([self._random.getReal64() for i in
                                        xrange(self._numColumns)])

	    if self._synPermTrimThreshold >= self._synPermConnected:
	      raise InvalidSPParamValueError(
	        "synPermTrimThreshold ({}) must be less than synPermConnected ({})"
	        .format(repr(self._synPermTrimThreshold),
	                repr(self._synPermConnected)))

	    self._updatePeriod = 50
	    initConnectedPct = 0.5
	    self._version = VERSION
	    self._iterationNum = 0
	    self._iterationLearnNum = 0
	    self._permanences = SparseMatrix(numColumns, numInputs)
	    self._connectedCounts = numpy.zeros(numColumns, dtype=realDType)

	    # Initialize the set of permanence values for each column. Ensure that
	    # each column is connected to enough input bits to allow it to be
	    # activated.
	    for columnIndex in xrange(numColumns):
	        potential = self._mapPotential(columnIndex, wrapAround=self._wrapAround)
	        self._potentialPools.replaceSparseRow(i, potential.nonzero()[0])
	        perm = self._initPermanence(potential, initConnectedPct)
	        self._updatePermanencesForColumn(perm, columnIndex, raisePerm=True)

	    self._overlapDutyCycles = numpy.zeros(numColumns, dtype=realDType)
	    self._activeDutyCycles = numpy.zeros(numColumns, dtype=realDType)
	    self._minOverlapDutyCycles = numpy.zeros(numColumns,
	                                             dtype=realDType)
	    self._minActiveDutyCycles = numpy.zeros(numColumns,
	                                            dtype=realDType)
	    self._boostFactors = numpy.ones(numColumns, dtype=realDType)

	    # The inhibition radius determines the size of a column's local
	    # neighborhood.  A cortical column must overcome the overlap score of
	    # columns in its neighborhood in order to become active. This radius is
	    # updated every learning round. It grows and shrinks with the average
	    # number of connected synapses per column.
	    self._inhibitionRadius = 0
	    self._updateInhibitionRadius()

	    if self._spVerbosity > 0:
	      self.printParameters()
	    super(AugmentedSpatialPooler, self).__init__()
	    

	def _calculateOverlap(self, inputVector):
	    """
	    This function determines each column's overlap with the current input
	    vector. The overlap of a column is the number of synapses for that column
	    that are connected (permanence value is greater than '_synPermConnected')
	    to input bits which are turned on. Overlap values that are lower than
	    the 'stimulusThreshold' are ignored. The implementation takes advantage of
	    the SparseBinaryMatrix class to perform this calculation efficiently.
	    Parameters:
	    ----------------------------
	    @param inputVector: a numpy array of 0's and 1's that comprises the input to
	                    the spatial pooler.
	    """

	    overlaps = numpy.zeros(self._numColumns).astype(realDType)
	    self._connectedSynapses.rightVecSumAtNZ_fast(inputVector, overlaps)
	    overlaps[overlaps < self._stimulusThreshold] = 0
	    return overlaps
	    '''
	def compute2(self, inputVector, learn, activeArray):
	    """
	    This is the primary public method of the SpatialPooler class. This
	    function takes a input vector and outputs the indices of the active columns.
	    If 'learn' is set to True, this method also updates the permanences of the
	    columns.
	    @param inputVector: A numpy array of 0's and 1's that comprises the input
	        to the spatial pooler. The array will be treated as a one dimensional
	        array, therefore the dimensions of the array do not have to match the
	        exact dimensions specified in the class constructor. In fact, even a
	        list would suffice. The number of input bits in the vector must,
	        however, match the number of bits specified by the call to the
	        constructor. Therefore there must be a '0' or '1' in the array for
	        every input bit.
	    @param learn: A boolean value indicating whether learning should be
	        performed. Learning entails updating the  permanence values of the
	        synapses, and hence modifying the 'state' of the model. Setting
	        learning to 'off' freezes the SP and has many uses. For example, you
	        might want to feed in various inputs and examine the resulting SDR's.
	    @param activeArray: An array whose size is equal to the number of columns.
	        Before the function returns this array will be populated with 1's at
	        the indices of the active columns, and 0's everywhere else.
	    """
	    if not isinstance(inputVector, numpy.ndarray):
	      	raise TypeError("Input vector must be a numpy array, not %s" %
	                      str(type(inputVector)))

	    if inputVector.size != self._numInputs:
	      	raise ValueError(
	          "Input vector dimensions don't match. Expecting %s but got %s" % (
	              inputVector.size, self._numInputs))

	    self._updateBookeepingVars(learn)
	    inputVector = numpy.array(inputVector, dtype=realDType)
	    inputVector.reshape(-1)
	
	    overlaps = self._calculateOverlap(inputVector)

	    # Apply boosting when learning is on
	    if learn:
	      	boostedOverlaps = self._boostFactors * overlaps
	    else:
	      	boostedOverlaps = overlaps

	    # Apply inhibition to determine the winning columns
	    activeColumns = self._inhibitColumns(boostedOverlaps)

	    if learn:
		    self._adaptSynapses(inputVector, activeColumns)
		    self._updateDutyCycles(overlaps, activeColumns)
		    self._bumpUpWeakColumns()
		    self._updateBoostFactors()
		    if self._isUpdateRound():
		        self._updateInhibitionRadius()
		        self._updateMinDutyCycles()

	    activeArray.fill(0)
	    if activeColumns.size > 0:
	      	activeArray[activeColumns] = 1

#input dimensions, i.e. dimensions of images
inputDimensions = (16,16)
#the number of columns
columnDimensions = (16, 16)
#input size = width * height of input
inputSize = numpy.array(inputDimensions).prod()
#columnNumber = width * height of columns
columnNumber = numpy.array(columnDimensions).prod()
#inputArray
inputArray = numpy.zeros(inputSize)
#activeArray
activeArray = numpy.zeros(columnNumber)
#last active array to see if there were any changes in the permanence since the last run
#through of all of the images
lastActiveArray=numpy.zeros(columnNumber)
#initialize spatialpooler class
sp = SpatialPooler(inputDimensions,
	                 columnDimensions,
	                 potentialRadius = inputSize,
	                 numActiveColumnsPerInhArea = int(0.02*columnNumber),
	                 globalInhibition = True,
	                 synPermActiveInc = 0.01)

# this function runs a set of images through the spatial pooler
def run(img_list):
    """Run the spatial pooler with the input vector"""
    for img in img_list:
    	#compute calculates the overlap, inhibits the columns and then runs the learn
    	#synapse connection method
    	#it then updates the activeArray

    	asp.compute2(img.ravel(), True, activeArray)

    	print activeArray.nonzero()

asp = AugmentedSpatialPooler((16,16), (16,16))
#in this example I read in a greyscale image and then take 16px x 16px samples from 
#different places in the image
img = Image.open('img/parrot.png').convert('1')
arr = numpy.array(img)
#convert the image to binary -so it should just be 0's and 1's
inputVector = arr.astype(int)
inputVector[inputVector>0]=1

img_list=[]

#create a list of the sample patches to make the sets easy to run through
img_list.append(inputVector[0:16, 1:17])
img_list.append(inputVector[125:141, 100:116])
img_list.append(inputVector[153:169, 44:60])

for i in range(0,3):
	print "current loop: " + str(i)
	run(img_list)  #run the calculateOverlap, inhibitColumns and learnSynapseConnections

	#check for convergance. If the last active array and the current active arrays 
	#are the same, then the spatial pooler has converged and exit the algorithm
	if numpy.array_equal(activeArray, lastActiveArray):
		print str(i)+ " loops"
		print activeArray.nonzero()
		print lastActiveArray.nonzero()
		break
	else:
		lastActiveArray=activeArray