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

"""This program runs an augmented spatial pooler"""

'''
TODO:
Test the script with more inputs and more complex images
'''

import numpy
from random import randrange, random
from spatial_pooler import SpatialPooler
import Image
from bindings.math import (SM32 as SparseMatrix,
                                 SM_01_32_32 as SparseBinaryMatrix,
                                 GetNTAReal,
                                 Random as NupicRandom)



realDType = GetNTAReal()
VERSION=2

#this class extends the SpatialPooler class. So it still has access to the Spatial Pooler 
#methods and variables, as well as its own methods 

class AugmentedSpatialPooler(SpatialPooler):
	def __init__(self, inputDimensions, columnDimensions):
	    SpatialPooler.__init__(self, inputDimensions, columnDimensions)


	def _calculateMeanInput(self, inputVector):
		return numpy.ndarray.mean(inputVector)

	# This is the equivalent of learnSynapseConnections
	def _adaptSynapses2(self, inputVector, activeColumns):
	    """
	    The primary method in charge of learning. Adapts the permanence values of
	    the synapses based on the input vector, and the chosen columns after
	    inhibition round. Permanence values are increased for synapses connected to
	    input bits that are turned on, and decreased for synapses connected to
	    inputs bits that are turned off.

	    Parameters:
	    ----------------------------
	    @param inputVector:
	                    A numpy array of 0's and 1's that comprises the input to
	                    the spatial pooler. There exists an entry in the array
	                    for every input bit.
	    @param activeColumns:
	                    An array containing the indices of the columns that
	                    survived inhibition.
	    """
	    meanInput = self._calculateMeanInput(inputVector)
	    #get indicies where the input synapse is greater than the mean average 
	    #check for if they are greater than connectThreshold will be done in 
	    #updatePermanencesForColumn
	    hiInputIndices = numpy.where((inputVector > meanInput) & (self._permanences >=self._synPermConnected))[0]
	    loInputIndices = numpy.where((inputVector > meanInput) & (self._permanences <self._synPermConnected))[0]
	    #create default array of permanence value changes
	    permChanges = numpy.zeros(self._numInputs)
	    permChanges2 = numpy.zeros(self._numInputs)
	    #fill array with default permanence values for inactive synapses
	    permChanges.fill(-1 * self._synPermInactiveDec)
	    permChanges2.fill(-1 * self._synPermInactiveDec)
	    #update permanence changes array with inputs above the meanInput
	    permChanges[loInputIndices] = self._synPermActiveInc
	    permChanges2[hiInputIndices] = self._synPermActiveInc
	    for i in activeColumns:
	        perm = self._permanences.getRow(i)
	        perm2 = self._permanences.getRow(i)
	        maskPotential = numpy.where(self._potentialPools.getRow(i) > 0)[0]
	        perm[maskPotential] += permChanges[maskPotential]
	        perm2[maskPotential] += permChanges2[maskPotential]
	        self._updatePermanencesForColumn(perm, i, raisePerm=True)
	        self._updatePermanencesForColumn2(perm2, i, meanInput, raisePerm=True)

	def _bumpUpWeakColumns(self):
	    """
	    This method increases the permanence values of synapses of columns whose
	    activity level has been too low. Such columns are identified by having an
	    overlap duty cycle that drops too much below those of their peers. The
	    permanence values for such columns are increased.
	    """
	    weakColumns = numpy.where(self._overlapDutyCycles
	                                < self._minOverlapDutyCycles)[0]
	    for i in weakColumns:
	        perm = self._permanences.getRow(i).astype(realDType)
	        maskPotential = numpy.where(self._potentialPools.getRow(i) > 0)[0]
	        perm[maskPotential] += self._synPermBelowStimulusInc
	        self._updatePermanencesForColumn(perm, i, raisePerm=False)

	def _raisePermanenceToThreshold(self, perm, mask):
	    """
	    This method ensures that each column has enough connections to input bits
	    to allow it to become active. Since a column must have at least
	    'self._stimulusThreshold' overlaps in order to be considered during the
	    inhibition phase, columns without such minimal number of connections, even
	    if all the input bits they are connected to turn on, have no chance of
	    obtaining the minimum threshold. For such columns, the permanence values
	    are increased until the minimum number of connections are formed.


	    Parameters:
	    ----------------------------
	    @param perm:    An array of permanence values for a column. The array is
	                    "dense", i.e. it contains an entry for each input bit, even
	                    if the permanence value is 0.
	    @param mask:    the indices of the columns whose permanences need to be
	                    raised.
	    """
	    if len(mask) < self._stimulusThreshold:
	        raise Exception("This is likely due to a " +
	        "value of stimulusThreshold that is too large relative " +
	        "to the input size. [len(mask) < self._stimulusThreshold]")

	    numpy.clip(perm, self._synPermMin, self._synPermMax, out=perm)
	    while True:
	        numConnected = numpy.nonzero(perm > self._synPermConnected)[0].size
	        if numConnected >= self._stimulusThreshold:
	            return
	        perm[mask] += self._synPermBelowStimulusInc

	def _updatePermanencesForColumn2(self, perm, index, meanInput, raisePerm=True):
	    """
	    This method updates the permanence matrix with a column's new permanence
	    values. The column is identified by its index, which reflects the row in
	    the matrix, and the permanence is given in 'dense' form, i.e. a full
	    array containing all the zeros as well as the non-zero values. It is in
	    charge of implementing 'clipping' - ensuring that the permanence values are
	    always between 0 and 1 - and 'trimming' - enforcing sparsity by zeroing out
	    all permanence values below '_synPermTrimThreshold'. It also maintains
	    the consistency between 'self._permanences' (the matrix storing the
	    permanence values), 'self._connectedSynapses', (the matrix storing the bits
	    each column is connected to), and 'self._connectedCounts' (an array storing
	    the number of input bits each column is connected to). Every method wishing
	    to modify the permanence matrix should do so through this method.

	    Parameters:
	    ----------------------------
	    @param perm:    An array of permanence values for a column. The array is
	                    "dense", i.e. it contains an entry for each input bit, even
	                    if the permanence value is 0.
	    @param index:   The index identifying a column in the permanence, potential
	                    and connectivity matrices
	    @param raisePerm: A boolean value indicating whether the permanence values
	                    should be raised until a minimum number are synapses are in
	                    a connected state. Should be set to 'false' when a direct
	                    assignment is required.
	    """

	    maskPotential = numpy.where(self._potentialPools.getRow(index) > 0)[0]
	    if raisePerm:
	        self._raisePermanenceToThreshold(perm, maskPotential)
	    perm[perm < self._synPermTrimThreshold] = 0
	    numpy.minimum(perm+self._synPermActiveInc, self._synPermConnected-self._synPermActiveInc, out=perm)
	    newConnected = numpy.where(perm >= self._synPermConnected)[0]
	    self._permanences.setRowFromDense(index, perm)
	    self._connectedSynapses.replaceSparseRow(index, newConnected)
	    self._connectedCounts[index] = newConnected.size

	def _updateBoostFactors2(self):
	    """
	    Update the boost factors for all columns. The boost factors are used to
	    increase the overlap of inactive columns to improve their chances of
	    becoming active. and hence encourage participation of more columns in the
	    learning process. This is a line defined as: y = mx + b boost =
	    (1-maxBoost)/minDuty * dutyCycle + maxFiringBoost. Intuitively this means
	    that columns that have been active enough have a boost factor of 1, meaning
	    their overlap is not boosted. Columns whose active duty cycle drops too much
	    below that of their neighbors are boosted depending on how infrequently they
	    have been active. The more infrequent, the more they are boosted. The exact
	    boost factor is linearly interpolated between the points (dutyCycle:0,
	    boost:maxFiringBoost) and (dutyCycle:minDuty, boost:1.0).

	            boostFactor
	                ^
	    maxBoost _  |
	                |\
	                | \
	          1  _  |  \ _ _ _ _ _ _ _
	                |
	                +--------------------> activeDutyCycle
	                   |
	            minActiveDutyCycle
	    """
	    #get boost factor indicies for columns where the boost factor is greater than the
	    #max boost
	    mask = numpy.where(self._boostFactors > self._maxBoost)[0]

	    #get columns where activity is less than min activity
	    self._boostFactors[(self._activeDutyCycles >self._minActiveDutyCycles)]
	    #set column boost values to 1 where activity(c) < minActivity (c) and boost(c) > max boost
	    self._boostFactors[mask] = 1.0

	    #maxPerm = self._permanences.max()[2]

	    #loop through columns
	    for i in range(0, self._numColumns):
	        perm = self._permanences.getRow(i)
	        #get indicies of disconnected synapses
	        disconnectedIndicies = numpy.where(self._synPermConnected > perm)[0]
	        #get the max value of the set of disconnected indicies
	        maxPerm = numpy.ndarray.max(disconnectedIndicies)
	        #get the array of disconnected synapses
	        disconnected = perm[disconnectedIndicies]
	        #get syanpses where the permanence of the synapse is greater than the max permanence
	    	resetPerm = numpy.where(disconnected > maxPerm)
	    	#get closest synapse
	    	maxS = resetPerm[0]
	    	#set the synapse permanence to the max synapse
	    	perm[maxS] = self._synPermConnected + self._synPermActiveInc
	    	

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
		    self._adaptSynapses2(inputVector, activeColumns)
		    self._updateDutyCycles(overlaps, activeColumns)
		    self._bumpUpWeakColumns()
		    self._updateBoostFactors2()
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