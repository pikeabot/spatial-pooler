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
from spatial_pooler import SpatialPooler
import Image
from bindings.math import (SM32 as SparseMatrix,
                                 SM_01_32_32 as SparseBinaryMatrix,
                                 GetNTAReal,
                                 Random as NupicRandom)



realDType = GetNTAReal()
VERSION=2


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

    	sp.compute(img.ravel(), True, activeArray)

    	print activeArray.nonzero()

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