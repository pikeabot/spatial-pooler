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

import numpy as np
from random import randrange, random
from nupic.bindings.algorithms import SpatialPooler as SP
import Image


class Spooler(object):
  	"""A class to hold our code.

  	TODO: Get rid of this class, it just makes it more difficult to read the
  	code.
  	"""


  	def __init__(self, inputDimensions, columnDimensions):
	    """
	     Parameters:
	     ----------
	     _inputDimensions: The size of the input. (m,n) will give a size m x n
	     _columnDimensions: The size of the 2 dimensional array of columns
	     """
	    self.inputDimensions = inputDimensions
	    self.columnDimensions = columnDimensions
	    self.inputSize = np.array(inputDimensions).prod()
	    self.columnNumber = np.array(columnDimensions).prod()
	    self.inputArray = np.zeros(self.inputSize)
	    self.activeArray = np.zeros(self.columnNumber)

	    self.sp = SP(self.inputDimensions,
	                 self.columnDimensions,
	                 potentialRadius = self.inputSize,
	                 numActiveColumnsPerInhArea = int(0.02*self.columnNumber),
	                 globalInhibition = True,
	                 synPermActiveInc = 0.01)

  	def run(self):
	    """Run the spatial pooler with the input vector"""

	    print "-" * 80 + "Computing the SDR" + "-" * 80

	    print self.inputArray
	    #activeArray[column]=1 if column is active after spatial pooling
	    self.sp.compute(self.inputArray, True, self.activeArray)

	    print self.activeArray.nonzero()

img = Image.open('img/parrot.png').convert('1')
arr = np.array(img)
inputVector = arr.astype(int)
inputVector[inputVector>0]=1
#print inputVector.shape
#print inputVector
#print inputVector[0:3, 1:4]

spooler = Spooler((32, 32), (64, 64))
spooler.inputArray=inputVector[0:16, 0:16]
spooler.run()
