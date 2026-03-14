import numpy as np
data1 = [[1,2,3,4],[5,6,7,8]]
arr1 = np.array(data1)  #to create an array from the list
print(arr1)
print(data1.__class__) #to find the data of a data structure
print(arr1.__class__)
print(arr1.ndim)   # to find the dimensionality of the array
print(arr1.shape)
print(arr1.dtype)


# TO CREATE AN ARRAY OF ZEROES

zero = np.zeros((3,6))
print(zero)

# AN ARRAY OF FIRST 15 NUMBERS


fifteen = np.arange(15).reshape(5,3)
print(fifteen)


