#PyTorch practice



#Tensors:
# - similar to arrays, but can run on GPU
# - often share the same memory location as numpy arrays, so
#   you don't need to copy data between them
# - optimized for automatic differentiation


import torch
import numpy as np

#creating a tensor directly from data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)


#creating a tensor from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


#tensors have a shape, datatype (dtype), and device they are stored on
#
