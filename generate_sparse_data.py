import numpy as np
from scipy.sparse import random

def generate_sparse_tensor(dim, t='int', tensor_density=1, tensor_format='coo'):
    tensor_max_dim = len(dim)
    assert(tensor_max_dim > 0)
    assert(dim[tensor_max_dim-1] > 0)

    if tensor_max_dim == 1:
        temp_sparse_tensor = random(dim[0], density=tensor_density, format=tensor_format, dtype=t)
    elif tensor_max_dim == 2:
        temp_sparse_tensor = random(dim[0], dim[1], density=tensor_density, format=tensor_format, dtype=t)
    else:
        temp_sparse_tensor = np.empty(dim, dtype=t)
        elements_higher_dim = 1
        for d in range(tensor_max_dim-1):
            elements_higher_dim *= dim[d]
        temp_sparse_tensor = random(elements_higher_dim, dim[tensor_max_dim-1], density=tensor_density, format=tensor_format, dtype=t)

    sparse_tensor = temp_sparse_tensor.A.reshape(dim)
    return sparse_tensor