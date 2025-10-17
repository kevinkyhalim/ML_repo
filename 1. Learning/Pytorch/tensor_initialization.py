import torch

print(torch.__version__)

# check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# # Example: Tensor operations
# x = torch.rand(3, 3).to(device)
# y = torch.rand(3, 3).to(device)
# z = x + y
# print(z)

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initialization methods
# empty is uninitialized data,not necessarily zeroes
x = torch.empty(size = (3, 3))
print(x)
x = torch.zeros((3, 3))
print(x)
# generate from uniform distribution
x = torch.rand((3, 3))
print(x)
# full of 1s
x = torch.ones((3, 3))
print(x)
# Identity matrix
x = torch.eye(5, 5)
print(x)
# range
x = torch.arange(start=0, end = 5, step = 1)
print(x)
# lin space
x = torch.linspace(start = 0.1, end = 1, steps = 10)
print(x)
# generate matrix and populate with normal distribution
x = torch.empty(size=(1, 5)).normal_(mean=0, std = 1)
print(x)
# create 3x3 diagonal matrix with 1s in the diagonal
x = torch.diag(torch.ones(3))
print(x)

# HOw to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool()) # will work regardless of using CPU or GPU
print(tensor.short()) # to int16
print(tensor.long()) # to int64
print(tensor.half()) # to float16
print(tensor.float()) # to float32
print(tensor.double()) # to float64

# Array to Tensor conversion and vice versa
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()