import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

# Ensure we have at least 2 GPUs
if num_gpus >= 2:
    # Set the current device to GPU number 2 (index 1)
    torch.cuda.set_device(1)
    print(f"Current CUDA device set to: {torch.cuda.current_device()}")
else:
    print("Not enough GPUs available.")

# Example model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

# Instantiate the model
model = SimpleModel()

# Use DataParallel to parallelize the model across multiple GPUs
if num_gpus > 1:
    model = nn.DataParallel(model)
    print("Model is now using DataParallel.")

# Move the model to the GPU
model = model.cuda()

# Example input
input_data = Variable(torch.randn(64, 10)).cuda()

# Forward pass
output = model(input_data)
print("Output shape:", output.shape)