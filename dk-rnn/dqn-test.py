import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the parameters
input_size = n  # Number of parameters for each player
output_size = 10  # Number of selected players

# Create an instance of the DQN model
model = DQN(input_size, output_size)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# # Define the DQN class
# class DQN(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, output_size)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Define the input and output training data
# input_data = np.random.rand(100, input_size)  # Replace with your input data
# output_data = np.random.rand(100, output_size)  # Replace with your output data

# # Convert the input and output data to PyTorch tensors
# input_tensor = torch.tensor(input_data, dtype=torch.float)
# output_tensor = torch.tensor(output_data, dtype=torch.float)

# # Define the training parameters
# num_epochs = 100
# batch_size = 16
# learning_rate = 0.001

# # Create an instance of the DQN model
# model = DQN(input_size, output_size)

# # Define the loss function
# criterion = nn.MSELoss()

# # Define the optimizer
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Training loop
# for epoch in range(num_epochs):
#     # Shuffle the data at the beginning of each epoch
#     indices = torch.randperm(input_tensor.size(0))
#     input_tensor = input_tensor[indices]
#     output_tensor = output_tensor[indices]

#     # Mini-batch training
#     for i in range(0, input_tensor.size(0), batch_size):
#         # Extract the mini-batch input and output
#         input_batch = input_tensor[i:i+batch_size]
#         output_batch = output_tensor[i:i+batch_size]

#         # Forward pass
#         output_pred = model(input_batch)

#         # Compute the loss
#         loss = criterion(output_pred, output_batch)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # Print the loss after each epoch
#     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
