import torch
import torch.nn as nn
import torch.optim as optim

class PlayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PlayerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.rnn(x)
        value = torch.sigmoid(self.fc(hidden[-1]))
        return value

# Define the input and hidden size
input_size = 10  # Number of input features (averagePoints, averageRebounds, averageAssists, price, player1, player2, player3, player4, player5, isStarting)
hidden_size = 64

# Instantiate the PlayerRNN model
player_rnn = PlayerRNN(input_size, hidden_size)

# Define the optimizer and loss function
optimizer = optim.Adam(player_rnn.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Generate a sample input tensor for demonstration
input_data = torch.tensor([[[10.0, 5.0, 3.0, 10000.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                            [8.0, 6.0, 4.0, 8000.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]]])

# Perform a forward pass
player_value = player_rnn(input_data)

# Compute the loss
target_value = torch.tensor([[1.0], [0.5]])  # Sample target values

loss = loss_fn(player_value, target_value)

# Perform backpropagation and update weights
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Print the model architecture
print(player_rnn)

# To train the above RNN, you can follow these general steps:

# Prepare your training dataset: 
# Collect or generate a dataset that consists of input sequences 
# (player features and team information) and 
# corresponding target values (player value rankings).
#  Ensure that your dataset covers a variety of scenarios and 
# reflects the desired constraints 
# (e.g., total price under 50,000, no more than 5 players from the same team).

# Preprocess the data: 
# Normalize the input features and scale the target values if necessary. 
# You can use techniques such as min-max scaling or standardization.

# Split the dataset: 
# Divide your dataset into training and validation sets. 
# The training set will be used to update the model's parameters, 
# while the validation set will be used to monitor the model's 
# performance and prevent overfitting.

# Define the loss function and optimizer: Choose an appropriate loss function for regression, such as mean squared error (MSE) loss. Select an optimizer, such as Adam or stochastic gradient descent (SGD), to update the model's parameters during training.

# Initialize the RNN model: Create an instance of the PlayerRNN model with appropriate input and hidden dimensions.

# Training loop: Iterate over your training dataset for a specified number of epochs. For each epoch, perform the following steps:

# Reset the gradients of the model's parameters.
# Pass the input sequences through the RNN model to obtain the predicted player value rankings.
# Calculate the loss between the predicted values and the target values.
# Backpropagate the gradients through time and update the model's parameters using the chosen optimizer.
# Optionally, monitor and log the training progress (e.g., loss values) to track the model's performance.
# Validation: After each epoch or at regular intervals, evaluate the model on the validation set. Calculate the validation loss and other relevant metrics to assess the model's performance and determine if it is overfitting.

# Hyperparameter tuning: Experiment with different hyperparameters, such as learning rate, hidden size, number of layers, and batch size. Adjust these hyperparameters based on the validation results to improve the model's performance.

# Test the model: Once training is complete, evaluate the trained model on a separate test set or real-world data to assess its performance in predicting player value rankings.

# It's important to note that the specific implementation details of training, including the number of epochs, batch size, and hyperparameter values, may vary depending on your specific dataset and problem. Experimentation and fine-tuning based on the training and validation results are key to achieving optimal performance.