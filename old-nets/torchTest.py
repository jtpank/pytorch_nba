import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torchvision import transforms

dtype = torch.float
#device = torch.device("cpu")
device = torch.device("cuda:0")


learning_rate = 1e-6



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #define fully connected layers to neural net
        #fc1 = fullyconnect<layernumber>
        # nn.Linear(input, output)
        # Linear is flat, fully connected network
        #NBA stats ~16 to 17 data points (w/ w/o team data) 
        #note: this is data PER PLAYER PER GAME
        #lets make 3 layers of x neurons
        self.fc1 = nn.Linear(19, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,128)
        #output layer
        self.fc4 = nn.Linear(128,2)

    def forward(self, x):
        #pass through layers, and use activation function over entire layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #if we make a classifier, change to 
        #F.log_softmax() for a distribution
        x = F.relu(self.fc4(x))
        return x


#ctrl+shift+a is block comment
""" #Sending data through
net = Net()
#print(net)
X = torch.rand(16,1)
output = net(X.flatten())
print(output)
X = X.view(-1,16)
output2 = net(X)
print(output2) """
net = Net()
print(net)
#optimizer (params that are adjustable, learning rate, )
#learning rate kind of dictates the size / step to get to the 
# local / (hopefully global) minimum
#must decay the learning rate**

optimizer = optim.Adam(net.parameters(), lr=0.00001)

#full passes through our dataset
EPOCHS = 125
""" trainset = [
    ([X1],[y1]),
    ([X2],[y2]),
    ...
] 
for Xi in R^Dx (R is real numbers, D is dimension)
Dx = 16
and yi in R^Dy
Dy = 6
"""

trainset = [
([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88, 102.2], [10, 37.5]),
([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88, 103.8], [8, 33.3]),   
([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88, 103.2], [8, 50.0]),  
([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88, 105.3], [7, 16.7]),    
([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88, 106.3], [5, 28.6]),    
([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88, 104.6], [7, 40.0]),  
([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88, 101.8], [7, 36.4]),   
([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88, 101.9], [10, 40.0]),
([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88, 103.0], [11, 42.9]),  
([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88, 104.4], [7,33.3]),
    
]
#past fantasy points https://www.numberfire.com/nba/players/daily-fantasy/lebron-james
""" t2 = [
    ([, 102.2], [44.4]),
([, 105.3], [40.9]),    
([, 106.3], [30.5]),    
([, 104.6], [37.9]),  
([, 101.8], [54.0]),   
([, 101.9], [56.5]),
([, 103.0], [41.7]),  
([, 104.4], [49.8])
]
 """

for epoch in range(EPOCHS):
    for data in trainset:
        #data is a batch of feature sets
        #unpack the tuple of data
        #x = torch.tensor([8.09, 0.5, 7.63])
        x, y = data
        #batch data to decrease training time
        #do not want to pass entire dataset due to overfitting issue
        x = torch.tensor(x)
        y = torch.tensor(y)
        net.zero_grad()
        #-1 is any batch size (could pass batch size)
        output = net(x.flatten())
        #how wrong were we? calculate loss
        #2 main ways to calculate loss:
        # 1. 1 hot vector output (array where a value is on e.g. [0,1,0])
        # -->use mean squared error
        # 2.  use nll_loss
        loss = F.mse_loss(output, y)
        #now backpropogate the loss
        #pytorch does this for us
        loss.backward()
        #this adjusts the weights for us
        optimizer.step()
    print(loss)

print("blah")
print(net.forward(torch.tensor([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88,107.5])))
print(net.forward(torch.tensor([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88,102.4])))
print(net.forward(torch.tensor([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88,105.1])))
print(net.forward(torch.tensor([8.09, 0.5, 7.63, 0.355, 6.78, 2.41, 0.501, 19.28, 9.66, 0.704, 5.81, 4.09, 0.56, 1.63, 25.81, 8.19, 1.0, 3.88,101.8])))