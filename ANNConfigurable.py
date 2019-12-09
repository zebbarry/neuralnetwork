import torch
import torch.nn as nn


def networkBlock(inParam, outParam, dimen=1):
    
    if dimen == 2:
        result = nn.Sequential(
            nn.Conv2d(inParam, outParam, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outParam),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
    else:
        result = nn.Sequential(
            nn.Linear(inParam, outParam),
            nn.ReLU())
    
    return result


class NeuralNetwork(nn.Module):
    
    def __init__(self, networkDimen, lr=0.003, dimen=1):
        super(NeuralNetwork, self).__init__()
        # Network parameters defining number of nodes in each layer
        # networkDimensions = [3, 2, 1] - 3 input 2 hidden 1 output
        length = len(networkDimen)
        self.numLayers = length
        self.lrate = lr
        self.dimen = dimen
        self.hiddenList = nn.ModuleList()
        self.correct = 0
        self.estimations = 0
        self.accuracy = -1
        
        if dimen == 2:
            length = self.numLayers - 1
        
        for inDimen, outDimen in zip(networkDimen, networkDimen[1:length-1]):    
            self.hiddenList.append(networkBlock(inDimen, outDimen, dimen=dimen))
            
        # Add final layer
        end = nn.Sequential(
            nn.Linear(networkDimen[-2], networkDimen[-1]),
            nn.LogSoftmax(dim=1))
        self.hiddenList.append(end)
        
        # Combine into one network
        self.hidden = nn.Sequential(*self.hiddenList)
        
        # Define the loss
        self.criterion = nn.NLLLoss()
        # Optimizers require the parameters to optimize and a learning rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        
        if torch.cuda.is_available():
            self = self.cuda()
            self.criterion = self.criterion.cuda()
            
        
        
    def forward(self, inputVal):
        # Forward multiplication of input values to determine output
        if self.dimen == 2:
            conv = self.hidden[:-1](inputVal)
            linear = conv.view(conv.size(0), -1)
            result = self.hidden[-1](linear)
        else:
            result = self.hidden(inputVal)
        return result
        
        
    def train(self, inputValues, desiredValues):
        # Converting the data into GPU format
        if torch.cuda.is_available():
            inputValues = inputValues.cuda()
            desiredValues = desiredValues.cuda()
        # Reset gradients
        self.optimizer.zero_grad()
        # Forward pass
        output = self.forward(inputValues)
        # Calculate losses and asjust weights
        self.loss = self.criterion(output, desiredValues)
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()
        
        
    def saveWeights(self, path):
        # Saves network using PyTorch internal storage
        torch.save(self, path)
        
        
    def loadWeights(self, path):
        # Loads network using PyTorch internal storage
        return torch.load(path)    
        
        
    def predict(self, inputVal, desiredVal=None):
        # Scale input and predict output
        print(inputVal.size())
        output = self.forward(inputVal).exp()
        
        if desiredVal:
            print("Desired: " + str(desiredVal))
            
        print("Output: ")
        for i in range(output.size()[1]):
            print("#{}: {:.3f}".format(i, output.squeeze().detach().numpy()[i]))
            
        return output
        
        

if __name__ == "__main__":
    x = torch.tensor(([2.0, 9], [1.0, 5.0], [3.0, 6.0]), dtype=torch.float)   # 3 x 2 tensor
    y = torch.tensor(([92.0], [100.0], [89.0]), dtype=torch.float)        # 3 x 1 tensor
    xPredicted = torch.tensor(([4.0, 8.0]), dtype=torch.float)          # 1 x 2 tensor
    
    # Scale units
    xMax, _ = torch.max(x, 0)
    xPredictedMax, _ = torch.max(xPredicted, 0)
    yMax, _ = torch.max(y, 0)
    
    x = torch.div(x, xMax)
    xPredicted = torch.div(xPredicted, xPredictedMax)
    y = torch.div(y, yMax)
    
    NN = NeuralNetwork([2, 3, 1])
    
    
    
    # Train
    iterations = 1000   # Number of training iterations
    for i in range(1000):
        loss = NN.train(x.float(), y)
        print("#" + str(i) + " Loss: " + str(loss))
    
    #NN.saveWeights(NN)
    #NN = torch.load("NN")
    NN.predict(xPredicted)