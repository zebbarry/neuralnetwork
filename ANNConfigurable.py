import torch
import torch.nn as nn


def net_block(inParam, outParam):
    return nn.Sequential(
        nn.Linear(inParam, outParam),
        nn.Sigmoid())


class NeuralNetwork(nn.Module):
    
    def __init__(self, networkDimen, lrate=0.003):
        super(NeuralNetwork, self).__init__()
        # Network parameters defining number of nodes in each layer
        # networkDimensions = [3, 2, 1] - 3 input 2 hidden 1 output
        length = len(networkDimen)
        self.numLayers = length
        self.lrate = lrate
        self.hiddenList = nn.ModuleList()
        
        for inDimen, outDimen in zip(networkDimen, networkDimen[1:length-1]):    
            self.hiddenList.append(net_block(inDimen, outDimen))
            
        # Add final layer
        end = nn.Sequential(
            nn.Linear(networkDimen[length-2], networkDimen[length-1]),
            nn.LogSoftmax(dim=1))
        self.hiddenList.append(end)
        
        # Combine into one network
        self.hidden = nn.Sequential(*self.hiddenList)
        
        # Define the loss
        self.criterion = nn.L1Loss()
        # Optimizers require the parameters to optimize and a learning rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lrate)        
        
        
    def forward(self, inputVal):
        # Forward multiplication of input values to determine output
        return self.hidden(inputVal)
        
        
    def train(self, inputValues, desiredValues):
        # Reset gradients
        self.optimizer.zero_grad()
        # Forward pass
        output = self.forward(inputValues)
        # Calculate losses and asjust weights
        print(self.hidden[0][0].weight)
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
        output = self.forward(inputVal).exp()
        
        print("Desired: " + str(desiredVal))
        print("Output: ")
        for i in range(output.size()[1]):
            print("#{}: {:.3f}".format(i, output.squeeze().detach().numpy()[i]))
            
        return output
        
        

if __name__ == "__main__":
    x = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float)   # 3 x 2 tensor
    y = torch.tensor(([92], [100], [89]), dtype=torch.float)        # 3 x 1 tensor
    xPredicted = torch.tensor(([4, 8]), dtype=torch.float)          # 1 x 2 tensor
    
    # Scale units
    #xMax, _ = torch.max(x, 0)
    #xPredictedMax, _ = torch.max(xPredicted, 0)
    #yMax, _ = torch.max(y, 0)
    
    #x = torch.div(x, xMax)
    #xPredicted = torch.div(xPredicted, xPredictedMax)
    #y = torch.div(y, yMax)
    
    NN = NeuralNetwork([2, 3, 1])
    
    
    
    # Train
    iterations = 1000   # Number of training iterations
    for i in range(1000):
        loss = NN.train(x.float(), y)
        print("#" + str(i) + " Loss: " + str(loss))
    
    #NN.saveWeights(NN)
    #NN = torch.load("NN")
    NN.predict(xPredicted)