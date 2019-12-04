import torch
import torch.nn as nn


def net_block(inParam, outParam):
    return nn.Sequential(
        nn.Linear(inParam, outParam),
        nn.ReLU())


class NeuralNetwork(nn.Module):
    
    def __init__(self, networkDimen, lrate=0.003):
        super(NeuralNetwork, self).__init__()
        # Network parameters defining number of nodes in each layer
        # networkDimensions = [3, 2, 1] - 3 input 2 hidden 1 output
        length = len(networkDimen)
        self.numLayers = length
        self.lrate = lrate
        self.hiddenList = []
        
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
        self.criterion = nn.NLLLoss()
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
        self.loss = self.criterion(output, desiredValues)
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()
        
        
    def saveWeights(self, network):
        # Saves network using PyTorch internal storage
        torch.save(network, "NN")
        # Can be loaded using:
        # torch.load("NN")
        
        
    def predict(self, inputVal):
        # Scale input and predict output
        inputMax, _ = torch.max(inputVal, 0)
        inputVal = torch.div(inputVal, inputMax)
        output = self.forward(inputVal)
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(inputVal))
        print("Output: \n" + str(output))
        return output
        
        

if __name__ == "__main__":
    x = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float)   # 3 x 2 tensor
    y = torch.tensor(([92], [100], [89]), dtype=torch.float)        # 3 x 1 tensor
    xPredicted = torch.tensor(([4, 8]), dtype=torch.float)          # 1 x 2 tensor
    
    # Scale units
    xMax, _ = torch.max(x, 0)
    xPredictedMax, _ = torch.max(xPredicted, 0)
    yMax, _ = torch.max(y, 0)
    
    x = torch.div(x, xMax)
    xPredicted = torch.div(xPredicted, xPredictedMax)
    y = torch.div(y, yMax)
    
    NN = NeuralNetwork([2, 3, 1])
    
    
    
    # Train
    #iterations = 1000   # Number of training iterations
    #for i in range(1000):
        #print("#" + str(i) + " Loss: " + str(torch.mean((y - NN(x)) ** 2).detach().item()))
        #NN.train(x, y)
    
    #NN.saveWeights(NN)
    #NN = torch.load("NN")
    NN.predict(xPredicted)