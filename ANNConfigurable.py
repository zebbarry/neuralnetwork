import torch
import torch.nn as nn

class Neural_Network(nn.Module):
    
    def __init__(self, networkDimen):
        super(Neural_Network, self).__init__()
        # Network parameters defining number of nodes in each layer
        # netwrokDimensions = [3, 2, 1] - 3 input 2 hidden 1 output
        self.layers = len(networkDimen)
        self.hidden = nn.ModuleList()
        self.hiddenVal = []
        
        for i in range(self.layers):
            self.hidden.append(nn.Linear(networkDimen[i], networkDimen[i+1]))
        
        # Weights
        #self.w1 = torch.randn(self.inputSize, self.hiddenSize)
        #self.w2 = torch.randn(self.hiddenSize, self.outputSize)
        
        
    def forward(self, inputVal):
        # Forward multiplication of input values to determine output
        x = inputVal
        for i in range(self.layers):
            x = self.hidden[i](x)
            self.hiddenVal.append(x)
            
        return self.hiddenVal[self.layers]
    
    
    def sigmoid(self, x):
        # Activation function, return value between 0-1
        return 1 / (1 + torch.exp(-x))
    
    
    def sigmoidPrime(self, x):
        # Derivative of sigmoid
        return x * (1 - x)
    
    
    def backward(self, inputVal, desiredVal, outputVal):
        # Backpropogation used to determine error and adjust weights
        self.outputError = desiredVal - outputVal
        self.outputDelta = self.outputError * self.sigmoidPrime(outputVal)
        
        # Determine error in hidden values
        self.hiddenError = torch.matmul(self.outputDelta, torch.t(self.w2))
        self.hiddenDelta = self.hiddenError * self.sigmoidPrime(self.hiddenVal)
        
        # Adjust weights based on error for each tensor
        self.w1 += torch.matmul(torch.t(inputVal), self.hiddenDelta)
        self.w2 += torch.matmul(torch.t(self.hiddenVal), self.outputDelta)
        
        
    def train(self, inputVal, desiredVal):
        # Forward and backward pass for training
        output = self.forward(inputVal)
        self.backward(inputVal, desiredVal, output)
        
        
    def saveWeights(self, network):
        # Saves network using PyTorch internal storage
        torch.save(network, "NN")
        # Can be loaded using:
        # torch.load("NN")
        
        
    def predict(self, inputVal):
        # Scale input and predict output
        inputMax, _ = torch.max(inputVal, 0)
        inputVal = torch.div(inputVal, inputMax)
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(inputVal))
        print("Output: \n" + str(self.forward(inputVal)))
        
        

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
    
    NN = Neural_Network([2, 3, 1])
    
    # Train
    #iterations = 1000   # Number of training iterations
    #for i in range(1000):
        #print("#" + str(i) + " Loss: " + str(torch.mean((y - NN(x)) ** 2).detach().item()))
        #NN.train(x, y)
    
    #NN.saveWeights(NN)
    #NN = torch.load("NN")
    NN.predict(xPredicted)