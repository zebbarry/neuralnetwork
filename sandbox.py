from ANNConfigurable import *

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#model = nn.Sequential(nn.Linear(784, 128),
                      #nn.ReLU(),
                      #nn.Linear(128, 64),
                      #nn.ReLU(),
                      #nn.Linear(64, 10),
                      #nn.LogSoftmax(dim=1))

model = NeuralNetwork([784, 256, 128, 64, 10])

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        running_loss += model.train(images, labels)
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

#model.saveWeights("model")
#model = torch.load("model")

def image_test(num):
    image, target = trainset[num]
    output = model.forward(image.view(image.shape[0], -1)).exp()

    xAxis = range(output.size()[1])
    yAxis = output.squeeze().detach().numpy()
    
    print("Desired: " + str(target))
    print("Output: ")
    for i in range(output.size()[1]):
        print("#{}: {:.3f}".format(i, yAxis[i]))    
    
    plt.figure(1)
    plt.bar(xAxis, yAxis)
    plt.figure(2)
    plt.imshow(image.numpy()[0], cmap='gray')
    plt.show()
        
    return output
    
    
    
image_test(0)