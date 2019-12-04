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

model = NeuralNetwork([784, 128, 64, 10])
#model = torch.load("MINST")

#image, target = trainset[0]

#plt.imshow(image.numpy()[0], cmap='gray')
#plt.show()

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        running_loss += model.train(images, labels)
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
        model.saveWeights("MNIST")

def image_test(num):
    image, target = trainset[num]
    
    print("Desired: " + str(target))
    print("Output: " + str(model.forward(image.view(image.shape[0], -1))))
    
    plt.imshow(image.numpy()[num], cmap='gray')
    plt.show()