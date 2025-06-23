"""
DATA LOADING

DATA VISUALIZATION

MODEL DEFINITON

LOSS FUNCTION AND OPTIMIZATION

TRAIN + TEST

"""

# Libraries

import torch    
import torch.nn as nn   # defition of NN
import torch.optim as optim  # optimization algorithms for back propagation
import torchvision   # Image Processing - predefined models
import torchvision.transforms as transforms   # Image transforms
import matplotlib.pyplot as plt  



# option: definition of devices

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DATA LOADING

def get_data_loaders(batch_size=64):   # the amount of data that  will be processed in each iteration 

    transform=transforms.Compose([               # Makes the images ready for algorithm
    transforms.ToTensor(), # transfrom image to Tensor and scale it   0-1
    transforms.Normalize((0.5,),(0.5,))])   # scaling pixels to -1 - -1

    # mnist dataset + train-test + preparing the dataset 

    train_set=torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set=torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # pytorch data_loader and  mini batch creator- it has just been downloaded in the previous stage

    train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=batch_size, shuffle=False)

    return train_loader,test_loader

train_loader,test_loader=get_data_loaders()




    
# Data Visualizaton


def visualize_samples(loader,n):
    images,labels=next(iter(loader)) # Getting images and labels from the first batch
    fig, axes=plt.subplots(1,n, figsize=(10,5))

    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_title(f"Label : {labels[i].item()}")     # Item() is used for converting tensor number to python int
        axes[i].axis("off")
plt.show()


visualize_samples(train_loader, 4)






# Model Definition- ANN

class NeuralNetwork(nn.Module):    # Inherits from pytorch nn.module class

    def __init__(self):
        super(NeuralNetwork,self).__init__ () # # Initialize nn.Module to inherit its features

        # Images to vector : 1D

        self.flatten=nn.Flatten()

        # First fully connected layer
        self.fc1=nn.Linear(28*28, 128)   # 784=input size 128=output size
        
        # Activation function
        self.relu=nn.ReLU()

        # Second fully connected layer
        self.fc2=nn.Linear(128,64) # 128=input size  64=output size

        # Output layer
        self.fc3=nn.Linear(64,10)  #64 input 10=output(0-9 labels)


    def forward(self,x):  # x=image
        # Inıtal x= 28*28 image

        x=self.flatten(x)        # 1D
        x=self.fc1(x)         # F layer
        x=self.relu(x)       # Activation
        x=self.fc2(x)         # Second layer
        x=self.relu(x)       # activation
        x=self.fc3(x)         # Output
        return x
    

# Create Model and Compile
    
model=NeuralNetwork().to(device)




# Loss function and optimization

define_loss_and_optimizer=lambda model:(
    nn.CrossEntropyLoss(),  # Multi class classification loss
    optim.Adam(model.parameters(), lr=0.001)  # update weight with Adam
)

criterion, optimizer=define_loss_and_optimizer(model)





# Train 

def train_model(model,train_loader, criterion,optimizer,epochs=10):

    model.train()     # take the model into training mode
    train_losses=[]  # create list for loss values in each epoch
    for epoch in range(epochs): # Training for defined 
        total_loss=0  # Sum of loss value
        for images,labels in train_loader:  # Iteration in all training data
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad() # Zero the Gradients

            predictions=model(images) # Apply model, forward pro.
            loss=criterion(predictions,labels) # Loss calculatıon - ypred yreal
            loss.backward()   # Back pro. Gradian calculation
            optimizer.step()  # Update weights
            total_loss=total_loss + loss.item()
        avg_loss=total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, loss :{avg_loss:.3f}")

    #loss graph
    plt.figure()
    plt.plot(range(1,epochs+1), train_losses, marker="o",linestyle="-", label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()


train_model(model,train_loader, criterion, optimizer, epochs=5)

# Test

def test_model(model,test_loader):
    model.eval()   # Evaluation mode
    correct=0    #Correct pred counter
    total=0 # Total data counter

    with torch.no_grad(): # Unnecessary for test
        for images,labels in test_loader:  # Loop test datas
            images,labels=images.to(device), labels.to(device)
            predictions=model(images)
            _, predicted=torch.max(predictions,1)  # Valuue + Index (finding max likehood )
            total+= labels.size(0)  # Update the total data count
            correct+=(predicted==labels).sum().item()  # Counting the correct pred

    print(f"Test Accuracy : {100*correct/total:.3f}%")

test_model(model,test_loader)




