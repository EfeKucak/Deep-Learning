"""Pipeline Overview:
1. Load and preprocess CIFAR-10 dataset (Normalization + DataLoader)
2. Visualize sample images from training set
3. Define CNN model with Conv2D, ReLU, MaxPooling, Dropout, and Fully Connected layers
4. Set up loss function (CrossEntropy) and optimizer (SGD with momentum)
5. Train model using forward pass, loss computation, backpropagation, and optimization
6. Evaluate model accuracy on test and training datasets"""




# Import Libraries
import torch
import torch.nn as nn   # For NN layers
import torch.optim as optim  # Optimization
import torch.utils.data.dataloader
import torchvision  # Image processing
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# Load Dataset

def get_data_loader(batch_size=64):

    transform=transforms.Compose([
        transforms.ToTensor(),  # Convert Image to tensor
        transforms.Normalize(((0.5,0.5,0.5)),(0.5,0.5,0.5 ))    # Normalize [-1,1] + Since the images have 3 colors(RGB)
       ])

    # Download CIFAR10 and create training set
    train_set=torchvision.datasets.CIFAR10(root="./datacifar", train=True,download=True, transform=transform)
    test_set=torchvision.datasets.CIFAR10(root="./datacifar", train=False, download=True, transform=transform)

    # Data Loader:
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True) 
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)

    return train_loader,test_loader






# Data Visualization

def imshow(img):
    img = img / 2 + 0.5  # Denormalize the image to see the inage properly
    npimg = img.numpy()  # Convert tensor to NumPy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # # Convert shape from (C, H, W) to (H, W, C)  - imshows expects H W C 
    plt.axis("off")




def visualize(n=3):
    train_loader, _ = get_data_loader()
    images, labels = next(iter(train_loader))

    plt.figure(figsize=(n * 2, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)      # 1 row, n columns 
        imshow(images[i])             # # Show each image
        plt.title(f"Label: {labels[i].item()}")
    plt.tight_layout()
    plt.show()

visualize()






# Build CNN Model

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()

        self.conv1=nn.Conv2d(3, 32, kernel_size=3, padding=1) #in_channels=rgb3, out_channels= amount of filter, kernel_size=3x3  ----- output 32 
        self.relu=nn.ReLU()    # Activation func. -------------  outpu 32
        self.pool=nn.MaxPool2d(kernel_size=2 , stride=2)  # Max pooling with 2x2 window, halves image size , enable to keep important information ----- output 16    ( if we set stride =1 , the dimension would be same)
        self.conv2=nn.Conv2d(32,64,kernel_size=3, padding=1)  # 64 filtre second conv layer  ---- output 16 
        self.dropout=nn.Dropout(0.2)   # Dropout %20 
        self.fc1=nn.Linear(64*8*8,128)   # fully connected layer 64 = filtersize , 8x8 =image size rightafter 2 layers   ------  after flatten output 128 
        self.fc2=nn.Linear(128,10)  # Output layer: 10 classes for classification  ---- output 10 

    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x)))  # Conv1 → ReLU → MaxPool - First
        x=self.pool(self.relu(self.conv2(x)))   # Conv2 → ReLU → MaxPool- Second
        x=x.view(-1,64*8*8) #  Flatten to 1D tensor
        x=self.dropout(self.relu(self.fc1(x)))  # Fully connected
        x=self.fc2(x)  # Output
        return x
    
"""Input:       (3×32×32)
↓ Conv1+ReLU+Pool → (32×16×16)        
↓ Conv2+ReLU+Pool → (64×8×8)
↓ Flatten           → (4096)
↓ Fully Connected   → (128)
↓ Dropout
↓ Output Layer      → (10)"""






# MODEL DEFINITON
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CNN().to(device)


# DEFINE LOSS FUNCTION AND OPTIMIZER
define_loss_and_optimizer=lambda model:(
   nn.CrossEntropyLoss(),
   optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # stoschastic gradient descent 
)








# TRAINING



def train_model(model,train_loader, criterion, optimizer,epochs=5):
    
    model.train()  # Training mode
    train_losses=[]# List for loss values
    for epoch in range(epochs):# For loop based on epoch
        total_loss=0# Total loss variable
        for  images,labels in train_loader:# for loop to scan entire dataset
            images,labels=images.to(device), labels.to(device)
            optimizer.zero_grad() # zero gradient
            output=model(images) # forward pro. ( model prediction)
            loss=criterion(output,labels)# loss value 
            loss.backward() # back propp. (calculate grad)
            optimizer.step()  #learning = optimize the parameters
            total_loss+=loss.item()
        
        avg_loss=total_loss/ len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epocj: {epoch+1}/{epochs}, Loss:{avg_loss:.5f}")

    # loss graph     
    plt.figure()
    plt.plot(range(1,epochs+1), train_losses, marker="o", linestyle="-", label="Train Losses")   
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
plt.show()



train_loader,test_loader=get_data_loader()
model=CNN().to(device)
criterion, optimizer=define_loss_and_optimizer(model)

train_model(model,train_loader, criterion, optimizer,epochs=10)












# TEST

def test_model(model, test_loader, dataset_type):
    model.eval()
    correct=0 # Correct prediction
    total=0 # total data 
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels=images.to(device), labels.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs,1)
            total += labels.size(0)  # total data amount
            correct += (predicted==labels).sum().item()
        print(f"Test Accuracy :{100* correct/total} %")

test_model(test_loader, dataset_type="test")
test_model(train_loader,dataset_type="training")