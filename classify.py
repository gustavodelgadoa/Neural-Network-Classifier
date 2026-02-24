# Program written by Gustavo Delgado

import sys
import torch                                        # Include PyTorch Library
import torch.nn as nn                               # Include Neural Network Module for building & training Neural Networks
import torch.nn.functional as functional            # Includes functions such for Activation functions & Loss Functions
import torch.utils.data                             # Includes tools for handling datasets & data loading for PyTorch
import torchvision                                  # Includes tools & functions for computer vision tasks (loading datasets, transforming images, and pre-trained models)
import torchvision.transforms as transforms         # Includes image transformation unils
import torch.optim as optim                         # Includes optimization algorithms
from PIL import Image                               # Includes iamging library for reading images

# CIFAR10 class labels
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ----------------- Design network class -----------------------

class CIFAR10_Net(nn.Module): 
    """
    MLP for CIFAR-10 image classification. 

    """    
    def __init__(self): 
        super(CIFAR10_Net, self).__init__()

        self.fc1 = nn.Linear(3072, 512)             # Input Layer
        self.fc2 = nn.Linear(512, 256)              # Hidden Layer
        self.fc3 = nn.Linear(256, 10)               # Output Layer

    def forward(self, x):
        """
        Defines the forward pass of the Neural Network

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            torch.Tensor: raw class scores
        """        
        x = x.view(x.size(0), -1) # flattens
        
        x = self.fc1(x)             # First fully connected layer
        x = functional.relu(x)      # ReLu activation

        x = self.fc2(x)             # Second fully connected layer
        x = functional.relu(x)      # ReLu activation

        x = self.fc3(x)             # Output layer
        return x


# ----------------- Download & load datasets helper function -----------------------
def load_data(): 
    """
    Downloads & loads the CIFAR10 training and testing datasets. 

    Returns:
        tuple: tuple containing training and testing data loaders
    """    
     #normalized pixel values from 0, 1 to -1, 1
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_data = torchvision.datasets.CIFAR10(
        './data/CIFAR10_Training_Data',
        train = True,
        transform = transform,
        download = True
    )     # downloads training dataset
    
    testing_data = torchvision.datasets.CIFAR10(
        './data/CIFAR10_Testing_Data', 
        train = False, 
        transform = transform, 
        download = True
    )     # downloads testing dataset

    batch_size = 64
    training_data_loader = torch.utils.data.DataLoader(
        dataset = training_data, 
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = 0
    )     # splits training data into batches for model feed  
    
    testing_data_loader = torch.utils.data.DataLoader(
        dataset = testing_data, 
        batch_size = batch_size, 
        shuffle = False, 
        num_workers = 0
    )      # splits testing data into batches for model feed

    return training_data_loader, testing_data_loader


# ----------------- train function -----------------------

def train(): 
    """
    Trains the CIFAR10_Net model on the CIFAR10 training dataset.
    """    

    # Load the datasets
    training_data_loader, testing_data_loader = load_data()
    
    # Create the model, Define loss function, and Optimizer
    net = CIFAR10_Net() # creates network instance
    criterion = nn.CrossEntropyLoss() # cross entropy loss
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001) # adam optimizer

    print(f'{"Loop":<10} {"Train Loss":<15} {"Train Acc %":<15} {"Test Loss":<15} {"Test Acc %":<15}') # table headers for training CLI

    # Training loop
    number_of_epochs = 10
    for epoch in range(number_of_epochs): 
        net.train() # sets model to training mode
        running_loss = 0.0 # accumulates loss over batches
        training_correct = 0 # counts correct training predictions
        training_total = 0 # counts all training samples

        for batch_idx, (images, labels) in enumerate(training_data_loader): 
            optimizer.zero_grad() # clears gradients
            outputs = net(images) #  forward pass
            loss = criterion(outputs, labels) # computes cross rntropy loss
            loss.backward() # backwards pass
            optimizer.step() # updates weights using gradient

            running_loss += loss.item() # adds batch loss
            _, predicted = torch.max(outputs.data, 1) # gets predicted class
            training_total += labels.size(0) # adds batch size
            training_correct += (predicted == labels).sum().item() # counts correct

        
        average_train_loss = running_loss / len(training_data_loader)
        training_accuracy = 100 * training_correct / training_total
        testing_accuracy, testing_loss = test(net, testing_data_loader, criterion)

        # prints epoch results
        print(f'{epoch+1}/{number_of_epochs}           {average_train_loss:.4f}          {training_accuracy:.4f}        {testing_loss:.4f}          {testing_accuracy:.4f}')

        # resets accumulators
        running_loss = 0.0
        training_correct = 0
        training_total = 0
    save_model(net)

# ----------------- test function -----------------------
def test(model, testing_data_loader, criterion): 
    """
    Evaluates model on the CIFAR10 test dataset

    Args:
        model (CIFAR10_Net): the neural network to evaluate
        testing_data_loader (DataLoader): dataloader of test dataset
        criterion (nn.CrossEntropyLoss): loss function for computing test loss
=
    """    
    model.eval() # sets to evaluation mode
    correct = 0 
    total = 0
    testing_loss = 0

    with torch.no_grad(): 
        for images, labels in testing_data_loader:
            outputs = model(images) # forward pass
            loss = criterion(outputs, labels) # computes batch loss
            testing_loss += loss.item() # accumulates loss

            _, predicted = torch.max(outputs.data, 1) # gets predicted class index
            total += labels.size(0) # counts total samples
            correct += (predicted == labels).sum().item() # counts correct predictions

    average_testing_loss = testing_loss / len(testing_data_loader) # average loss per batch
    accuracy = 100 * correct / total
    return accuracy, average_testing_loss

# ----------------- save model function -----------------------
def save_model(model):
    """
    Saves the trained models state dict to model directory

    """    
    import os
    os.makedirs('model', exist_ok=True) # creates model folder if not exists
    path = 'model/model.pt'
    torch.save(model.state_dict(), path) # saves model weights
    print(f'Model saved in file: ./{path}')

# ----------------- predict function -----------------------
def predict(image_path): 
    """
    Loads trained model and predicts the class of a single input image
    """
    model = CIFAR10_Net()
    model.load_state_dict(torch.load('model/model.pt', weights_only=True))
    model.eval() 
    
    image = Image.open(image_path)
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), # resizes to CIFAR10 dimensions
         transforms.ToTensor(),  # converts to tensors
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # normalized RGB channels
    )
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad(): 
        output = model(image_tensor) # forward pass
        _, predicted = torch.max(output, 1) # get highest scoring class index
        pred_index = predicted.item() 

    # maps index to class name and prints
    predicted_class = classes[pred_index]
    print(f'prediction result: {predicted_class}')


# Invoke the main function
if __name__ == "__main__":  
    if len(sys.argv) < 2: 
        print("Usage: python classify.py train OR python classify.py test <image.png>")
        sys.exit(1)
    command = sys.argv[1]
    if command == "train": 
        train()
    elif command == "test" or command == "predict": 
        if len(sys.argv) != 3: 
            print("Usage: python classify.py test <image.png>")
            sys.exit(1)
        predict(sys.argv[2])
    else: 
        print("Unknown command, please try again")    

