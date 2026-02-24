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
    def __init__(self): 
        super(CIFAR10_Net, self).__init__()

        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = functional.relu(x)

        x = self.fc2(x)
        x = functional.relu(x)

        x = self.fc3(x)
        return x


# ----------------- Download & load datasets helper function -----------------------
def load_data(): 
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


# ----------------- train helper function -----------------------

def train(): 

    # Load the datasets
    training_data_loader, testing_data_loader = load_data()
    
    # Create the model, Define loss function, and Optimizer
    net = CIFAR10_Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    print(f'{"Loop":<10} {"Train Loss":<15} {"Train Acc %":<15} {"Test Loss":<15} {"Test Acc %":<15}')

    # Training loop
    number_of_epochs = 10
    for epoch in range(number_of_epochs): 
        net.train()
        running_loss = 0.0
        training_correct = 0
        training_total = 0

        for batch_idx, (images, labels) in enumerate(training_data_loader): 
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            training_total += labels.size(0)
            training_correct += (predicted == labels).sum().item()

        
        average_train_loss = running_loss / len(training_data_loader)
        training_accuracy = 100 * training_correct / training_total
        testing_accuracy, testing_loss = test(net, testing_data_loader, criterion)

        print(f'{epoch+1}/{number_of_epochs}           {average_train_loss:.4f}          {training_accuracy:.4f}        {testing_loss:.4f}          {testing_accuracy:.4f}')
        running_loss = 0.0
        training_correct = 0
        training_total = 0
    save_model(net)


def test(model, testing_data_loader, criterion): 
    model.eval()
    correct = 0
    total = 0
    testing_loss = 0

    with torch.no_grad(): 
        for images, labels in testing_data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            testing_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    average_testing_loss = testing_loss / len(testing_data_loader)
    accuracy = 100 * correct / total
    return accuracy, average_testing_loss

def save_model(model):
    import os
    os.makedirs('model', exist_ok=True)
    path = 'model/model.pt'
    torch.save(model.state_dict(), path)
    print(f'Model saved in file: ./{path}')

def predict(image_path): 
    model = CIFAR10_Net()
    model.load_state_dict(torch.load('model/model.pt', weights_only=True))
    model.eval() 
    
    image = Image.open(image_path)
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), 
         transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad(): 
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        pred_index = predicted.item()

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

