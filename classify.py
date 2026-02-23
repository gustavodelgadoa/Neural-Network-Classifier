# Program written by Gustavo Delgado

import torch                                        # Include PyTorch Library
import torch.nn as nn                               # Include Neural Network Module for building & training Neural Networks
import torch.nn.functional as functional            # Includes functions such for Activation functions & Loss Functions
import torch.utils.data                             # Includes tools for handling datasets & data loading for PyTorch
import torchvision                                  # Includes tools & functions for computer vision tasks (loading datasets, transforming images, and pre-trained models)



def main(): 

    print("Starting the program")

    # ----------------- Download data -----------------------
    
    training_data = download_dataset('./data/CIFAR10_Training_Data', True, torchvision.transforms.ToTensor(), True)     # downloads training dataset
    testing_data = download_dataset('./data/CIFAR10_Testing_Data', False, torchvision.transforms.ToTensor(), True)     # downloads testing dataset
    
    # ----------------- Load data ---------------------------

    training_data_loader = load_data(training_data, 64, True, 0, False)     # splits training data into batches for model feed  
    testing_data_loader = load_data(testing_data, 64, False, 0, False)      # splits testing data into batches for model feed

    # ----------------- designing network class -----------------------

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

    net = CIFAR10_Net()
    print(net)

    # ----------------- try a random 32 x 32 input -----------------------
    test_input = torch.randn(1, 3, 32, 32)
    output = net(test_input)
    print(output.shape)

    # ----------------- Loss Fucntion -----------------------
    loss_function = nn.CrossEntropyLoss()
    print(loss_function)


    
    optimzier = ""
    print("Ending the program")


def download_dataset(destination_folder, data_set, transform_type, download):
    """
     Creating the dataset object, automatically downloads the data from the Internet & stores it in a data directory located in project root directory/


    Args:
        destination_folder (String): Location for dataset to be stored
        data_set (boolean): True to get training dataset, False to get test dataset
        transform_type (function): Converts PIL/numpy to torch.FloatTensor of shape (C x H x W)
        download (boolean): True will download data if not exists at root, False assumes data exists (error if not exists)
    """    
    download_dataset = torchvision.datasets.CIFAR10(
        root = destination_folder, 
        train = data_set, 
        transform = transform_type,
        download = download
    ) # download_dataset
    return download_dataset

def load_data(dataset_name, batch_size_amount, shuffle_order, number_of_workers, drop_last_batches):
    """
    Will split dataset objects into smaller chunks (batches), handles shuffling, parallel loading, and iteration automatically.

    Args:
        dataset_name (String): CIFAR10 object
        batch_size_amount (integer): number of images to load at once
        shuffle_order (boolean): True for randomizing order, False will not randomize
        number_of_workers (integer): number of CPU processes to use for loading data
        drop_last_batches (boolean): True will drop incomplete batch, False will keep batch (Total images / batch_size_amount = full batches + 1 incomplete batch)
    """    
    load_data = torch.utils.data.DataLoader(
        dataset = dataset_name, 
        batch_size = batch_size_amount,
        shuffle = shuffle_order, 
        num_workers = number_of_workers, 
        drop_last = drop_last_batches
    ) # load_data 
    return load_data



# Invoke the main function
if __name__ == "__main__":  
    main()

