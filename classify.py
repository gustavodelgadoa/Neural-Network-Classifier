# Program written by Gustavo Delgado

import torch                        # Include PyTorch Library
import torch.nn                     # Include Neural Network Module for building & training Neural Networks
import torch.nn.functional          # Includes functions such for Activation functions & Loss Functions
import torch.utils.data             # Includes tools for handling datasets & data loading for PyTorch
import torchvision                  # Includes tools & functions for computer vision tasks (loading datasets, transforming images, and pre-trained models)



def main(): 

    print("Starting the program")
    load_dataset('./data.CIFAR10_Training_Data', True, torchvision.transforms.ToTensor(), True)     # Loads training dataset
    load_dataset('./data.CIFAR10_Testing_Data', False, torchvision.transforms.ToTensor(), True)     # loads testing dataset



def load_dataset(destination_folder, data_set, transform_type, download):
    """
     Creating the dataset object, automatically downloads the data from the Internet & stores it in a data directory located in project root directory/


    Args:
        destination_folder (String): Location for dataset to be stored
        data_set (boolean): True to get training dataset, False to get test dataset
        transform_type (function): Converts PIL/numpy to torch.FloatTensor of shape (C x H x W)
        download (boolean): True will download data if not exists at root, False assumes data exists (error if not exists)
    """    
    training_dataset = torchvision.datasets.CIFAR10(
        root = destination_folder, 
        train = data_set, 
        transform = transform_type,
        download = download
    ) # training_dataset
    return training_dataset



# Invoke the main function
if __name__ == "__main__": 
    main()

