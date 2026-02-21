# Program written by Gustavo Delgado

import torch                        # Include PyTorch Library
import torch.nn                     # Include Neural Network Module for building & training Neural Networks
import torch.nn.functional          # Includes functions such for Activation functions & Loss Functions
import torch.utils.data             # Includes tools for handling datasets & data loading for PyTorch
import torchvision                  # Includes tools & functions for computer vision tasks (loading datasets, transforming images, and pre-trained models)



def main(): 

    print("Starting the program")
    download_dataset('/data/CIFAR10_Training_Data', True, torchvision.transforms.ToTensor(), True)     # downloads training dataset
    download_dataset('/data/CIFAR10_Testing_Data', False, torchvision.transforms.ToTensor(), True)     # downloads testing dataset



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

