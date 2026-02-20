# Program written by Gustavo Delgado

import torch                        # Include PyTorch Library
import torch.nn                     # Include Neural Network Module for building & training Neural Networks
import torch.nn.functional          # Includes functions such for Activation functions & Loss Functions
import torch.utils.data             # Includes tools for handling datasets & data loading for PyTorch
import torchvision                  # Includes tools & functions for computer vision tasks (loading datasets, transforming images, and pre-trained models)



def main(): 
    """
    
    Args: 

    Returns: 

    Raises: 
    
    """
    print("Starting the program")
    load_training_dataset()


############################ Training dataset #################################
## The torchvision.datasets contains pre-built dataset classes, where CIFAR-10 is one of them.
## Creating the dataset object, automatically downloads the data from the Internet & stores it in a data directory located in project root directory/

# torchvision.datasets.CIFAR10(root, train, )
def load_training_dataset(destination_folder, boolean, transform_type, download):
    """
    """
    training_dataset = torchvision.datasets.CIFAR10(
        root = destination_folder, 
        train = boolean, 
        transform = transform_type,
        download = download
    ) # training_dataset




############################ Testing dataset #################################
## The torchvision.datasets contains pre-built dataset classes, where CIFAR-10 is one of them.
## Creating the dataset object, automatically downloads the data from the Internet & stores it in a data directory located in project root directory/

def load_testing_dataset(): 
    testing_dataset = torchvision.datasets.CIFAR10( 


    ) # testing_dataset



# Invoke the main function
if __name__ == "__main__": 
    main()

