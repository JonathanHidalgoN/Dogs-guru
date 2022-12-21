#This file assumes that the dataset is the following https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

import os

class DogsDataset:

    """
    A class to represent the Stanford Dogs Dataset.
    Attributes:
        path: A string representing the path to the dataset.
    """

    def __init__(self, path : str) -> None:

        self.path = path

    @property
    def get_species(self) ->list:
            """
            Returns a list of classes in the dataset.
            """
            species = os.listdir(self.path)
            return [specie.split('-')[-1] for specie in species]

    
    


if __name__ == "__main__":
    images_path = "images/Images"
    dataset = DogsDataset(path = images_path)
    print(dataset.get_species)
