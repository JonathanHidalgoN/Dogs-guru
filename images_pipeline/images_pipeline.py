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
        self._species = self.get_species

    @property
    def species(self) ->list:
        """
        Returns a list of classes in the dataset.
        Returns:
        A list of strings representing the classes in the dataset.
        """
        species = os.listdir(self.path)
        return [specie.split('-')[-1] for specie in species]

    @species.setter
    def add_species(self, new_species : list) -> None:
        """
        Adds new classes to the dataset.
        Args:
            new_species: A list of strings representing the new classes to add.
        """
        self._species.extend(new_species)
        #Now make new folders for the new species
        #TO DO: check duplicates
        for new_specie in new_species:
            os.mkdir(os.path.join(self.path, new_specie))
                



if __name__ == "__main__":
    images_path = "images/Images"
    dataset = DogsDataset(path = images_path)
    print(dataset.get_species)
    new_species = ["new_specie_1", "new_specie_2"]
    dataset.add_species = new_species
    print(dataset.get_species[-2:])