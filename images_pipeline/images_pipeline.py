#This file assumes that the dataset is the following https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

import os

class DogsDataset:
    #TO DO: maybe add private attributes for paths of subfolders, because I use them a lot
    """
    A class to represent the Stanford Dogs Dataset.
    Attributes:
        path: A string representing the path to the dataset.
    """

    def __init__(self, path : str) -> None:

        self.path = path
        self._species = self.species

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
            try :
                os.mkdir(os.path.join(self.path, new_specie))
            except FileExistsError:
                print(f"Folder {new_specie} already exists")

    
    def remove_species(self, species_to_remove : list) -> None:
        """
        Removes classes from the dataset.
        Args:
            species_to_remove: A list of strings representing the classes to remove.
        """
        #TO DO: check if the species to remove are in the dataset
        #TO DO: add a confirmation message, or a safety check
        index =[]
        dirs = os.listdir(self.path)
        for idx,specie in enumerate(dirs):
            if specie.split("-")[-1] in species_to_remove:
                index.append(idx)
            else :
                print(f"Specie {specie} not found")
        for idx in index:
            os.rmdir(os.path.join(self.path, dirs[idx]))

    def count_images(self) -> dict:
        """
        Counts the number of images in each class.
        Returns:
            A dictionary with the classes as keys and the number of images as values.
        """
        sub_paths = os.listdir(self.path)
        count = {sub_path.split('-')[-1]:len(os.listdir(os.path.join(self.path, sub_path))) for sub_path in sub_paths}
        return count
        
    



if __name__ == "__main__":
    images_path = "images/Images"
    dataset = DogsDataset(path = images_path)
    count = dataset.count_images()
    print(count)    