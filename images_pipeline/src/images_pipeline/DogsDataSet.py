#This file assumes that the dataset is the following https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset
#other datasets might work, if they are structured in the same way
#images/Images
#    n02085620-Chihuahua(Folder with images)
#    n02085782-Japanese_spaniel(Folder with images)
#    ...
#Notice specie have a - before the name, and the name is in the folder.

import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch import Tensor as torch_tensor
from google_images_download import google_images_download
from shutil import rmtree 

class DogsDataSet(Dataset):
    #TO DO: maybe add private attributes for paths of subfolders, because I use them a lot
    """
    A class to represent the Stanford Dogs Dataset.
    Attributes:
        path: A string representing the path to the dataset.
        species: A list of strings representing the classes in the dataset.
        transform: A list of transformations to apply to the images in the dataset.
    """

    def __init__(self, path : str, transform : object = None) -> None:

        self.path = path
        self._species = self.species
        self._full_paths = self._get_full_paths(path)
        self.transform = transform

    @staticmethod
    def _get_full_paths(path : str) -> list:
        """
        Returns a list of full paths to the images in the dataset.
        Args:
            path: A string representing the path to the dataset.
        Returns:
            A list of strings representing the full paths to the images in the dataset.
        """
        sub_paths = os.listdir(path)
        full_paths = []
        for sub_path in sub_paths:
            full_paths.extend([os.path.join(path, sub_path, image) for image in os.listdir(os.path.join(path, sub_path))])
        return full_paths

    @property
    def species(self) ->list:
        """
        Returns a list of classes in the dataset.
        Returns:
        A list of strings representing the classes in the dataset.
        """
        species = os.listdir(self.path)
        return [specie.split('-')[-1] for specie in species]

    def _populate_species(self, query:str, number_of_images : int = 100) -> list:
        """
        Populates the dataset with images from Google Images.
        Args:
            query: A string representing the query to search for.
            number_of_images: An integer representing the number of images to download.
        Returns:
            A list of strings representing the full paths to the downloaded images.
        """
        response = google_images_download.googleimagesdownload()
        new_path = os.path.join(self.path, query)
        arguments = {"keywords":query, "limit":number_of_images, "print_urls":False, "output_directory":self.path}
        response.download(arguments)
        return [os.path.join(new_path, image) for image in os.listdir(new_path)]
        

    def add_species(self, new_species : list, images_per_specie : list = []) -> None:
        """
        Adds new classes to the dataset.
        Args:
            new_species: A list of strings representing the new classes to add.
        """
        self._species.extend(new_species)
        if images_per_specie == []:
            images_per_specie = [100] * len(new_species)
        #Now make new folders for the new species
        #TO DO: check duplicates
        for idx,new_specie in enumerate(new_species):
            try :
                #TO DO: maybe delete return in _populate_species, not sure if it's useful
                _ = self._populate_species(new_specie, images_per_specie[idx])
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
            target_name = specie.split('-')[-1]
            if target_name in species_to_remove:
                index.append(idx)
                species_to_remove.remove(target_name)
        for idx in index:
            path_to_remove = os.path.join(self.path, dirs[idx])
            try:
                os.rmdir(path_to_remove)
            except OSError:
                rmtree(path_to_remove)
        for remaining_specie in species_to_remove:
            print(f"{remaining_specie} not found in dataset")

    def count_images(self) -> dict:
        """
        Counts the number of images in each class.
        Returns:
            A dictionary with the classes as keys and the number of images as values.
        """
        sub_paths = os.listdir(self.path)
        return {sub_path: len(os.listdir(os.path.join(self.path, sub_path))) for sub_path in sub_paths}
        
        
    def __len__(self) ->int:
        """
        Returns the number of classes in the dataset.
        Returns:
            An integer representing the number of classes in the dataset.
        """
        return sum(self.count_images().values())

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.
        Returns:
            A string representing the dataset.
        """
        #len(self) calls the __len__ method
        return f"Dataset with {len(self)} classes"

    def __getitem__(self, index : int) -> torch_tensor:
        """
        Returns the class at the given index.
        Args:
            index: An integer representing the index of the class to return.
        Returns:
            A string representing the class at the given index.
        """
        if index >= len(self):
            raise IndexError("Index out of range")
        else :
            image_path = self._full_paths[index]
            image = read_image(image_path)
            if self.transform:
                image = self.transform(image)
            

            return image




        
        


if __name__ == "__main__":
    images_path = "images/Images"
    dataset = DogsDataSet(path = images_path)
    couted_images = dataset.count_images()
    print(sum(couted_images.values()))
    print(len(couted_images.keys()))