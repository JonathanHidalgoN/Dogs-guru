# This file assumes that the dataset is the following https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset
# other datasets might work, if they are structured in the same way
# images/Images
#    n02085620-Chihuahua(Folder with images)
#    n02085782-Japanese_spaniel(Folder with images)
#    ...
# Notice specie have a - before the name, and the name is in the folder.

import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch import zeros as torch_zeros
from torch import Tensor as torch_tensor
from google_images_download import google_images_download
from shutil import rmtree
from typing import Union, Generator, List
from warnings import warn


class DogsDataSet(Dataset):
    # TO DO: maybe add private attributes for paths of subfolders, because I use them a lot
    """
    A class to represent the Stanford Dogs Dataset.
    Attributes:
        path: A string representing the path to the dataset.
        species: A list of strings representing the classes in the dataset.
        transform: A list of transformations to apply to the images in the dataset.
        indexes: A generator of indexes, in utils.py there is a function to generate them.
    """

    def __init__(
        self,
        path: str,
        generator: Generator[int, List[float], torch_tensor],
        transform: object = None,
    ) -> None:

        self.path = path
        self._species = self.species
        self.indexes = next(generator)
        self._full_paths = self._get_full_paths()
        self.transform = transform

    def _get_full_paths(self) -> List[str]:
        """
        Returns a list of full paths to the images in the dataset.
        Args:
            path: A string representing the path to the dataset.
        Returns:
            A list of strings representing the full paths to the images in the dataset.
        """
        sub_paths = os.listdir(self.path)
        full_paths = []
        for sub_path in sub_paths:
            full_paths.extend(
                [
                    os.path.join(self.path, sub_path, image)
                    for image in os.listdir(os.path.join(self.path, sub_path))
                ]
            )
        return [full_paths[index] for index in self.indexes]
        # Cant use this because Tensor of strings is not supported
        # return torch_index_select(torch_tensor(full_paths), 0, self.indexes)

    @property
    def species(self) -> List[str]:
        """
        Returns a list of classes in the dataset.
        Returns:
        A list of strings representing the classes in the dataset.
        """
        species = os.listdir(self.path)
        return [specie.split("-")[-1] for specie in species]

    def _populate_species(self, query: str, number_of_images: int = 100) -> List[str]:
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
        arguments = {
            "keywords": query,
            "limit": number_of_images,
            "print_urls": False,
            "output_directory": self.path,
        }
        response.download(arguments)
        return [os.path.join(new_path, image) for image in os.listdir(new_path)]

    def add_species(
        self, new_species: List[str], images_per_specie: List[int] = []
    ) -> None:
        """
        Adds new classes to the dataset.
        Args:
            new_species: A list of strings representing the new classes to add.
        """
        self._species.extend(new_species)
        if images_per_specie == []:
            images_per_specie = [100] * len(new_species)
        # Now make new folders for the new species
        # TO DO: check duplicates
        for idx, new_specie in enumerate(new_species):
            try:
                # TO DO: maybe delete return in _populate_species, not sure if it's useful
                _ = self._populate_species(new_specie, images_per_specie[idx])
            except FileExistsError:
                print(f"Folder {new_specie} already exists")

    def remove_species(self, species_to_remove: List[str]) -> None:
        """
        Removes classes from the dataset.
        Args:
            species_to_remove: A list of strings representing the classes to remove.
        """
        # TO DO: check if the species to remove are in the dataset
        # TO DO: add a confirmation message, or a safety check
        index = []
        dirs = os.listdir(self.path)
        for idx, specie in enumerate(dirs):
            target_name = specie.split("-")[-1]
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
        # Not working properly since I added the indexes, this counts all the images in the dataset
        warn(
            f"({self.count_images.__name__}) counts all the images in the dataset, not just the ones in the current instance of the class"
        )
        sub_paths = os.listdir(self.path)
        return {
            sub_path: len(os.listdir(os.path.join(self.path, sub_path)))
            for sub_path in sub_paths
        }

    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.
        Returns:
            An integer representing the number of classes in the dataset.
        """
        return len(self._full_paths)

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.
        Returns:
            A string representing the dataset.
        """
        # len(self) calls the __len__ method
        return f"Dataset with {len(self)} classes"

    def __getitem__(self, index: Union[int, List[str]]) -> torch_tensor:
        # This is not always a tensor, it can be a list of tensors or an image
        """
        Returns the class at the given index.
        Args:
            index: An integer representing the index of the class to return.
        Returns:
            A string representing the class at the given index.
        """
        if isinstance(index, int):
            if index >= len(self):
                raise IndexError("Index out of range")
            else:
                image_path = self._full_paths[index]
                image = read_image(image_path)
                if self.transform is not None:
                    image = self.transform(image)
                return image
        elif isinstance(index, list):
            images = []
            for idx in index:
                images.append(self[idx])
            return images
        else:
            raise TypeError("Index must be an integer or a list of integers")

    def _extract_name(self, full_label : str) -> str:
        """
        Extracts the name of the class from the full path.
        Args:
            full_label: A string representing the full path to the class.
        Returns:
            A string representing the name of the class.
        Logic :
            The full path is of the form "images/Images/n02085620-Chihuahua/n02085620_10074.jpg"
            The name of the class is "Chihuahua"
            Looking for the first "-" and the first "/" after it gives the name of the class
        """
        start_index = full_label.index("-") + 1
        end_index = full_label.index("/", start_index)
        return full_label[start_index:end_index]


    def get_labels(self) -> torch_tensor:
        """
        Returns the labels of the dataset.
        Returns:
            A tensor representing the labels of the dataset.
        """
        names = [self._extract_name(path) for path in self._full_paths]
        different_species = len(set(names))
        try:
            # Check that the number of species in the dataset is the same as the number of species in the class
            # This is to avoid errors when creating the labels 
            assert different_species == len(self._species)
        except AssertionError:
            raise AssertionError(f"Number of species in dataset ({different_species}) does not match number of species in class ({len(self._species)})")
        self.specie_to_int = {name: idx for idx, name in enumerate(self._species)}
        full_zeros = torch_zeros(len(names), different_species)
        for idx, name in enumerate(names):
            full_zeros[idx, self.specie_to_int[name]] = 1
        return full_zeros        

if __name__ == "__main__":
    from utils import generate_indexes, count_total_images

    path = "images/Images"
    total_images = count_total_images(path)
    proportion = [0.8, 0.1, 0.1]
    index_generator = generate_indexes(total_images, proportion)
    train_dataset = DogsDataSet(path, index_generator)
    test_dataset = DogsDataSet(path, index_generator)
    val_dataset = DogsDataSet(path, index_generator)
    long1 = len(train_dataset)
    long2 = len(test_dataset)
    long3 = len(val_dataset)
    print(long1, long2, long3)
    print(sum([long1, long2, long3]))
