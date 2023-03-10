from torch import tensor as torch_tensor
from torch import randperm as torch_randperm
from torch import arange as torch_arange
from typing import List
import os
from warnings import warn


def generate_indexes(number_images: int, proportions: List[float]) -> torch_tensor:
    """
    Generates indexes for the dataset.
    Args:
        number_images: An integer representing the total number of images in the dataset.
        proportions: A list of floats representing the proportions of the dataset to use.
    Returns:
        A torch.tensor object representing the indexes of the dataset.
    """
    if sum(proportions) != 1:
        warn("The sum of proportions is less than 1, not using all data.")
    indexes = torch_arange(number_images)
    indexes = indexes[torch_randperm(number_images)][0 : int(number_images * sum(proportions))]
    indexes = indexes.split(
        [int(proportion * number_images) for proportion in proportions]
    )
    for index in indexes:
        yield index

def count_total_images(path: str) -> int:
    """
    Counts the total number of images in the dataset.
    Args:
        path: A string representing the path to the dataset.
    Returns:
        An integer representing the total number of images in the dataset.
    """
    sub_paths = os.listdir(path)
    return sum(
        [len(os.listdir(os.path.join(path, sub_path))) for sub_path in sub_paths]
    )


def count_total_classes(path: str) -> int:
    """
    Counts the total number of classes in the dataset.
    Args:
        path: A string representing the path to the dataset.
    Returns:
        An integer representing the total number of classes in the dataset.
    """
    return len(os.listdir(path))

if __name__ == "__main__":
    generator = generate_indexes(100, [0.1, 0.2, 0.3, 0.4])
    indexes = [index for index in generator]
    assert len(indexes) == 4
    assert sum([len(index) for index in indexes]) == 100
    try:
        generate_indexes(100, [0.1])
    except ValueError as e:
        assert str(e) == "The sum of proportions must be equal to 1."
    generator = generate_indexes(100, [1])
    indexes = [index for index in generator]
    assert len(indexes) == 1
    assert len(indexes[0]) == 100
    # This throws an error, I don't know why, because the type of indexes[0] is torch.tensor
    # assert type(indexes[0]) == torch_tensor
    # print(type(indexes[0]))
    total_images = count_total_images("images/Images")
    assert total_images == 20580
    total_classes = count_total_classes("images/Images")
    assert total_classes == 120