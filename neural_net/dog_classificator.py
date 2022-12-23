from images_pipeline.DogsDataSet import DogsDataSet
from images_pipeline.Transformations import Rescale, RandomCrop, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from random import shuffle

def shuffle_data(total_images : int, proportion : float = .8) -> list:
    """
    Shuffle the data and return the indexes of the training and test data
    Args:
        total_images: total number of images
        proportion: proportion of data to be used for training
    Returns:
        indexes of the training and test data
    """
    to_train = int(total_images * proportion)
    random_list = [i for i in range(total_images)]
    shuffle(random_list)
    return random_list[:to_train], random_list[to_train:]

