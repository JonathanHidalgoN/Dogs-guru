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

train_parameters = {"resize" : (256,56),
                   "crop" : 224,
                   "path" : "/kaggle/input/stanford-dogs-dataset/images/Images",
                   "batch_size" : 128,
                   "shuffle" : True,
                   "proportion" : .8}

if __name__ == "__main__":
    composed = transforms.Compose([
        Rescale(train_parameters["resize"]),RandomCrop(train_parameters["crop"]),ToTensor()])
    transformed_dataset = DogsDataSet(path = train_parameters["path"], transform = composed)
    total_images = len(transformed_dataset)
    train_index , test_index = shuffle_data(total_images, train_parameters["proportion"])
    train_dataloader = DataLoader(transformed_dataset[train_index], 
                                  batch_size=train_parameters["batch_size"], 
                                  shuffle=train_parameters["shuffle"], 
                                  num_workers=4)
    test_dataloader = DataLoader(transformed_dataset[test_index], 
                                  batch_size=train_parameters["batch_size"], 
                                  shuffle=train_parameters["shuffle"], 
                                  num_workers=4)
    