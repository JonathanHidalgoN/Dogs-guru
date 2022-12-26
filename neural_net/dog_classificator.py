import torchvision
import torch
from images_pipeline.DogsDataSet import DogsDataSet
from images_pipeline.Transformations import Rescale, RandomCrop, ToTensor
from images_pipeline.utils import generate_indexes, count_total_images
from torchvision import transforms
from torch.utils.data import DataLoader


total_images = count_total_images("/kaggle/input/stanford-dogs-dataset/images/Images")
train_parameters = {
    "resize": (256, 256),
    "crop": 224,
    "path": "/kaggle/input/stanford-dogs-dataset/images/Images",
    "batch_size": 128,
    "shuffle": False,
    "proportion": 0.8,
    "index_generator": generate_indexes(total_images, [0.8, 0.1, 0.1]),
}

if __name__ == "__main__":
    composed = transforms.Compose(
        [
            Rescale(train_parameters["resize"]),
            RandomCrop(train_parameters["crop"]),
            ToTensor(),
        ]
    )
    train_dataset = DogsDataSet(
        train_parameters["path"], train_parameters["index_generator"]
    )
    test_dataset = DogsDataSet(
        train_parameters["path"], train_parameters["index_generator"]
    )
    val_dataset = DogsDataSet(
        train_parameters["path"], train_parameters["index_generator"]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_parameters["batch_size"],
        shuffle=train_parameters["shuffle"],
        num_workers=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=train_parameters["batch_size"],
        shuffle=train_parameters["shuffle"],
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_parameters["batch_size"],
        shuffle=train_parameters["shuffle"],
        num_workers=4,
    )


class TrainDogsNet:

    """
    This class is used to train a neural network on the DogsDataSet.
    Attributes:
        model: A torchvision.models object representing the neural network to train.
        criterion: A torch.nn.modules.loss object representing the loss function to use.
        optimizer: A torch.optim.sgd.SGD object representing the optimizer to use.
        scheduler: A torch.optim.lr_scheduler object representing the scheduler to use.
    """

    def __init__(
        self,
        model: torchvision.models,
        criterion: torch.nn.modules.loss,
        optimizer: torch.optim.sgd.SGD,
        scheduler: torch.optim.lr_scheduler,
    ):
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler

    # Define the properties of the class and their setters, maybe not necessary but won't hurt.
    # ------------------------------------
    @property
    def model(self) -> torchvision.models:
        return self._model

    @model.setter
    def model(self, model: torchvision.models) -> None:
        self._model = model

    @property
    def criterion(self) -> torch.nn.modules.loss:
        return self._criterion

    @criterion.setter
    def criterion(self, criterion: torch.nn.modules.loss) -> None:
        self._criterion = criterion

    @property
    def optimizer(self) -> torch.optim.sgd.SGD:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: torch.optim.sgd.SGD) -> None:
        self._optimizer = optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler:
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler: torch.optim.lr_scheduler) -> None:
        self._scheduler = scheduler

    # ------------------------------------
