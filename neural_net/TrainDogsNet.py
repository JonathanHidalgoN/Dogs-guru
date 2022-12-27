#TO DO : Reduce imports to only the necessary ones
import torchvision
import torch
from torch.utils.data import DataLoader
from time import time as time_time
from typing import List


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
        optimizer: torch.optim,
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
    def optimizer(self) -> torch.optim:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: torch.optim) -> None:
        self._optimizer = optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler:
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler: torch.optim.lr_scheduler) -> None:
        self._scheduler = scheduler

    # ------------------------------------

    def _enconde_labels(self, labels: List[str], number_of_classes : int) -> torch.Tensor:
        """
        This method is used to encode the labels of the dataset.
        Args:
            labels: A list of strings representing the labels of the dataset.
            number_of_classes: An integer representing the number of classes in the dataset.
        Returns:
            A torch.Tensor object representing the encoded labels.
        """

    def train(self, epochs: int, train_dataloader: DataLoader, val_dataloader: DataLoader, verbose : bool = True) -> torchvision.models:
        #SOURCE: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        """
        This method is used to train the neural network.
        Args:
            epochs: An integer representing the number of epochs to train the neural network.
            train_dataloader: A torch.utils.data.dataloader.DataLoader object representing the training set.
            val_dataloader: A torch.utils.data.dataloader.DataLoader object representing the validation set.
            verbose: A boolean representing whether to print the training progress or not.
        Returns:
            A torchvision.models object representing the trained neural network.
        """
        since = time_time()
        best_model_wts = self._model.state_dict()
        best_acc = 0.0

        for epoch in range(epochs):
            
            if verbose:
                print("Epoch {}/{}".format(epoch, epochs - 1))
                print("-" * 10)
            for phase in ["train", "val"]:
                if phase == "train":
                    self._model.train()
                else:
                    self._model.eval()

            running_loss = 0.0
            running_corrects = 0

            pass
        pass
