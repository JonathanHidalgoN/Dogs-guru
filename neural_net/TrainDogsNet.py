# TO DO : Reduce imports to only the necessary ones
import torchvision
import torch
from torch.utils.data import DataLoader
from time import time as time_time


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

    def train(
        self,
        epochs: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        verbose: bool = True,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> torchvision.models:
        # SOURCE: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
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
        best_model_wts = self.model.state_dict()
        best_acc = 0.0
        for epoch in range(epochs):
            if verbose:
                print("Epoch {}/{}".format(epoch, epochs - 1))
                print("-" * 10)
            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                if phase == "train":
                    dataloader = train_dataloader
                else:
                    dataloader = val_dataloader
                for inputs,labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs.float())
                        _, preds = torch.max(outputs, 1)
                        num_classes = labels.shape[1]
                        preds = torch.nn.functional.one_hot(preds, num_classes)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "train":
                    self.scheduler.step()

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)

                if verbose:
                    print(
                        "{} Loss: {:.4f} Acc: {:.4f}".format(
                            phase, epoch_loss, epoch_acc
                        )
                    )

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.model.state_dict()

            if verbose:
                print()

        time_elapsed = time_time() - since
        if verbose:
            print(
                "Training complete in {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )
            print("Best val Acc: {:4f}".format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

