from TrainDogsNet import TrainDogsNet
from images_pipeline.DogsDataSet import DogsDataSet
from images_pipeline.Transformations import Rescale, RandomCrop, ToTensor
from images_pipeline.utils import (
    generate_indexes,
    count_total_images,
    count_total_classes,
)
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss as nn_CrossEntropyLoss
from torch.nn import Linear as nn_Linear
from torch import optim as torch_optim
from torch.optim import lr_scheduler as torch_lr_scheduler
from torch import device as torch_device
from torch.cuda import is_available as torch_cuda_is_available
from torchvision import transforms
from torchvision.models.resnet import resnet18 as models_resnet18

total_images = count_total_images("images/Images")
total_classes = count_total_classes("images/Images")
train_parameters = {
    "resize": (256, 256),
    "crop": 224,
    "path": "images/Images",
    "batch_size": 128,
    "shuffle": False,
    "index_generator": generate_indexes(total_images, [0.8, 0.1, 0.1]),
    "epochs": 10,
    "device": torch_device("cuda:0" if torch_cuda_is_available() else "cpu")
}

# Myabe parameters should be in a separate file, or in a function that returns a dictionary.
model = models_resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn_Linear(num_ftrs, total_classes)
device = torch_device("cuda:0" if torch_cuda_is_available() else "cpu")
model.to(device)
optimizer = torch_optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch_lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn_CrossEntropyLoss()

neural_net_parameters = {
    "model": model,
    "optimizer": optimizer,
    "scheduler": scheduler,
    "criterion": criterion,
}
# ------------------------------------
if __name__ == "__main__":
    composed = transforms.Compose(
        [
            Rescale(train_parameters["resize"]),
            RandomCrop(train_parameters["crop"]),
            ToTensor(),
        ]
    )
    train_dataset = DogsDataSet(
        train_parameters["path"], train_parameters["index_generator"], transform=composed
    )
    test_dataset = DogsDataSet(
        train_parameters["path"], train_parameters["index_generator"], transform=composed
    )
    val_dataset = DogsDataSet(
        train_parameters["path"], train_parameters["index_generator"], transform=composed
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

    training_class = TrainDogsNet(
        model=neural_net_parameters["model"],
        optimizer=neural_net_parameters["optimizer"],
        scheduler=neural_net_parameters["scheduler"],
        criterion=neural_net_parameters["criterion"],
    )

    training_class.train(
        epochs = train_parameters["epochs"],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        verbose=True,
        device=train_parameters["device"]
    )