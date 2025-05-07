from torch.nn.modules import Dropout2d
import torch
import torch.nn as nn
import torch.nn.init as init

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # Here we define the CNN model architecture.

        self.model = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p = dropout),

            nn.Conv2d(64, 64, 3, padding=1),  # 64x224x224
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p = dropout),
            nn.MaxPool2d(2, 2),  # 64x112x112

            nn.Conv2d(64, 128, 3, padding=1),  # 128x112x112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p = dropout),

            nn.Conv2d(128, 128, 3, padding=1),  # 128x112x112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p = dropout),
            nn.MaxPool2d(2,2),   #  128x56x56

            nn.Conv2d(128, 256, 3, padding=1),  # 256x56x56
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p = dropout),
            
            nn.Conv2d(256, 256, 3, padding=1),  # 256x28x28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p = dropout),
            nn.MaxPool2d(2,2), # 256x14x14

            nn.Conv2d(256, 512, 3, padding=1),  # 512x14x14
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p = dropout),
            
            nn.Conv2d(512, 1024, 3, padding=1),  # 1024x14x14
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout2d(p = dropout),
            
            nn.Conv2d(1024, 1024, 3, padding=1),  # 1024x14x14
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout2d(p = dropout),

            nn.Conv2d(1024, 512, 3, padding=1),  # 512x14x14
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p = dropout),
            nn.MaxPool2d(2,2), # 512x7x7

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),

            nn.Linear(512, 265),
            nn.Dropout(p = dropout),
            nn.BatchNorm1d(265),
            nn.ReLU(),
            nn.Linear(265, 128),
            nn.Dropout(p = dropout),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    # Apply Xavier initialization to the convolutional and linear layers
    # This is a type of weight initializations 
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)  
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
