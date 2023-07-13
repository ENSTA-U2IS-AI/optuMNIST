from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch import nn
from torch.nn import Module
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.routines.classification import ClassificationSingle

from .utils import NonConvergence


class OptuMNIST(Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=4,
            stride=1,
            bias=False,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=3,
        )
        self.conv2 = nn.Conv2d(
            in_channels=2,
            out_channels=10,
            kernel_size=5,
            groups=2,
            stride=1,
            bias=False,
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(
            in_features=10,
            out_features=3,
            bias=True,
        )
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(
            in_features=3,
            out_features=10,
            bias=True,
        )

    def forward(self, x):
        y = self.relu1(self.conv1(x))
        y = self.pool1(y)
        y = self.relu2(self.conv2(y))
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.relu3(self.fc1(y))
        y = self.fc2(y)
        return y


def optim_mnist(model: nn.Module) -> dict:
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.04,
        weight_decay=0.0002,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[15, 30],
        gamma=0.5,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    root = Path(__file__).parent.absolute()

    datamodule = MNISTDataModule

    parser = ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser = datamodule.add_argparse_args(parser)

    args = parser.parse_args("")
    args.batch_size = 2**6
    args.max_epochs = 60
    args.precision = 16
    args.accelerator = "cpu"
    args.num_workers = 4

    if isinstance(root, str):
        root = Path(root)

    # datamodule
    args.root = str(root / args.root)
    dm = datamodule(**vars(args))

    # model
    model = OptuMNIST()
    baseline = ClassificationSingle(
        num_classes=10,
        model=model,
        loss=nn.CrossEntropyLoss,
        optimization_procedure=optim_mnist,
    )

    tb_logger = TensorBoardLogger(
        str(root / "logs"),
        name="optumnist-v1",
        default_hp_metric=False,
    )

    early_stop_callback = NonConvergence(monitor="hp/val_acc", mode="max", divergence_threshold=0.15)

    save_checkpoints = ModelCheckpoint(
        monitor="hp/val_acc",
        mode="max",
        save_weights_only=False,
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        deterministic=False,
        logger=tb_logger,
        callbacks=[save_checkpoints, early_stop_callback],
    )

    # training and testing
    trainer.fit(baseline, dm)
    test = trainer.test(datamodule=dm, ckpt_path="best")
