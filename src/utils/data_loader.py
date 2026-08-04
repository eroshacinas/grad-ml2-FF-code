from dataclasses import dataclass, field
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import Compose

@dataclass
class DataConfig:
    data_dir:        str   = "./data"
    batch_size:      int   = 64
    val_split:       float = 0.1
    num_workers:     int   = 2
    pin_memory:      bool  = True
    normalize:       bool  = True
    flatten:         bool  = True
    seed:            int   = 42

    mean: Tuple[float, ...] = field(default_factory=lambda: (0.1307,))
    std:  Tuple[float, ...] = field(default_factory=lambda: (0.3081,))


# TRANSFORM
def flatten_image(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1)

def build_transforms(cfg: DataConfig) -> Tuple[Compose, Compose]:
    base = [transforms.ToTensor()]

    if cfg.normalize:
        base.append(transforms.Normalize(cfg.mean, cfg.std))

    if cfg.flatten:
        base.append(transforms.Lambda(flatten_image)) # 28×28 --> 784

    train_tf = transforms.Compose(base) 
    eval_tf  = transforms.Compose(base)

    return train_tf, eval_tf



def get_dataset(cfg: DataConfig, train: bool, transform: Compose):
    return datasets.MNIST(
        root=cfg.data_dir,
        train=train,
        download=True,
        transform=transform,
    )

def get_dataloaders(
    cfg: Optional[DataConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    returns (train_loader, val_loader, test_loader).

    usage
    -----
        cfg = DataConfig(batch_size=128, flatten=True)
        train_loader, val_loader, test_loader = get_dataloaders(cfg)
    """
    if cfg is None:
        cfg = DataConfig()

    train_tf, eval_tf = build_transforms(cfg)

    # raw datasets
    full_train = get_dataset(cfg, train=True,  transform=train_tf)
    test_set   = get_dataset(cfg, train=False, transform=eval_tf)

    # train / val split
    n_val   = int(len(full_train) * cfg.val_split)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)

    # eval transform for val
    val_set.dataset.transform = eval_tf

    shared = dict(
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,  **shared
    )
    val_loader = DataLoader(
        val_set,   batch_size=cfg.batch_size, shuffle=False, **shared
    )
    test_loader = DataLoader(
        test_set,  batch_size=cfg.batch_size, shuffle=False, **shared
    )

    return train_loader, val_loader, test_loader



# metadata
@dataclass
class DatasetInfo:
    input_dim:   int
    num_classes: int
    train_size:  int
    val_size:    int
    test_size:   int

    def __str__(self) -> str:
        return (
            f"Input dim   : {self.input_dim}\n"
            f"Num classes : {self.num_classes}\n"
            f"Train size  : {self.train_size}\n"
            f"Val size    : {self.val_size}\n"
            f"Test size   : {self.test_size}"
        )


def get_dataset_info(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
) -> DatasetInfo:
    x, y = next(iter(train_loader))
    return DatasetInfo(
        input_dim   = x.shape[-1],
        num_classes = int(y.max().item()) + 1,
        train_size  = len(train_loader.dataset),   
        val_size    = len(val_loader.dataset),     
        test_size   = len(test_loader.dataset),
    )

