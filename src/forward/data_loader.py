import random
from dataclasses import dataclass, field
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

from utils.data_loader import DataConfig, build_transforms, get_dataset


@dataclass
class FFDataConfig(DataConfig):
    num_classes: int = 10
    # label is stamped into the flat vector
    flatten: bool = True


# HELPERS for label stamping
def stamp_label(x: Tensor, label: int, num_classes: int) -> Tensor:
    """
    overwrite the first `num_classes` elements with a one-hot label vector
    """
    out = x.clone()
    out[:num_classes] = 0.0
    out[label] = 1.0
    return out


def random_wrong_label(correct: int, num_classes: int) -> int:
    """uniform sample from all labels except the correct one"""
    wrong = correct
    while wrong == correct:
        wrong = random.randrange(num_classes)
    return wrong


class FFDataset(Dataset):
    """
    wrap any (image, label) dataset and emits (positive, negative, label) triples

    positive: image stamped with the CORRECT one-hot label
    negative: same image stamped with a WRONG  one-hot label. ie corrupted
    label
    """

    NEG_STRATEGIES = {"wrong_label", "hybrid"}

    def __init__(
        self,
        base_dataset: Dataset,
        num_classes:  int = 10,
        neg_strategy: str = "wrong_label",
    ):
        assert neg_strategy in self.NEG_STRATEGIES, \
            f"neg_strategy must be one of {self.NEG_STRATEGIES}"
        
        self.base         = base_dataset
        self.num_classes  = num_classes
        self.neg_strategy = neg_strategy

    # NEGATIVE CONSTRUCTORS
    def _neg_wrong_label(self, x: Tensor, label: int) -> Tensor:
        wrong = random_wrong_label(label, self.num_classes)
        return stamp_label(x, wrong, self.num_classes)

    def _neg_hybrid(self, x: Tensor, label: int) -> Tensor:
        # TODO
        pass

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        x, label = self.base[idx]

        pos = stamp_label(x, label, self.num_classes)

        if self.neg_strategy == "wrong_label":
            neg = self._neg_wrong_label(x, label)
        else:
            neg = self._neg_hybrid(x, label)

        return pos, neg, label

    def __len__(self) -> int:
        return len(self.base)




# INFERENCE HELPERS. no negative labels here
class FFInferenceDataset(Dataset):
    """
    no labels to corrup at test time.
    instead, stamp every possible label and let the network
    pick the one with the highest goodness

    returns: (candidates, true_label)
      candidates: tensor of shape (num_classes, input_dim)
      candidates[k] = image stamped with label k
    """

    def __init__(self, base_dataset: Dataset, num_classes: int = 10):
        self.base        = base_dataset
        self.num_classes = num_classes

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        x, label = self.base[idx]
        candidates = torch.stack(
            [stamp_label(x, k, self.num_classes) for k in range(self.num_classes)]
        )
        return candidates, label

    def __len__(self) -> int:
        return len(self.base)