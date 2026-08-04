# TODO: stitch together funcs to complete train-eval pipeline of forward-forward algorithm
from forward.trainer import ff_epoch, ff_accuracy
from forward.model import ForwardNet, ForwardNetConfig

from utils.data_loader import build_transforms, get_dataset
from forward.data_loader import FFDataConfig, FFDataset, FFInferenceDataset

import torch
from torch.utils.data import random_split, DataLoader, Dataset
import os


class Transformed(Dataset):
    def __init__(self, subset, tf): self.subset, self.tf = subset, tf
    def __len__(self):  return len(self.subset)
    def __getitem__(self, i):
        x, y = self.subset[i]
        return self.tf(x), y


def build_data():
    cfg = FFDataConfig()
    train_tf, eval_tf = build_transforms(cfg)

    full_train = get_dataset(cfg, train=True,  transform=None)
    raw_test   = get_dataset(cfg, train=False, transform=eval_tf) # test keeps its transform directly

    n_val   = int(len(full_train) * cfg.val_split)
    n_train = len(full_train) - n_val

    generator = torch.Generator().manual_seed(cfg.seed)
    train_split, val_split = random_split(full_train, [n_train, n_val], generator=generator)

    # per-split transforms
    train_split = Transformed(train_split, train_tf)
    val_split   = Transformed(val_split,   eval_tf)

    shared = dict(num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    neg_strategy = "wrong_label" # hybrid not implemented yet

    train_ds     = FFDataset(train_split, cfg.num_classes, neg_strategy) # pos/neg pairs for FF updates
    val_infer_ds = FFInferenceDataset(val_split, cfg.num_classes) # all-label stampings for goodness eval
    test_ds      = FFInferenceDataset(raw_test,  cfg.num_classes)

    train_loader = DataLoader(train_ds,     batch_size=cfg.batch_size, shuffle=True,  **shared)
    val_loader   = DataLoader(val_infer_ds, batch_size=cfg.batch_size, shuffle=False, **shared)
    test_loader  = DataLoader(test_ds,      batch_size=cfg.batch_size, shuffle=False, **shared)

    return train_loader, val_loader, test_loader



def main():
    train_loader, val_loader, test_loader = build_data()

    model_config = ForwardNetConfig(
        hidden_dims=[500, 500, 500, 500],
        activation='relu'
    )

    model = ForwardNet(model_config)
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



    layer_optimizers = [
        torch.optim.Adam(layer.parameters(), lr=0.03)
        for layer in model.hidden_layers
    ]

    # doesnt work with FF
    # layer_optimizers = [
    #         torch.optim.SGD(layer.parameters(), lr=0.02)   # was lr=1e-3
    #         for layer in model.hidden_layers
    #     ]

    EPOCHS = 3
    for epoch in range(1, EPOCHS + 1):
        tr_losses = ff_epoch(model, train_loader, layer_optimizers, device, train=True)
        va_acc    = ff_accuracy(model, val_loader, device)

        layer_loss_str = "  ".join(f"L{i+1}={l:.4f}" for i, l in enumerate(tr_losses))
        print(f"Epoch {epoch:>3}/{EPOCHS} │ train [{layer_loss_str}] │ val acc={va_acc:.3f}")

    # TEST ACC
    te_acc = ff_accuracy(model, test_loader, device)
    print(f"Test acc={te_acc:.3f}")

    os.makedirs('checkpoints', exist_ok=True)


if __name__ == "__main__":
    main()