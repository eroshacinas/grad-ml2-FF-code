from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader



@dataclass
class TrainerConfig:
    epochs:        int   = 3
    learning_rate: float = 1e-3
    weight_decay:  float = 0.0
    optimizer:     str   = "adam"
    momentum:      float = 0.9
    use_scheduler: bool  = False      
    patience:      int   = 5           
    device:        str   = "auto"     
    log_every:     int   = 1



@dataclass
class EpochResult:
    epoch:     int
    train_loss: float
    val_loss:   float
    train_acc:  float
    val_acc:    float
    elapsed:    float # seconds

    def __str__(self) -> str:
        return (
            f"Epoch {self.epoch:>3} | "
            f"train loss {self.train_loss:.4f}  acc {self.train_acc:.2%} | "
            f"val loss {self.val_loss:.4f}  acc {self.val_acc:.2%} | "
            f"{self.elapsed:.1f}s"
        )


# HELPERS
def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _build_optimizer(model: nn.Module, cfg: TrainerConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'. Choose 'adam' or 'sgd'.")


def _run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None, # None is eval mode
) -> tuple[float, float]:
    """
    run one epoch
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss   = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss    += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy



class Trainer:
    """
    wrap training loop, validation, early stopping, and logging

    usage
    -----
        trainer = Trainer(model, train_loader, val_loader, TrainerConfig())
        history = trainer.fit()
        trainer.evaluate(test_loader)       # final test set score
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          Optional[TrainerConfig] = None,
    ):
        self.cfg          = cfg or TrainerConfig()
        self.device       = _resolve_device(self.cfg.device)
        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = nn.CrossEntropyLoss() # multi-class classification loss
        self.optimizer    = _build_optimizer(self.model, self.cfg)
        self.scheduler    = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.epochs
            )
            if self.cfg.use_scheduler else None
        )
        self.history: List[EpochResult] = []


    def fit(self) -> List[EpochResult]:
        best_val_loss   = float("inf")
        epochs_no_improve = 0
        best_state      = None

        print(f"Training on {self.device}  |  {self.cfg.epochs} epochs\n")

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.perf_counter()

            train_loss, train_acc = _run_epoch(
                self.model, self.train_loader, self.criterion,
                self.device, self.optimizer,
            )
            val_loss, val_acc = _run_epoch(
                self.model, self.val_loader, self.criterion,
                self.device, optimizer=None,
            )

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.perf_counter() - t0
            result  = EpochResult(epoch, train_loss, val_loss, train_acc, val_acc, elapsed)
            self.history.append(result)

            if epoch % self.cfg.log_every == 0:
                print(result)

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_state       = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if self.cfg.patience > 0 and epochs_no_improve >= self.cfg.patience:
                print(f"\nEarly stopping at epoch {epoch} (no val improvement for {self.cfg.patience} epochs)")
                break

        # restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\nRestored best weights (val loss {best_val_loss:.4f})")

        return self.history


    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Run once on the test set. Call after fit()."""
        loss, acc = _run_epoch(
            self.model, test_loader, self.criterion,
            self.device, optimizer=None,
        )
        print(f"\nTest  |  loss {loss:.4f}  acc {acc:.2%}")
        return {"test_loss": loss, "test_acc": acc}


    def summary(self) -> None:
        """Print best epoch from history."""
        if not self.history:
            print("No training history yet.")
            return
        best = min(self.history, key=lambda r: r.val_loss)
        print(f"\nBest epoch: {best}")