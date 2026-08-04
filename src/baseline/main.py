from utils.data_loader import DataConfig, get_dataloaders, get_dataset_info
from baseline.model import DenseNet
from baseline.trainer import TrainerConfig, Trainer
import torch



def main():
    # load dataset
    data_cfg = DataConfig(data_dir="../data", batch_size=64, flatten=True)
    train_loader, val_loader, test_loader = get_dataloaders(data_cfg)

    info = get_dataset_info(train_loader, val_loader, test_loader)
    print(info)

    x, y = next(iter(train_loader))
    print(f"\nbatch shapes x: {x.shape}, y: {y.shape}")
    print(f"x dtype: {x.dtype}; range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"y dtype: {y.dtype}; unique labels: {y.unique().tolist()}")

    # load model
    model = DenseNet()
    print(model)

    # instantiate optimizers and loss function
    # instantiate optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss() # for multi-class classification

    trainer_config = TrainerConfig(epochs=3, patience=5, optimizer='adam')

    trainer = Trainer(model, train_loader, val_loader, trainer_config)

    history = trainer.fit()

    trainer.evaluate(test_loader)
    trainer.summary()





if __name__ == "__main__":
    main()