"""
scripts/tune.py — Hyperparameter tuning with Ray Tune + PyTorch Lightning.
"""
import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from ray import air, tune
from ray.air import session
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

from armor_unet.data import ArmorDataModule
from armor_unet.lit_module import ArmorUNet


def train_tune(config, data_root: str, max_epochs: int, deterministic: bool):
    """Single Ray Tune trial that runs the Lightning trainer with a sampled config."""
    pl.seed_everything(42, workers=True)

    datamodule = ArmorDataModule(
        data_root=data_root,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 2),
    )

    model = ArmorUNet(
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        base_channels=config["base_channels"],
    )

    wandb_logger = WandbLogger(
        project=os.environ.get("WANDB_PROJECT", "armor-unet"),
        entity=os.environ.get("WANDB_ENTITY"),
        save_dir=os.environ.get("LOG_DIR", "logs"),
        name=session.get_trial_name(),
        log_model=False,
    )

    callbacks = [
        TuneReportCallback({"val_dice": "val_dice", "val_loss": "val_loss"}, on="validation_end"),
        ModelCheckpoint(monitor="val_dice", mode="max", save_top_k=1),
        EarlyStopping(monitor="val_dice", patience=10, mode="max"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=deterministic,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=datamodule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.getenv("DATA_ROOT", "Dataset_Robomaster-1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=12, help="Number of Ray trials")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    search_space = {
        "lr": tune.loguniform(1e-5, 3e-3),
        "weight_decay": tune.loguniform(1e-7, 1e-3),
        "base_channels": tune.choice([16, 32, 64]),
        "batch_size": tune.choice([4, 8, 12]),
        "num_workers": tune.choice([0, 2, 4]),
    }

    scheduler = ASHAScheduler(metric="val_dice", mode="max", grace_period=3, reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_parameters(
            train_tune,
            data_root=args.data_root,
            max_epochs=args.epochs,
            deterministic=args.deterministic,
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_dice",
            mode="max",
            scheduler=scheduler,
            num_samples=args.samples,
        ),
        run_config=air.RunConfig(
            name="armor-unet-tune",
            storage_path=os.environ.get("TUNE_OUTPUT", "ray_results"),
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
