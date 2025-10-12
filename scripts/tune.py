import os
import sys
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import ResultGrid
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.air import session

# Ensure project root is on sys.path when running as a file
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from armor_unet.data import ArmorDataModule
from armor_unet.lit_module import ArmorUNet


def train_tune(config: dict):
    """Ray Tune trainable that runs one Lightning training trial.

    Preserves model architecture and dataloaders; only hyperparameters and trainer
    settings are varied via the `config` dict.
    """
    pl.seed_everything(42, workers=True)

    # Resolve paths
    data_root = os.environ.get("DATA_ROOT", "Dataset_Robomaster-1")
    trial_dir = session.get_trial_dir()
    tb_dir = os.path.join(trial_dir, "tb")
    ckpt_dir = os.path.join(trial_dir, "ckpts")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # DataModule (unchanged)
    dm = ArmorDataModule(
        data_root=data_root,
        batch_size=config.get("batch_size", 8),
        num_workers=config.get("num_workers", 2),
    )

    # Model (architecture preserved; only hparams are tuned)
    loss_name = config.get("loss_name", "bce")
    loss_params = {}
    if loss_name == "bce_dice":
        loss_params["dice_lambda"] = float(config.get("dice_lambda", 1.0))
    elif loss_name == "focal":
        loss_params["alpha"] = float(config.get("focal_alpha", 0.25))
        loss_params["gamma"] = float(config.get("focal_gamma", 2.0))
    elif loss_name == "tversky":
        loss_params["alpha"] = float(config.get("tversky_alpha", 0.3))
        loss_params["beta"] = float(config.get("tversky_beta", 0.7))
        loss_params["gamma"] = float(config.get("tversky_gamma", 1.33))

    model = ArmorUNet(
        learning_rate=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
        base_channels=config.get("base_channels", 32),
        loss_name=loss_name,
        loss_params=loss_params,
    )

    # Logger per trial
    logger = TensorBoardLogger(save_dir=tb_dir, name="armor_unet")

    # Callbacks: report validation metrics to Ray; checkpoint best
    tune_report = TuneReportCallback({"val_dice": "val_dice", "val_loss": "val_loss"},
                                     on="validation_end")
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="armor-unet-{epoch:02d}-{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        verbose=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=int(config.get("max_epochs", 5)),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=int(config.get("devices", 1)),
        logger=logger,
        callbacks=[tune_report, ckpt_cb, lr_cb],
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)


def main():
    parser = argparse.ArgumentParser(description="Ray Tune HPO for Armor U-Net")
    parser.add_argument("--samples", type=int, default=8, help="Number of samples (trials)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument("--gpus", type=float, default=1, help="GPUs per trial (e.g., 1)")
    parser.add_argument("--cpus", type=float, default=4, help="CPUs per trial")
    parser.add_argument("--name", type=str, default="armor_unet_tune", help="Ray experiment name")
    args = parser.parse_args()

    # Search space (safe ranges; architecture preserved)
    param_space = {
        "learning_rate": tune.loguniform(1e-5, 3e-3),
        "weight_decay": tune.loguniform(1e-7, 1e-3),
        "base_channels": tune.choice([16, 32, 48, 64]),
        "batch_size": tune.choice([4, 8, 12]),
        "num_workers": tune.choice([2, 4]),
        "max_epochs": args.epochs,
        "devices": 1,
        "loss_name": tune.choice(["bce_dice", "focal", "tversky"]),
        "dice_lambda": tune.choice([0.5, 1.0, 1.5]),
        "focal_alpha": tune.choice([0.25, 0.5]),
        "focal_gamma": tune.choice([1.5, 2.0, 2.5]),
        "tversky_alpha": tune.choice([0.3, 0.4]),
        "tversky_beta": tune.choice([0.7, 0.6]),
        "tversky_gamma": tune.choice([1.0, 1.33, 1.5]),
    }

    scheduler = ASHAScheduler(
        time_attr="training_iteration",  # mapped by the callback to epochs
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2,
        metric="val_dice",
        mode="max",
    )

    trainable = tune.with_resources(train_tune, {"cpu": args.cpus, "gpu": args.gpus})

    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="val_dice",
            mode="max",
            scheduler=scheduler,
            num_samples=args.samples,
        ),
        run_config=tune.RunConfig(
            local_dir="ray_results",
            name=args.name,
            verbose=1,
        ),
    )

    result_grid: ResultGrid = tuner.fit()
    best = result_grid.get_best_result(metric="val_dice", mode="max")
    print("Best trial dir:", best.path)
    print("Best metrics:", best.metrics)

    print("\nView all trials in TensorBoard:")
    print(f"  tensorboard --logdir ray_results/{args.name}")


if __name__ == "__main__":
    main()
