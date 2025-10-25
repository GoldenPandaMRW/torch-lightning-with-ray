import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Optional

from armor_unet.data import ArmorDataModule
from armor_unet.lit_module import ArmorUNet

# Limit PyTorch CPU threading via env (helps reduce CPU spikes)
try:
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
except Exception:
    pass
try:
    torch.set_num_interop_threads(int(os.getenv("TORCH_NUM_INTEROP_THREADS", "1")))
except Exception:
    pass


DATA_ROOT = os.environ.get('DATA_ROOT', 'Dataset_Robomaster-1')
CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', 'checkpoints')
LOG_DIR = os.environ.get('LOG_DIR', 'logs')


def train_armor_detector(
    data_root=DATA_ROOT,
    batch_size: Optional[int] = None,
    max_epochs: Optional[int] = None,
    learning_rate: float = 1e-4,
    base_channels: Optional[int] = None,
    num_workers: Optional[int] = None,
    img_size: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    precision: Optional[str] = None,
    limit_train: Optional[float] = None,
    limit_val: Optional[float] = None,
    sleep_ms: Optional[int] = None,
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
):
    """Complete training pipeline with PyTorch Lightning"""

    pl.seed_everything(42, workers=True)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("="*80)
    print("ARMOR PLATE DETECTION - PyTorch Lightning")
    print("="*80)

    # Read runtime controls from environment (with conservative defaults)
    bs = int(os.getenv("BATCH_SIZE", str(batch_size if batch_size is not None else 1)))
    me = int(os.getenv("MAX_EPOCHS", str(max_epochs if max_epochs is not None else 1)))
    base_ch = int(os.getenv("BASE_CHANNELS", str(base_channels if base_channels is not None else 16)))
    nw = int(os.getenv("NUM_WORKERS", str(num_workers if num_workers is not None else 0)))
    isz = int(os.getenv("IMG_SIZE", str(img_size if img_size is not None else 320)))
    pm_env = os.getenv("PIN_MEMORY")
    pm = pin_memory if pin_memory is not None else (False if pm_env is None else pm_env.strip() not in ("0", "false", "False"))

    if precision is None:
        default_prec = "16-mixed" if torch.cuda.is_available() else "32-true"
        precision = os.getenv("PRECISION", default_prec)

    limit_train = float(os.getenv("LIMIT_TRAIN", str(limit_train if limit_train is not None else 1.0)))
    limit_val = float(os.getenv("LIMIT_VAL", str(limit_val if limit_val is not None else 1.0)))
    sleep_ms = int(os.getenv("SLEEP_MS", str(sleep_ms if sleep_ms is not None else 0)))

    print("\nInitializing data module...")
    datamodule = ArmorDataModule(
        data_root=data_root,
        batch_size=bs,
        num_workers=nw,
        img_size=isz,
        pin_memory=pm,
    )

    print("Creating model...")
    model = ArmorUNet(
        learning_rate=learning_rate,
        base_channels=base_ch,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='armor-unet-{epoch:02d}-{val_dice:.4f}',
        monitor='val_dice',
        mode='max',
        save_top_k=1,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_dice',
        patience=15,
        mode='max',
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    logger = TensorBoardLogger(log_dir, name='armor_unet')

    # Optional tiny sleep to further reduce sustained utilization
    class Throttle(pl.Callback):
        def __init__(self, ms: int = 0):
            self.delay = max(0, ms) / 1000.0
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self.delay > 0:
                import time
                time.sleep(self.delay)

    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
    if sleep_ms and sleep_ms > 0:
        callbacks.append(Throttle(sleep_ms))

    trainer = pl.Trainer(
        max_epochs=me,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        num_sanity_val_steps=0,
    )

    print("\nStarting training...")
    print("="*80)
    trainer.fit(model, datamodule)

    print("\n" + "="*80)
    print("Testing best model...")
    trainer.test(model, datamodule, ckpt_path='best')

    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best validation Dice: {checkpoint_callback.best_model_score:.4f}")
    print(f"\nView training in TensorBoard:")
    print(f"  tensorboard --logdir {log_dir}")

    return model, trainer, datamodule


if __name__ == '__main__':
    # Minimal CLI via env vars; users can also edit defaults above
    max_epochs_env = os.environ.get('MAX_EPOCHS')
    if max_epochs_env is not None:
        try:
            me = int(max_epochs_env)
        except ValueError:
            me = 5
        train_armor_detector(max_epochs=me)
    else:
        train_armor_detector()
