# torch-lightning-with-ray

This project trains a small U-Net for armor plate segmentation using PyTorch Lightning.

## Setup (with mamba/conda)

1) Install mamba (recommended) or conda
- Mamba: https://mamba.readthedocs.io/
- Conda: https://docs.conda.io/

2) Create the environment
```bash
mamba env create -f environment.yml
# or: conda env create -f environment.yml
```

3) Activate the environment
```bash
mamba activate armor-unet
# or: conda activate armor-unet
```

4) (Optional) CPU-only installs
- If you do not have NVIDIA drivers/CUDA, comment out `pytorch-cuda` in `environment.yml`.
- The trainer will automatically fall back to CPU when CUDA is unavailable.

## Alternative setup (pip only)
If you prefer a virtualenv + pip flow:
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Windows (CPython + CUDA 12.4)
If you want GPU acceleration on Windows, use CPython and the official cu124 wheels from PyTorch.

1) Create/refresh the venv with CPython 3.12
```powershell
py -0p            # lists installed Python interpreters
py -3.12 -m venv .venv
```

2) Allow script activation in PowerShell and activate
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
```

3) Install GPU PyTorch (CUDA 12.4), then the rest
```powershell
python -m pip install -U pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
python -m pip install -r requirements.txt
```

4) Verify CUDA is detected
```powershell
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('cuda version', torch.version.cuda)
if torch.cuda.is_available():
    print('device0', torch.cuda.get_device_name(0))
PY
```

CPU-only alternative:
```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
python -m pip install -r requirements.txt
```

Notes:
- Use the `py` launcher to avoid MinGW/alternate Python builds. Wheels are for `win_amd64` CPython.
- If prior docs referenced cu121, use cu124 instead as that’s the current stable index.

## Data layout
Place your dataset in a directory (default: `Dataset_Robomaster-1`) with COCO-style files under `train/`, `valid/`, `test/` containing `_annotations.coco.json` and images referenced therein.

Example:
```
Dataset_Robomaster-1/
  train/
    _annotations.coco.json
    image_001.jpg
    ...
  valid/
    _annotations.coco.json
    ...
  test/
    _annotations.coco.json
    ...
```

## Running training
By default, the script looks for:
- DATA_ROOT: `Dataset_Robomaster-1`
- CHECKPOINT_DIR: `checkpoints`
- LOG_DIR: `logs`

You can override via environment variables.

### Windows PowerShell
```powershell
$env:DATA_ROOT = "C:\\path\\to\\Dataset_Robomaster-1"
$env:CHECKPOINT_DIR = "checkpoints"
$env:LOG_DIR = "logs"
python train.py
```

### macOS/Linux (bash/zsh)
```bash
export DATA_ROOT=/path/to/Dataset_Robomaster-1
export CHECKPOINT_DIR=checkpoints
export LOG_DIR=logs
python train.py
```

## TensorBoard
After (or during) training, view logs:
```bash
tensorboard --logdir logs
```
Open the printed URL in your browser.

## Hyperparameter Tuning
This repository currently ships without Ray Tune. If you want to add tuning back later, see the notes at the end of this README for a clean, minimal Ray Tune setup.

## Sharing TensorBoard Logs
To inspect runs on another machine:

1. Archive logs on this machine:
   ```powershell
   Compress-Archive -Path logs -DestinationPath lightning_runs.zip
   ```
2. Copy `lightning_runs.zip` to the other device (USB, cloud drive, etc.).
3. Extract it there, then launch TensorBoard pointing at the extracted folders:
   ```powershell
   tensorboard --logdir C:\\path\\to\\lightning_runs\\logs
   ```

The `.gitignore` already excludes `lightning_runs.zip` and `lightning_runs/` so archived logs stay out of version control.

## Project structure
```
armor_unet/
  __init__.py
  data.py          # Dataset and LightningDataModule
  lit_module.py    # LightningModule and dice metric
  models.py        # DoubleConv and SmallUNet
train.py            # Entrypoint for training/evaluation
requirements.txt
environment.yml
scripts/
  (optional) scripts/tune.py  # Add your own HPO script later

## Add Ray Tune Later (clean template)
If you want to re-introduce Ray Tune from scratch without touching the core training code:

- Install: `python -m pip install "ray[tune]"`
- Create `scripts/tune.py` with a minimal trainable that instantiates `ArmorDataModule` and `ArmorUNet`, and reports validation metrics via `TuneReportCallback`.
- Keep all Ray-specific code isolated in `scripts/` so the main training entrypoint remains Ray-free.

Minimal outline for `scripts/tune.py`:

```python
import os, argparse, torch, pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from armor_unet.data import ArmorDataModule
from armor_unet.lit_module import ArmorUNet

def train_tune(cfg):
    pl.seed_everything(42, workers=True)
    dm = ArmorDataModule(data_root=cfg['data_root'], batch_size=cfg['batch_size'])
    model = ArmorUNet(learning_rate=cfg['lr'], weight_decay=cfg['wd'], base_channels=cfg['base_ch'])
    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="tb")
    callbacks = [
        TuneReportCallback({"val_dice": "val_dice", "val_loss": "val_loss"}, on="validation_end"),
        ModelCheckpoint(monitor="val_dice", mode="max", save_top_k=1)
    ]
    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.getenv("DATA_ROOT", "Dataset_Robomaster-1"))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    space = {
        "lr": tune.loguniform(1e-5, 3e-3),
        "wd": tune.loguniform(1e-7, 1e-3),
        "base_ch": tune.choice([16, 32, 64]),
        "batch_size": tune.choice([4, 8, 12]),
        "epochs": args.epochs,
        "data_root": args.data_root,
    }
    tuner = tune.Tuner(train_tune,
        param_space=space,
        tune_config=tune.TuneConfig(metric="val_dice", mode="max", scheduler=ASHAScheduler(), num_samples=args.samples))
    tuner.fit()
```
```
