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
```
