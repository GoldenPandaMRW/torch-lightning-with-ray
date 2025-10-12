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

## Hyperparameter Tuning (Ray Tune)
Run multiple trials in parallel without changing the model or dataloaders.

Install Ray Tune (once):
```powershell
python -m pip install "ray[tune]"
```

Run a quick search (5 epochs/trial, 8 trials):
```powershell
python scripts/tune.py --samples 8 --epochs 5 --gpus 1 --cpus 4
```

What it does:
- Tunes `learning_rate`, `weight_decay`, `base_channels`, `batch_size` with ASHA.
- Each trial writes TensorBoard logs to its own subfolder.
- Results are under `ray_results/armor_unet_tune`.

View all trials in TensorBoard:
```bash
tensorboard --logdir ray_results/armor_unet_tune
```

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
  tune.py          # Ray Tune search over hparams
```
