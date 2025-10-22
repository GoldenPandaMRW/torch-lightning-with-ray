1:# torch-lightning-with-ray
2:
3:This project trains a small U-Net for armor plate segmentation using PyTorch Lightning.
4:
5:## Setup (with mamba/conda)
6:
7:1) Install mamba (recommended) or conda
8:- Mamba: https://mamba.readthedocs.io/
9:- Conda: https://docs.conda.io/
10:
11:2) Create the environment
12:```bash
13:mamba env create -f environment.yml
14:# or: conda env create -f environment.yml
15:```
16:
17:3) Activate the environment
18:```bash
19:mamba activate armor-unet
20:# or: conda activate armor-unet
21:```
22:
23:4) (Optional) CPU-only installs
24:- If you do not have NVIDIA drivers/CUDA, comment out `pytorch-cuda` in `environment.yml`.
25:- The trainer will automatically fall back to CPU when CUDA is unavailable.
26:
27:## Alternative setup (pip only)
28:If you prefer a virtualenv + pip flow:
29:```bash
30:python -m venv .venv
31:# Windows PowerShell:
32:.\.venv\Scripts\Activate.ps1
33:# macOS/Linux:
34:source .venv/bin/activate
35:pip install -r requirements.txt
36:```
37:
38:## Windows (CPython + CUDA 12.4)
39:If you want GPU acceleration on Windows, use CPython and the official cu124 wheels from PyTorch.
40:
41:1) Create/refresh the venv with CPython 3.12
42:```powershell
43:py -0p            # lists installed Python interpreters
44:py -3.12 -m venv .venv
45:```
46:
47:2) Allow script activation in PowerShell and activate
48:```powershell
49:Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
50:. .\.venv\Scripts\Activate.ps1
51:```
52:
53:3) Install GPU PyTorch (CUDA 12.4), then the rest
54:```powershell
55:python -m pip install -U pip setuptools wheel
56:python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
57:python -m pip install -r requirements.txt
58:```
59:
60:4) Verify CUDA is detected
61:```powershell
62:python - <<'PY'
63:import torch
64:print('torch', torch.__version__)
65:print('cuda available', torch.cuda.is_available())
66:print('cuda version', torch.version.cuda)
67:if torch.cuda.is_available():
68:    print('device0', torch.cuda.get_device_name(0))
69:PY
70:```
71:
72:CPU-only alternative:
73:```powershell
74:python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
75:python -m pip install -r requirements.txt
76:```
77:
78:Notes:
79:- Use the `py` launcher to avoid MinGW/alternate Python builds. Wheels are for `win_amd64` CPython.
80:- If prior docs referenced cu121, use cu124 instead as that’s the current stable index.
81:
82:## Data layout
83:Place your dataset in a directory (default: `Dataset_Robomaster-1`) with COCO-style files under `train/`, `valid/`, `test/` containing `_annotations.coco.json` and images referenced therein.
84:
85:Example:
86:```
87:Dataset_Robomaster-1/
88:  train/
89:    _annotations.coco.json
90:    image_001.jpg
91:    ...
92:  valid/
93:    _annotations.coco.json
94:    ...
95:  test/
96:    _annotations.coco.json
97:    ...
98:```
99:
100:## Running training
101:By default, the script looks for:
102:- DATA_ROOT: `Dataset_Robomaster-1`
103:- CHECKPOINT_DIR: `checkpoints`
104:- LOG_DIR: `logs`
105:
106:You can override via environment variables.
107:
108:### Windows PowerShell
109:```powershell
110:$env:DATA_ROOT = "C:\\path\\to\\Dataset_Robomaster-1"
111:$env:CHECKPOINT_DIR = "checkpoints"
112:$env:LOG_DIR = "logs"
113:python train.py
114:```
115:
116:### macOS/Linux (bash/zsh)
117:```bash
118:export DATA_ROOT=/path/to/Dataset_Robomaster-1
119:export CHECKPOINT_DIR=checkpoints
120:export LOG_DIR=logs
121:python train.py
122:```
123:
124:## TensorBoard
125:After (or during) training, view logs:
126:```bash
127:tensorboard --logdir logs
128:```
129:Open the printed URL in your browser.
130:
131:## Hyperparameter Tuning
132:This repository currently ships without Ray Tune. If you want to add tuning back later, see the notes at the end of this README for a clean, minimal Ray Tune setup.
133:
134:## Sharing TensorBoard Logs
135:To inspect runs on another machine:
136:
137:1. Archive logs on this machine:
138:   ```powershell
139:   Compress-Archive -Path logs -DestinationPath lightning_runs.zip
140:   ```
141:2. Copy `lightning_runs.zip` to the other device (USB, cloud drive, etc.).
142:3. Extract it there, then launch TensorBoard pointing at the extracted folders:
143:   ```powershell
144:   tensorboard --logdir C:\\path\\to\\lightning_runs\\logs
145:   ```
146:
147:The `.gitignore` already excludes `lightning_runs.zip` and `lightning_runs/` so archived logs stay out of version control.
148:
149:## Project structure
150:```
151:armor_unet/
152:  __init__.py
153:  data.py          # Dataset and LightningDataModule
154:  lit_module.py    # LightningModule and dice metric
155:  models.py        # DoubleConv and SmallUNet
156:train.py            # Entrypoint for training/evaluation
157:requirements.txt
158:environment.yml
159:scripts/
160:  (optional) scripts/tune.py  # Add your own HPO script later
161:
162:## Add Ray Tune Later (clean template)
163:If you want to re-introduce Ray Tune from scratch without touching the core training code:
164:
165:- Install: `python -m pip install "ray[tune]"`
166:- Create `scripts/tune.py` with a minimal trainable that instantiates `ArmorDataModule` and `ArmorUNet`, and reports validation metrics via `TuneReportCallback`.
167:- Keep all Ray-specific code isolated in `scripts/` so the main training entrypoint remains Ray-free.
168:
169:Minimal outline for `scripts/tune.py`:
170:
171:```python
172:import os, argparse, torch, pytorch_lightning as pl
173:from pytorch_lightning.callbacks import ModelCheckpoint
174:from pytorch_lightning.loggers import TensorBoardLogger
175:from ray import tune
176:from ray.tune.schedulers import ASHAScheduler
177:from ray.tune.integration.pytorch_lightning import TuneReportCallback
178:from armor_unet.data import ArmorDataModule
179:from armor_unet.lit_module import ArmorUNet
180:
181:def train_tune(cfg):
182:    pl.seed_everything(42, workers=True)
183:    dm = ArmorDataModule(data_root=cfg['data_root'], batch_size=cfg['batch_size'])
184:    model = ArmorUNet(learning_rate=cfg['lr'], weight_decay=cfg['wd'], base_channels=cfg['base_ch'])
185:    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="tb")
186:    callbacks = [
187:        TuneReportCallback({"val_dice": "val_dice", "val_loss": "val_loss"}, on="validation_end"),
188:        ModelCheckpoint(monitor="val_dice", mode="max", save_top_k=1)
189:    ]
190:    trainer = pl.Trainer(
191:        max_epochs=cfg['epochs'],
192:        accelerator="gpu" if torch.cuda.is_available() else "cpu",
193:        devices=1,
194:        logger=logger,
195:        callbacks=callbacks,
196:        deterministic=True,
197:    )
198:    trainer.fit(model, datamodule=dm)
199:
200:if __name__ == "__main__":
201:    ap = argparse.ArgumentParser()
202:    ap.add_argument("--data-root", default=os.getenv("DATA_ROOT", "Dataset_Robomaster-1"))
203:    ap.add_argument("--samples", type=int, default=8)
204:    ap.add_argument("--epochs", type=int, default=5)
205:    args = ap.parse_args()
206:
207:    space = {
208:        "lr": tune.loguniform(1e-5, 3e-3),
209:        "wd": tune.loguniform(1e-7, 1e-3),
210:        "base_ch": tune.choice([16, 32, 64]),
211:        "batch_size": tune.choice([4, 8, 12]),
212:        "epochs": args.epochs,
213:        "data_root": args.data_root,
214:    }
215:    tuner = tune.Tuner(train_tune,
216:        param_space=space,
217:        tune_config=tune.TuneConfig(metric="val_dice", mode="max", scheduler=ASHAScheduler(), num_samples=args.samples))
218:    tuner.fit()
219:```
220:```
