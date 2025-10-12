import platform
import sys

try:
    import torch
except Exception as e:
    print("torch import failed:", e)
    sys.exit(1)

print("python:", sys.version)
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))

