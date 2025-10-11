param(
    [string]$DataRoot = $env:DATA_ROOT,
    [string]$CheckpointDir = $env:CHECKPOINT_DIR,
    [string]$LogDir = $env:LOG_DIR
)

# Resolve defaults if not provided via env or params
if (-not $DataRoot -or $DataRoot.Trim() -eq "") {
    $DataRoot = Join-Path (Get-Location) "Dataset_Robomaster-1"
}
if (-not $CheckpointDir -or $CheckpointDir.Trim() -eq "") {
    $CheckpointDir = Join-Path (Get-Location) "checkpoints"
}
if (-not $LogDir -or $LogDir.Trim() -eq "") {
    $LogDir = Join-Path (Get-Location) "logs"
}

# Create output dirs if missing
New-Item -ItemType Directory -Force -Path $CheckpointDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "Using DATA_ROOT     : $DataRoot"
Write-Host "Using CHECKPOINT_DIR: $CheckpointDir"
Write-Host "Using LOG_DIR       : $LogDir"

$env:DATA_ROOT = $DataRoot
$env:CHECKPOINT_DIR = $CheckpointDir
$env:LOG_DIR = $LogDir

# Kick off training
python "$(Join-Path (Get-Location) 'train.py')"

