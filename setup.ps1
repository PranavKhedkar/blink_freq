# Blink Detection Project Setup Script for Windows
# Run this script in PowerShell to set up the project

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Blink Detection Project Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check if virtual environment exists
Write-Host ""
Write-Host "Checking virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "[OK] Virtual environment already exists" -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = ".\venv\Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Activation script not found" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Pip upgraded" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Failed to upgrade pip (continuing anyway)" -ForegroundColor Yellow
}

# Install required packages
Write-Host ""
Write-Host "Installing required packages..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Cyan

$packages = @(
    "opencv-python",
    "mediapipe",
    "numpy",
    "pandas",
    "matplotlib",
    "python-docx",
    "openpyxl"
)

$failed = @()
foreach ($package in $packages) {
    Write-Host "  Installing $package..." -ForegroundColor Gray
    python -m pip install $package --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] $package installed" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] Failed to install $package" -ForegroundColor Red
        $failed += $package
    }
}

# Check installation results
Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "[SUCCESS] All packages installed!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "[WARNING] Some packages failed to install:" -ForegroundColor Yellow
    foreach ($pkg in $failed) {
        Write-Host "  - $pkg" -ForegroundColor Red
    }
    Write-Host "========================================" -ForegroundColor Yellow
}

# Create results directory
Write-Host ""
Write-Host "Creating output directories..." -ForegroundColor Yellow
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
    Write-Host "[OK] Created results/ directory" -ForegroundColor Green
} else {
    Write-Host "[OK] results/ directory already exists" -ForegroundColor Green
}

# List installed packages
Write-Host ""
Write-Host "Installed packages:" -ForegroundColor Cyan
python -m pip list

# Final instructions
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the blink detection workflow:" -ForegroundColor Yellow
Write-Host "  1. Make sure virtual environment is activated:" -ForegroundColor White
Write-Host "     .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Run the workflow:" -ForegroundColor White
Write-Host "     python blink_workflow_fixed.py --video input.mp4 --ir-data ir_data.csv" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. View results in the results/ folder" -ForegroundColor White
Write-Host ""
Write-Host "For help, run:" -ForegroundColor Yellow
Write-Host "  python blink_workflow_fixed.py --help" -ForegroundColor Gray
Write-Host ""