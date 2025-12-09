# Blink Frequency Detection Project - Setup Script
# Automates virtual environment creation and dependency installation
# Usage: .\setup.ps1

param(
    [string]$EnvName = "blink",
    [string]$PythonCmd = "python"
)

Write-Host "`n$('='*70)" -ForegroundColor Cyan
Write-Host "Blink Frequency Detection Project - Setup" -ForegroundColor Cyan
Write-Host "$('='*70)`n" -ForegroundColor Cyan

# Check Python installation
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = & $PythonCmd --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8+" -ForegroundColor Red
    Write-Host "   Download: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`n[2/4] Creating virtual environment '$EnvName'..." -ForegroundColor Yellow
if (Test-Path "$EnvName") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
} else {
    try {
        & $PythonCmd -m venv $EnvName
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    } catch {
        Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "`n[3/4] Activating virtual environment..." -ForegroundColor Yellow
$activateScript = ".\$EnvName\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "✗ Activation script not found at $activateScript" -ForegroundColor Red
    exit 1
}

try {
    & $activateScript
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "   Try running: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "`n[4/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
if (-not (Test-Path "requirements.txt")) {
    Write-Host "✗ requirements.txt not found in current directory" -ForegroundColor Red
    exit 1
}

try {
    # Upgrade pip, setuptools, wheel
    Write-Host "  - Upgrading pip, setuptools, wheel..." -ForegroundColor Cyan
    & python -m pip install --upgrade pip setuptools wheel | Out-Null
    
    # Install requirements
    Write-Host "  - Installing packages from requirements.txt..." -ForegroundColor Cyan
    & pip install -r requirements.txt
    
    Write-Host "`n✓ Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host "`n[VERIFY] Testing imports..." -ForegroundColor Yellow
try {
    & python -c "import cv2, mediapipe, pandas, scipy, matplotlib; print('All packages imported successfully!')" 2>&1
    Write-Host "✓ All imports successful" -ForegroundColor Green
} catch {
    Write-Host "✗ Import verification failed" -ForegroundColor Red
    exit 1
}

# Final message
Write-Host "`n$('='*70)" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "$('='*70)`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Activate virtual environment (if not active):"
Write-Host "     .\$EnvName\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "  2. Run the workflow:"
Write-Host "     python blink_workflow.py --video blink_data.mp4 --ir-data IR_data.csv" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Check results in the 'results/' directory" -ForegroundColor Yellow
Write-Host ""
Write-Host "For help: python blink_workflow.py --help" -ForegroundColor Yellow
Write-Host "`n"
