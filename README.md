# Blink Frequency Detection Project

A comprehensive workflow for detecting and analyzing blink patterns from video and IR sensor data. Compares video-based blink detection with IR baseline to identify reduced blinking frequency (digital eye strain, screen fatigue, etc.).

## Features

- **Video Blink Detection**: Uses MediaPipe face mesh + Eye Aspect Ratio (EAR) to detect blinks from webcam/video files
- **IR Sensor Integration**: Processes IR sensor data to establish baseline blink patterns
- **Comparative Analysis**: Compares video vs IR data using a 6-second inter-blink interval threshold
- **Visual Reports**: Generates blink timelines and detailed comparison reports
- **Accurate Timestamps**: Logs blink midpoints for precise timing analysis

## Project Structure

```
Blink Freq Project/
├── blink/                              # Python virtual environment
├── blink_detector_accurate_timestamps.py  # Video blink detection class
├── blink_workflow_fixed.py              # Main orchestrator workflow
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── setup.ps1                            # Setup script (Windows PowerShell)
└── results/                             # Output directory (generated)
    ├── video_blinks.csv
    ├── ir_blinks.csv
    ├── blink_timeline.png
    └── comparison_report.txt
```

## Installation & Setup

### Prerequisites
- **Python 3.8+** (check: `python --version`)
- **Windows PowerShell 5.1+** (or adapt commands to your shell)
- **Video file** (MP4, AVI, MOV, etc.)
- **IR data CSV** with timestamp and blink indicator columns

### Quick Setup (Automated - Windows)

```powershell
cd 'D:\Blink Freq Project'
.\setup.ps1
```

This script will:
1. Create a virtual environment named `blink`
2. Install all required packages from `requirements.txt`
3. Verify the installation

### Manual Setup (Cross-platform)

```bash
# Navigate to project directory
cd 'D:\Blink Freq Project'

# Create virtual environment
python -m venv blink

# Activate (Windows PowerShell)
.\blink\Scripts\Activate.ps1

# If blocked by execution policy:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\blink\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import cv2, mediapipe, pandas, scipy; print('All packages installed!')"
```

## Usage

### Running the Complete Workflow

```bash
# Activate virtual environment (if not already active)
.\blink\Scripts\Activate.ps1

# Run the workflow
python blink_workflow_fixed.py --video path\to\video.mp4 --ir-data path\to\ir_data.csv --output-dir results

# Example:
python blink_workflow_fixed.py --video "C:\Videos\screen_session.mp4" --ir-data "C:\Data\ir_sensor.csv" --output-dir results
```

### Command-line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video` | Yes | — | Path to video file (MP4, AVI, MOV, etc.) |
| `--ir-data` | Yes | — | Path to IR sensor data CSV |
| `--output-dir` | No | `results` | Output directory for reports and CSVs |

### IR Data CSV Format

Your IR CSV must have:
- **Timestamp column**: Named `Timestamp`, `Time`, or `Date` (case-insensitive)
- **Blink indicator column**: Named `Blink`, `IR`, `Sensor`, `Detect`, or `Eye` (case-insensitive)

**Example IR data:**
```csv
Timestamp,Blink_Signal
2025-12-08T10:00:00.000000,0.1
2025-12-08T10:00:00.033333,0.2
2025-12-08T10:00:00.066667,0.8
2025-12-08T10:00:00.100000,0.9
2025-12-08T10:00:00.133333,0.7
2025-12-08T10:00:00.166667,0.2
```

If your CSV uses different column names, edit `blink_workflow_fixed.py` lines 176–185 to match your column names.

## Workflow Steps

### Step 1: Video Processing
- Reads video file frame-by-frame
- Extracts facial landmarks using MediaPipe
- Calculates Eye Aspect Ratio (EAR) for each frame
- Detects blinks when EAR drops below threshold (default: 0.18)
- Outputs: `video_blinks.csv`

### Step 2: IR Data Processing
- Reads IR sensor CSV
- Parses timestamps (handles malformed ISO8601 dates)
- Detects blinks as periods where signal exceeds threshold (default: 0.5)
- Groups contiguous high-signal regions into individual blinks
- Outputs: `ir_blinks.csv`

### Step 3: Comparative Analysis
- Calculates inter-blink intervals for both video and IR data
- **Key metric**: Intervals > 6 seconds (long pauses = reduced blinking)
- Flags as "reduced blinking" if:
  - `> 20%` of intervals exceed 6 seconds, OR
  - Average interval > 5 seconds

### Step 4: Report Generation
- **blink_timeline.png**: Visual timeline of blinks (color-coded by interval length)
- **comparison_report.txt**: Detailed text report with metrics and conclusion

## Output Files

### video_blinks.csv
Detected blinks from video analysis:
- `Blink_Number`: Sequential ID
- `Start_Frame`, `End_Frame`: Video frame indices
- `Midpoint_Time_Seconds`: Most accurate blink timestamp
- `Duration_Seconds`: How long eye was closed
- `Min_EAR`: Minimum Eye Aspect Ratio during blink

### ir_blinks.csv
Detected blinks from IR sensor:
- Same structure as video_blinks.csv
- Timestamps derived from IR sensor data

### blink_timeline.png
Visual representation:
- **Horizontal axis**: Time (seconds)
- **Colored rectangles**: Blink events
  - **Teal** (#4ECDC4): Normal interval (< 6 seconds)
  - **Red** (#FF6B6B): Long interval (> 6 seconds)

### comparison_report.txt
Human-readable summary:
- Video and IR blink metrics (count, rate, average interval)
- Long interval analysis (count and percentage > 6s)
- Assessment: `[OK] NORMAL` or `[!] REDUCED BLINKING`
- Recommendations if reduced blinking detected

## Configuration & Tuning

### Adjust Thresholds

Edit `blink_workflow_fixed.py`:

**Video blink detection (line 34)**
```python
ear_threshold=0.18  # Lower = more sensitive to closed eyes
```

**IR blink detection (line 246)**
```python
threshold = 0.5  # Adjust if IR values range 0-100 (use 50) or 0-1 (use 0.5)
```

**Long interval definition (line 381)**
```python
long_intervals = intervals[intervals > 6.0]  # Change 6.0 to desired threshold
```

**Reduced blinking threshold (line 388)**
```python
LONG_INTERVAL_THRESHOLD_PERCENT = 20.0  # % of intervals > 6s to flag as reduced
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2'"
**Solution**: Ensure virtual environment is activated and packages are installed:
```bash
.\blink\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "No blinks detected in video"
**Causes & fixes**:
- Video too dark or low quality → Try adjusting `ear_threshold` (line 34)
- Face not visible → Ensure camera faces the user
- Video codec unsupported → Convert to MP4 using ffmpeg:
  ```bash
  ffmpeg -i input.avi -c:v libx264 output.mp4
  ```

### Issue: "No blinks detected in IR data"
**Causes & fixes**:
- IR column name not recognized → Manually specify in code (line 196)
- Threshold too high → Lower `threshold` in line 246
- Signal values don't match expected range → Check IR data statistics (printed to console)

### Issue: Timestamps fail to parse
**Solution**: The code auto-fixes common malformed ISO8601 dates. If issues persist, ensure CSV timestamps are in format: `YYYY-MM-DDTHH:MM:SS.ffffff`

## Sharing This Project

### Option 1: ZIP Archive (Recommended)
```powershell
# Create archive (excludes venv and results)
$files = @(
    'blink_detector_accurate_timestamps.py',
    'blink_workflow_fixed.py',
    'requirements.txt',
    'README.md',
    'setup.ps1'
)
Compress-Archive -Path $files -DestinationPath 'BlinkProject.zip'
```

### Option 2: GitHub Repository
```bash
git init
git add -A
git commit -m "Initial commit: Blink detection workflow"
git remote add origin https://github.com/yourusername/blink-detection
git push -u origin main
```

**`.gitignore` content:**
```
blink/
results/
*.pyc
__pycache__/
.DS_Store
*.csv
*.png
```

### Option 3: Share with Instructions
Include:
1. This README.md
2. All Python files (`.py`)
3. `requirements.txt`
4. `setup.ps1`
5. Sample IR data (anonymized)

Recipients can then:
```bash
git clone <repo> && cd blink-detection
.\setup.ps1
python blink_workflow_fixed.py --video test.mp4 --ir-data test_ir.csv
```

## Performance Notes

- **Video processing**: ~1–2 minutes per 10-minute video (depends on resolution, hardware)
- **IR processing**: Near-instant (CSV-based)
- **Recommended RAM**: 4+ GB
- **Disk space**: ~500MB for dependencies + output

## Advanced: Custom IR Detection

If IR signal is very noisy, enable smoothing by editing `blink_workflow_fixed.py` line 238:

```python
# Add after loading ir_data:
ir_data[blink_col] = ir_data[blink_col].rolling(window=5, center=True).mean()
```

This applies a moving median filter to reduce false detections.

## License

[Add your license here, e.g., MIT, GPL, etc.]

## Support & Contributions

- **Issues**: Check troubleshooting section above
- **Improvements**: Submit pull requests on GitHub
- **Questions**: Contact project maintainer

---

**Last Updated**: December 8, 2025  
**Python Version**: 3.8+  
**Tested On**: Windows 10/11, Python 3.10, 3.11
