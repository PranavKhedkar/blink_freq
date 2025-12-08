# Quick Start Guide

## For First-Time Users

### 1. Extract & Navigate
```powershell
# Extract BlinkProject.zip (if applicable)
# Then open PowerShell in the project directory
cd 'C:\Path\To\Blink Freq Project'
```

### 2. Run Setup Script
```powershell
.\setup.ps1
```

This will:
- Check Python installation ✓
- Create virtual environment ✓
- Install all dependencies ✓
- Verify everything works ✓

### 3. Prepare Your Data
You need:
- **Video file** (e.g., `screen_session.mp4`)
  - Formats: MP4, AVI, MOV, WebM
  - Quality: 720p+ recommended
  - Duration: 1–30 minutes typical
  
- **IR Data CSV** (e.g., `ir_baseline.csv`)
  - Columns: `Timestamp` or `Time`, and blink signal (0=open, 1=closed)
  - Format: CSV with headers
  - Sample rate: 30+ Hz recommended

**Example IR CSV:**
```
Timestamp,Blink_Indicator
2025-12-08T10:00:00.000000,0
2025-12-08T10:00:00.033333,0.2
2025-12-08T10:00:00.066667,0.9
2025-12-08T10:00:00.100000,0.8
...
```

### 4. Run Analysis
```powershell
# Activate environment (should be active after setup)
.\blink\Scripts\Activate.ps1

# Run workflow
python blink_workflow_fixed.py `
  --video "path\to\your\video.mp4" `
  --ir-data "path\to\your\ir_data.csv" `
  --output-dir results
```

**Example:**
```powershell
python blink_workflow_fixed.py `
  --video "C:\Data\recording.mp4" `
  --ir-data "C:\Data\ir_sensor.csv" `
  --output-dir results
```

### 5. Check Results
After 1–5 minutes (depending on video length), check:
```
results/
  ├── video_blinks.csv          # Detected blinks from video
  ├── ir_blinks.csv             # Detected blinks from IR
  ├── blink_timeline.png        # Visual timeline
  └── comparison_report.txt     # Analysis & conclusion
```

**Open `comparison_report.txt` to see:**
- Total blinks detected
- Blink rate (blinks per minute)
- Long pauses (> 6 seconds)
- Conclusion: `[OK]` or `[!] REDUCED BLINKING`

---

## Troubleshooting

### Setup Script Fails
**Error: "PowerShell execution policy"**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

**Error: "Python not found"**
- Install Python 3.8+ from https://www.python.org
- Or specify Python path: `.\setup.ps1 -PythonCmd "C:\Python310\python.exe"`

### Workflow Fails at Step 1 (Video)
**"No face detected"**
- Ensure video shows a clear face
- Try adjusting `ear_threshold` in `blink_workflow_fixed.py` line 34

**"Cannot open video"**
- Verify video path is correct and accessible
- Convert video: `ffmpeg -i input.avi -c:v libx264 output.mp4`

### Workflow Fails at Step 2 (IR)
**"No blinks detected in IR data"**
- Check IR CSV columns match `Timestamp` and a signal column
- Check signal values (statistics printed to console)
- Adjust threshold in `blink_workflow_fixed.py` line 246

**"Timestamp parsing failed"**
- Ensure timestamps are in ISO8601 format: `YYYY-MM-DDTHH:MM:SS.ffffff`

---

## Getting Results

### What Do the Results Mean?

**blink_timeline.png:**
- **Teal blocks** = Normal blink pattern (< 6 second gaps)
- **Red blocks** = Long pauses (> 6 second gaps) — may indicate reduced blinking

**comparison_report.txt:**

```
[OK] Normal blinking pattern.
  Video shows 18.5 blinks/min vs IR baseline of 17.2 blinks/min.
  Only 2 intervals exceeded 6 seconds (5.6%).
  Blink pattern is within healthy range.
```

OR

```
[!] ALERT: Reduced blinking detected.
  Video shows 12.1 blinks/min vs IR baseline of 18.3 blinks/min.
  Critical finding: 8 inter-blink intervals exceeded 6 seconds (44.4% of all intervals).
  
  Reasons:
    1. 44.4% of inter-blink intervals exceed 6 seconds
    2. Average inter-blink interval is 5.82s (approaching 6s threshold)
  
  This may indicate:
    - Significant digital eye strain
    - Screen time fatigue
    - Dry eye symptoms requiring attention
  
  Recommendation: Consider taking breaks or adjusting screen settings.
```

---

## Advanced: Custom Configuration

Edit `blink_workflow_fixed.py` to customize:

| Setting | Line | Default | Description |
|---------|------|---------|-------------|
| Video EAR threshold | 34 | 0.18 | Lower = more sensitive |
| IR signal threshold | 246 | 0.5 | Adjust if IR values are 0–100 (use 50) |
| Long interval definition | 381 | 6.0 | Seconds (change to 5.0 or 7.0 as needed) |
| Reduced blinking % | 388 | 20.0 | % of intervals > threshold |

---

## Need Help?

1. **Check README.md** for full documentation
2. **Review console output** — workflow prints detailed diagnostics
3. **Check troubleshooting section** above
4. **Contact project maintainer** with error messages and sample data

---

**Happy analyzing! 👁️**
