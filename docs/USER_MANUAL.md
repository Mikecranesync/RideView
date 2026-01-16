# RideView User Manual

**Version 0.1.0**

RideView is a real-time computer vision system for detecting broken or compromised torque stripes on industrial bolts. It provides PASS/WARNING/FAIL classification through live video analysis.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding Detection Results](#understanding-detection-results)
3. [Controls Reference](#controls-reference)
4. [Adjusting Pass/Fail Criteria](#adjusting-passfail-criteria)
5. [Web Interface Guide](#web-interface-guide)
6. [Configuration Reference](#configuration-reference)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Requirements

- Python 3.11 or higher
- uv package manager
- Webcam or video source

### Installation

```bash
# Install dependencies
uv sync --all-extras

# Run live detection (OpenCV window)
uv run python -m rideview

# Run with web interface
uv run python -m rideview --web
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--web` | Start web server instead of live detection window |
| `--camera N` | Use camera index N (0 = first camera, 1 = second, etc.) |
| `--config PATH` | Path to custom configuration directory |
| `--debug` | Enable debug logging |

**Examples:**

```bash
uv run python -m rideview                    # Default camera, live window
uv run python -m rideview --web              # Web interface at http://localhost:5000
uv run python -m rideview --camera 1         # Use second camera
uv run python -m rideview --camera video.mp4 # Use video file
```

---

## Understanding Detection Results

RideView classifies torque stripes into four categories:

### Status Colors

| Status | Color | Meaning |
|--------|-------|---------|
| **PASS** | Green | Stripe intact - 95%+ coverage, no gaps detected |
| **WARNING** | Orange | Minor degradation - 80%+ coverage, up to 2 small gaps |
| **FAIL** | Red | Significant damage - below 80% coverage or more than 2 gaps |
| **NO_STRIPE** | Gray | No stripe detected in frame |

### What Each Status Means

**PASS (Green)**
- The torque stripe is in excellent condition
- Coverage is 95% or higher
- No discontinuities or gaps detected
- The bolt has not been tampered with

**WARNING (Orange)**
- The stripe shows minor wear or degradation
- Coverage is between 80% and 95%
- Up to 2 small gaps detected (each ≤20 pixels)
- Recommend closer inspection

**FAIL (Red)**
- The stripe is significantly damaged or broken
- Coverage below 80%, OR
- More than 2 gaps detected, OR
- A gap larger than 20 pixels exists
- The bolt may have been loosened or tampered with

**NO_STRIPE (Gray)**
- No valid stripe found in the camera view
- Could indicate: stripe not in frame, poor lighting, or stripe completely missing

### Display Metrics

The interface shows these real-time metrics:

- **Coverage** - Percentage of expected stripe area that is visible (0-100%)
- **Gaps** - Number of discontinuities detected in the stripe
- **Max Gap** - Size of the largest gap in pixels
- **Confidence** - Detection confidence level (0-100%)
- **Processing** - Time to analyze frame in milliseconds

---

## Controls Reference

### Keyboard Controls (Live Detection Mode)

| Key | Action | Description |
|-----|--------|-------------|
| `q` | Quit | Close the application |
| `s` | Snapshot | Save current frame to `snapshots/` folder |
| `c` | Comparison | Toggle side-by-side comparison view |
| `Space` | Pause | Freeze video for inspection |
| `r` | Reset ROI | Clear Region of Interest selection |

### Mouse Controls

- **Click and drag** - Draw a rectangle to select Region of Interest (ROI)
- Detection will focus only on the selected area
- Press `r` to reset and use full frame

### Comparison View

Press `c` to toggle comparison mode, which shows:
1. Original frame
2. Color mask (detected stripe pixels)
3. Annotated result

This is useful for understanding what the detector is seeing and for calibration.

---

## Adjusting Pass/Fail Criteria

### Method 1: Edit Configuration File

Edit `config/default.yaml`:

```yaml
thresholds:
  pass:
    min_coverage: 0.95    # Minimum coverage for PASS (95%)
    max_gaps: 0           # Maximum gaps allowed for PASS
    max_gap_size: 0       # Maximum gap size for PASS (pixels)

  warning:
    min_coverage: 0.80    # Minimum coverage for WARNING (80%)
    max_gaps: 2           # Maximum gaps allowed for WARNING
    max_gap_size: 20      # Maximum gap size for WARNING (pixels)

  min_confidence: 0.5     # Minimum confidence to make classification
```

### Method 2: Environment Variables

Override any threshold using `RIDEVIEW_` prefix:

```bash
# Make PASS stricter (require 98% coverage)
set RIDEVIEW_THRESHOLDS_PASS_MIN_COVERAGE=0.98
uv run python -m rideview

# Make WARNING more lenient (allow 3 gaps)
set RIDEVIEW_THRESHOLDS_WARNING_MAX_GAPS=3
uv run python -m rideview

# Combine multiple overrides
set RIDEVIEW_THRESHOLDS_PASS_MIN_COVERAGE=0.90
set RIDEVIEW_THRESHOLDS_WARNING_MIN_COVERAGE=0.70
uv run python -m rideview
```

### Method 3: Web API (Runtime)

Send a POST request to update thresholds:

```bash
curl -X POST http://localhost:5000/api/config \
  -H "Content-Type: application/json" \
  -d '{"thresholds": {"pass": {"min_coverage": 0.90}}}'
```

### Common Adjustments

**Stricter Detection (fewer false positives):**
```yaml
thresholds:
  pass:
    min_coverage: 0.98
    max_gaps: 0
  warning:
    min_coverage: 0.90
    max_gaps: 1
    max_gap_size: 10
```

**More Lenient Detection (fewer false negatives):**
```yaml
thresholds:
  pass:
    min_coverage: 0.90
    max_gaps: 1
    max_gap_size: 5
  warning:
    min_coverage: 0.70
    max_gaps: 3
    max_gap_size: 30
```

---

## Web Interface Guide

Start the web interface:
```bash
uv run python -m rideview --web
```

Open your browser to: **http://localhost:5000**

### Dashboard Features

**Video Stream**
- Live detection feed with overlay
- Status indicator in top-right corner

**Controls**
- **Snapshot** - Capture current frame
- **Raw Feed** - Toggle to unprocessed video (useful for setup)
- **Refresh** - Restart video stream

**Detection Results Panel**
- Real-time metrics: Status, Coverage, Gaps, Confidence, Processing time
- Analysis text explaining the current classification

**Quick Actions**
- **List Cameras** - Show available camera indices
- **View Config** - See current configuration
- **Color Ranges** - View HSV color detection ranges

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Current detection status (JSON) |
| `/api/config` | GET | Current configuration |
| `/api/config` | POST | Update configuration |
| `/api/colors` | GET | HSV color ranges |
| `/api/snapshot` | POST | Capture and save snapshot |
| `/api/cameras` | GET | List available cameras |
| `/stream/video_feed` | GET | MJPEG stream with detection |
| `/stream/raw_feed` | GET | MJPEG stream without processing |
| `/health` | GET | Health check |

---

## Configuration Reference

### Full Configuration File Structure

Location: `config/default.yaml`

```yaml
# Camera Settings
camera:
  source: 0                # Camera index or video file path
  backend: "CAP_MSMF"      # Windows: CAP_MSMF or CAP_DSHOW
  width: 1280              # Resolution width
  height: 720              # Resolution height
  fps: 30                  # Frame rate
  buffer_size: 1           # Lower = less latency
  auto_exposure: true      # Automatic exposure

# Preprocessing
preprocessing:
  blur_kernel: [5, 5]      # Gaussian blur kernel (odd numbers)
  clahe:
    enabled: true          # Contrast enhancement
    clip_limit: 2.0        # CLAHE clip limit
    tile_grid_size: [8, 8] # CLAHE tile size

# Color Detection (HSV ranges)
colors:
  enabled_colors:
    - red
    - yellow
    - orange

  custom_ranges:
    red_low:
      lower: [0, 100, 100]
      upper: [10, 255, 255]
    red_high:
      lower: [170, 100, 100]
      upper: [179, 255, 255]
    yellow:
      lower: [20, 100, 100]
      upper: [35, 255, 255]
    orange:
      lower: [10, 100, 100]
      upper: [20, 255, 255]

# Morphological Operations
morphology:
  close_kernel: [5, 5]     # Fills small gaps in detection
  close_iterations: 2
  open_kernel: [3, 3]      # Removes noise
  open_iterations: 1

# Line Analysis
line_analysis:
  min_stripe_area: 500     # Minimum pixels for valid stripe
  expected_width_range: [10, 100]
  scan_direction: "horizontal"
  gap_threshold: 5         # Minimum gap size to count

# Classification Thresholds
thresholds:
  pass:
    min_coverage: 0.95
    max_gaps: 0
    max_gap_size: 0
  warning:
    min_coverage: 0.80
    max_gaps: 2
    max_gap_size: 20
  min_confidence: 0.5

# Visualization
visualization:
  pass_color: [0, 255, 0]        # Green (BGR)
  warning_color: [0, 165, 255]   # Orange (BGR)
  fail_color: [0, 0, 255]        # Red (BGR)
  no_stripe_color: [128, 128, 128]
  font_scale: 1.0
  show_mask_overlay: true
  mask_opacity: 0.3

# Web Server
web:
  host: "0.0.0.0"
  port: 5000
  stream_quality: 85       # JPEG quality (0-100)
  max_stream_fps: 15

# Snapshots
snapshots:
  directory: "snapshots"
  filename_format: "%Y%m%d_%H%M%S_{result}.jpg"
  include_overlay: true

# Logging
logging:
  level: "INFO"
  file: "logs/rideview.log"
```

### Environment Variables

All config values can be overridden with `RIDEVIEW_` prefix:

| Variable | Example | Description |
|----------|---------|-------------|
| `RIDEVIEW_ENV` | development | Environment (development/production) |
| `RIDEVIEW_CAMERA_SOURCE` | 1 | Camera index |
| `RIDEVIEW_CAMERA_WIDTH` | 1920 | Video width |
| `RIDEVIEW_CAMERA_HEIGHT` | 1080 | Video height |
| `RIDEVIEW_WEB_PORT` | 8080 | Web server port |
| `RIDEVIEW_LOGGING_LEVEL` | DEBUG | Log level |

---

## Troubleshooting

### Camera Not Detected

**Error:** `Failed to open camera source: 0`

**Solutions:**
1. Check camera is connected
2. Try different camera index: `--camera 1`
3. On Windows, try different backend in config:
   ```yaml
   camera:
     backend: "CAP_DSHOW"  # or "CAP_MSMF"
   ```
4. List available cameras via web API: `GET /api/cameras`

### No Stripe Detected

**Possible causes:**
1. Stripe not in frame - position camera correctly
2. Poor lighting - ensure adequate illumination
3. Wrong color range - stripe color not matching configured HSV ranges
4. Use comparison view (`c` key) to see what detector is seeing

**To adjust color detection:**
```yaml
colors:
  custom_ranges:
    red_low:
      lower: [0, 80, 80]     # Lower saturation/value for dim lighting
      upper: [10, 255, 255]
```

### High Latency / Low FPS

1. Reduce resolution:
   ```yaml
   camera:
     width: 640
     height: 480
   ```
2. Increase buffer size (adds latency but smoother):
   ```yaml
   camera:
     buffer_size: 2
   ```
3. Disable CLAHE preprocessing:
   ```yaml
   preprocessing:
     clahe:
       enabled: false
   ```

### WSL Camera Access

WSL cannot directly access Windows cameras. Run RideView on Windows directly:

```cmd
cd C:\Users\YourName\Documents\GitHub\RideView
uv sync --all-extras
uv run python -m rideview
```

---

## Support

For issues and feature requests, visit:
https://github.com/anthropics/RideView/issues
