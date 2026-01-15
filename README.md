# RideView

**Torque Stripe Verification System**

Real-time computer vision system for detecting broken or compromised torque stripes on industrial bolts. Point a USB camera at torque-marked bolts and get instant PASS/FAIL feedback.

## What are Torque Stripes?

Torque stripes (also called torque seal or anti-loosening marks) are painted lines applied across bolt heads and their mating surfaces. When a bolt loosens, the stripe breaks or shows a visible gap. This system automates the visual inspection process.

```
INTACT STRIPE (PASS)          BROKEN STRIPE (FAIL)
┌─────────────────┐           ┌─────────────────┐
│  ══════════════ │           │  ═════    ═════ │
│     [BOLT]      │           │     [BOLT]      │
│  ══════════════ │           │  ═════    ═════ │
└─────────────────┘           └─────────────────┘
  Continuous line               Gap = loosening
```

## Quick Start (5 Steps)

### Prerequisites
- Windows 10/11
- Python 3.11 or higher
- USB webcam (ArduCam UVC or any standard webcam)

### Installation

```batch
# 1. Clone the repository
git clone https://github.com/Mikecranesync/RideView.git
cd RideView

# 2. Run the setup script
setup.bat

# 3. Test with your camera
uv run python -m rideview

# 4. (Optional) Start web interface for remote viewing
uv run python -m rideview --web
# Then open http://localhost:5000 in your browser
```

### First Run

1. Connect your USB camera
2. Run `uv run python -m rideview`
3. Point camera at a torque-striped bolt
4. See real-time PASS/FAIL overlay

## Features

- **Real-time Detection**: Process live video at 30+ FPS
- **PASS/WARNING/FAIL Classification**: Clear status indication
- **Color Calibration**: Tune detection for red, yellow, or orange stripes
- **Web Interface**: View results remotely on your phone
- **Configurable Thresholds**: Adjust sensitivity for your use case
- **Snapshot Capture**: Save images for documentation

## Keyboard Controls (Live Mode)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save snapshot |
| `c` | Open color calibration |
| `r` | Reset region of interest |
| `Space` | Pause/Resume |

## Configuration

Edit `config/default.yaml` to customize:

```yaml
# Detection thresholds
thresholds:
  pass:
    min_coverage: 0.95  # 95%+ coverage = PASS
    max_gaps: 0         # No gaps allowed for PASS
  warning:
    min_coverage: 0.80  # 80%+ = WARNING
    max_gaps: 2         # Up to 2 small gaps

# Camera settings
camera:
  source: 0           # Camera index (0 = first camera)
  width: 1280
  height: 720
  fps: 30

# Color ranges (HSV)
colors:
  enabled_colors:
    - red
    - yellow
    - orange
```

## Hardware Setup

### Recommended Camera Position
- **Distance**: 6-12 inches from bolt
- **Angle**: Perpendicular to stripe
- **Lighting**: Diffuse lighting, avoid glare
- **Focus**: Ensure stripe is sharp

### ArduCam UVC Setup
1. Connect via USB OTG adapter
2. Camera should appear as device index 0
3. If not detected, try `RIDEVIEW_CAMERA_SOURCE=1`

## Project Structure

```
RideView/
├── src/rideview/           # Main application code
│   ├── core/               # Camera, config, detector
│   ├── detection/          # Detection algorithms
│   ├── web/                # Flask web interface
│   └── utils/              # Visualization helpers
├── config/                 # YAML configuration files
├── tests/                  # Unit tests
├── scripts/                # Utility scripts
└── samples/                # Sample images
```

## Detection Algorithm

1. **Preprocessing**: Gaussian blur + CLAHE contrast enhancement
2. **Color Segmentation**: HSV-based isolation of stripe colors
3. **Morphological Cleanup**: Close gaps, remove noise
4. **Line Analysis**: Detect discontinuities using projection profile
5. **Classification**: Apply thresholds for PASS/WARNING/FAIL

## API Endpoints (Web Mode)

When running with `--web`:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard |
| `GET /stream/video_feed` | MJPEG video stream with overlay |
| `GET /stream/raw_feed` | Raw video without processing |
| `GET /api/status` | Current detection status (JSON) |
| `GET /api/config` | Current configuration |
| `POST /api/config` | Update configuration |
| `POST /api/snapshot` | Capture and save snapshot |

## Troubleshooting

### Camera not detected
```batch
# List available cameras
uv run python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Try different camera index
set RIDEVIEW_CAMERA_SOURCE=1
uv run python -m rideview
```

### Poor detection accuracy
1. Run color calibration: `uv run python scripts/calibrate_colors.py`
2. Adjust HSV ranges in `config/default.yaml`
3. Improve lighting conditions
4. Clean camera lens

### Low frame rate
- Reduce resolution in config
- Close other applications
- Use wired USB (not hub)

## Development

```batch
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest tests -v

# Run linting
uv run ruff check src tests

# Format code
uv run ruff format src tests
```

## Roadmap

- [x] Phase 1: Windows desktop application
- [ ] Phase 2: Android app (React Native / Expo)
- [ ] Phase 3: Batch processing mode
- [ ] Phase 4: ML-enhanced detection

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request
