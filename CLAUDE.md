# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RideView is a real-time computer vision system for detecting broken or compromised torque stripes on industrial bolts. It provides PASS/WARNING/FAIL classification through live video analysis with an optional web interface.

## Common Commands

```bash
# Install dependencies
uv sync --all-extras

# Run live detection mode (OpenCV window)
uv run python -m rideview

# Run with web interface
uv run python -m rideview --web

# Run tests
uv run pytest tests -v

# Run single test file
uv run pytest tests/unit/test_line_analyzer.py -v

# Run tests with coverage
uv run pytest tests --cov=src/rideview

# Lint
uv run ruff check src tests

# Format
uv run ruff format src tests

# Type check
uv run mypy src
```

## Architecture

### Detection Pipeline

The core detection flow is orchestrated by `TorqueStripeDetector` (src/rideview/core/detector.py):

```
Frame → Preprocessor → ColorSegmenter → Morphology → LineAnalyzer → StripeValidator → Result
```

1. **Preprocessor** (`detection/preprocessor.py`): Gaussian blur + CLAHE contrast enhancement
2. **ColorSegmenter** (`detection/color_segmenter.py`): HSV-based isolation of stripe colors (red, yellow, orange). Red requires two HSV ranges since it wraps around hue 0.
3. **Morphology** (in detector.py): Close operation fills small gaps, open operation removes noise
4. **LineAnalyzer** (`detection/line_analyzer.py`): Uses projection profile analysis to detect gaps along the stripe direction. Returns `LineMetrics` with coverage, gap count, max gap size.
5. **StripeValidator** (`detection/stripe_validator.py`): Classifies based on thresholds → `DetectionResult` enum (PASS/WARNING/FAIL/NO_STRIPE)

### Result Types

- `DetectionResult` (enum): PASS, WARNING, FAIL, NO_STRIPE
- `StripeAnalysis` (dataclass): Full analysis including result, confidence, coverage, gap metrics, masks, and annotated frame
- `LineMetrics` (dataclass): Raw metrics from line continuity analysis

### Entry Points

- **CLI**: `src/rideview/__main__.py` - argparse-based, supports `--web`, `--camera N`, `--config path`
- **Web**: `src/rideview/web/app.py` - Flask application with MJPEG streaming

### Web Routes

- `/stream/video_feed` - MJPEG stream with detection overlay
- `/stream/raw_feed` - Raw video without processing
- `/api/status` - Current detection status (JSON)
- `/api/config` - GET/POST configuration
- `/api/snapshot` - Capture snapshot

### Configuration

YAML-based configuration in `config/` directory. Key sections:
- `camera`: source index, resolution, fps
- `colors`: HSV ranges for stripe detection
- `thresholds`: pass/warning classification thresholds (min_coverage, max_gaps, max_gap_size)
- `morphology`: kernel sizes for cleanup operations
- `line_analysis`: gap detection parameters

Environment overrides: `RIDEVIEW_CAMERA_SOURCE`, `RIDEVIEW_ENV`

## Testing

Tests use pytest fixtures from `tests/conftest.py` that generate synthetic stripe images:
- `sample_intact_stripe` / `sample_intact_mask` - continuous line
- `sample_broken_stripe` / `sample_broken_mask` - line with gap
- `sample_warning_stripe` - partial degradation
- `test_config` - standard test configuration dict
