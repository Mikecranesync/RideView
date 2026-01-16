"""
Configuration loader with environment variable support.

Loads configuration from YAML files with hierarchical overrides:
1. config/default.yaml (base configuration)
2. config/{RIDEVIEW_ENV}.yaml (environment-specific)
3. Environment variables (RIDEVIEW_*)
"""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """
    Hierarchical configuration loader.

    Load order (later overrides earlier):
    1. default.yaml
    2. {RIDEVIEW_ENV}.yaml (development, production, etc.)
    3. Environment variables (RIDEVIEW_*)

    Usage:
        config = Config()
        camera_source = config.get('camera.source', 0)
        # or
        camera_source = config['camera']['source']
    """

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize configuration loader.

        Args:
            config_dir: Path to configuration directory. Defaults to project config/
        """
        if config_dir is None:
            # Find config directory relative to this file
            config_dir = Path(__file__).parent.parent.parent.parent / "config"
        self.config_dir = Path(config_dir)
        self.env = os.getenv("RIDEVIEW_ENV", "development")
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load and merge configuration files."""
        config: dict[str, Any] = {}

        # Load default
        default_path = self.config_dir / "default.yaml"
        if default_path.exists():
            with open(default_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        # Load environment-specific
        env_path = self.config_dir / f"{self.env}.yaml"
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                env_config = yaml.safe_load(f) or {}
                config = self._deep_merge(config, env_config)

        # Override with environment variables
        config = self._apply_env_overrides(config)

        return config

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_env_overrides(self, config: dict) -> dict:
        """
        Apply RIDEVIEW_* environment variables.

        Example: RIDEVIEW_CAMERA_SOURCE=1 -> config['camera']['source'] = 1
        """
        prefix = "RIDEVIEW_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                path = key[len(prefix) :].lower().split("_")
                self._set_nested(config, path, self._parse_value(value))
        return config

    def _set_nested(self, d: dict, keys: list, value: Any) -> None:
        """Set a nested dictionary value."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        # Boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Dot-separated path like 'camera.source' or 'thresholds.pass.min_coverage'
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Get top-level configuration section."""
        return self._config.get(key, {})

    @property
    def as_dict(self) -> dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()

    def reload(self) -> None:
        """Reload configuration from files."""
        self._config = self._load_config()
