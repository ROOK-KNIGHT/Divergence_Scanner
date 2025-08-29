"""
Configuration loader for the Divergence Scanner.
Handles loading and validation of configuration from YAML file.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional

class ConfigLoader:
    """Loads and manages configuration for the Divergence Scanner."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_file (str): Path to the configuration file
        """
        self.config_file = config_file
        self.config = None
        self.logger = logging.getLogger(__name__)
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"Configuration file {self.config_file} not found")
            
            with open(self.config_file, 'r') as file:
                self.config = yaml.safe_load(file)
            
            self.logger.info(f"Configuration loaded successfully from {self.config_file}")
            self._validate_config()
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate that required configuration sections exist."""
        required_sections = [
            'scanning', 'indicators', 'divergence', 'trading', 
            'market', 'symbols', 'notifications', 'logging', 'api'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        self.logger.info("Configuration validation passed")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path (str): Dot-separated path to the configuration value
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value or default
            
        Example:
            config.get('indicators.rsi.period', 14)
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section (str): Section name
            
        Returns:
            Dict[str, Any]: Configuration section or empty dict
        """
        return self.config.get(section, {})
    
    def update_config(self, key_path: str, value: Any) -> None:
        """
        Update a configuration value using dot notation.
        
        Args:
            key_path (str): Dot-separated path to the configuration value
            value (Any): New value to set
        """
        try:
            keys = key_path.split('.')
            config_ref = self.config
            
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]
            
            # Set the value
            config_ref[keys[-1]] = value
            self.logger.info(f"Updated configuration: {key_path} = {value}")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration {key_path}: {str(e)}")
            raise
    
    def save_config(self, output_file: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_file (str, optional): Output file path. Uses original file if None.
        """
        try:
            file_path = output_file or self.config_file
            
            with open(file_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.load_config()
    
    # Convenience methods for commonly used configuration values
    
    def get_scanning_config(self) -> Dict[str, Any]:
        """Get scanning configuration."""
        return self.get_section('scanning')
    
    def get_indicators_config(self) -> Dict[str, Any]:
        """Get indicators configuration."""
        return self.get_section('indicators')
    
    def get_divergence_config(self) -> Dict[str, Any]:
        """Get divergence detection configuration."""
        return self.get_section('divergence')
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading strategy configuration."""
        return self.get_section('trading')
    
    def get_market_config(self) -> Dict[str, Any]:
        """Get market configuration."""
        return self.get_section('market')
    
    def get_symbols_config(self) -> Dict[str, Any]:
        """Get symbols configuration."""
        return self.get_section('symbols')
    
    def get_notifications_config(self) -> Dict[str, Any]:
        """Get notifications configuration."""
        return self.get_section('notifications')
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get_section('api')
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get_section('logging')
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.get_section('performance')
    
    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug configuration."""
        return self.get_section('debug')


# Global configuration instance
_config_instance = None

def get_config(config_file: str = "config.yaml") -> ConfigLoader:
    """
    Get the global configuration instance.
    
    Args:
        config_file (str): Path to configuration file
        
    Returns:
        ConfigLoader: Configuration loader instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigLoader(config_file)
    
    return _config_instance

def reload_config() -> None:
    """Reload the global configuration."""
    global _config_instance
    
    if _config_instance is not None:
        _config_instance.reload_config()
