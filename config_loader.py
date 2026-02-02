"""
Configuration Loader for FLQC
Loads and validates configuration from config.yaml
"""
import yaml
import os
from typing import Dict, Any


class Config:
    """Configuration container with dot notation access"""
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __repr__(self):
        return f"Config({self.__dict__})"


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object with nested attribute access
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create config.yaml or specify a valid path."
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['training', 'privacy', 'security', 'data', 'model', 'system']
        missing = [s for s in required_sections if s not in config_dict]
        if missing:
            raise ValueError(f"Missing required configuration sections: {missing}")
        
        return Config(config_dict)
    
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {e}")


def get_default_config() -> Config:
    """
    Returns default configuration if config.yaml is not available.
    Used as fallback.
    """
    default = {
        'training': {
            'num_rounds': 5,
            'epochs_per_round': 1,
            'learning_rates': {'image': 0.01, 'text': 0.001, 'tabular': 0.001},
            'batch_sizes': {'image': 32, 'text': 32, 'tabular': 32},
            'chunk_sizes': {'image': 2000, 'text': 1000, 'tabular': 1000}
        },
        'privacy': {
            'enabled': True,
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'noise_multiplier': 0.01
        },
        'security': {
            'aggregation_method': 'fedavg',
            'norm_threshold': 1500.0,
            'krum_f': 1,
            'trimmed_mean_beta': 0.1,
            'quantum_key_length': 128,
            'enable_encryption': True
        },
        'data': {
            'paths': {
                'images': './data/images',
                'text': './data/test.txt',
                'tabular': './data/table.csv'
            },
            'synthetic': {
                'text_vocab_size': 5000,
                'text_seq_length': 20,
                'tabular_input_dim': 10
            }
        },
        'model': {
            'image': {'input_channels': 3, 'num_classes': 4},
            'text': {'vocab_size': 5000, 'embed_dim': 64, 'hidden_dim': 128, 'num_classes': 2},
            'tabular': {'input_dim': 10, 'num_classes': 3}
        },
        'system': {
            'device': 'auto',
            'seed': 42,
            'log_level': 'INFO'
        },
        'ui': {
            'page_title': 'FLQC Secure Framework',
            'theme': 'dark',
            'show_metrics': True,
            'show_privacy_budget': True
        }
    }
    return Config(default)


if __name__ == "__main__":
    # Test configuration loading
    try:
        cfg = load_config()
        print("✓ Configuration loaded successfully")
        print(f"  Training rounds: {cfg.training.num_rounds}")
        print(f"  Privacy epsilon: {cfg.privacy.epsilon}")
        print(f"  Device: {cfg.system.device}")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        print("  Using default configuration...")
        cfg = get_default_config()
        print("✓ Default configuration loaded")
