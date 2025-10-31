"""
상수 정의
"""
from pathlib import Path
from typing import Dict, Any

# 기본 경로
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_CONFIG_DIR = Path("configs")

# 기본 설정값
DEFAULT_CONFIG: Dict[str, Any] = {
    'model': {
        'vocab_size': 30000,
        'embedding_dim': 300,
        'hidden_dim': 256,
        'num_layers': 2,
        'num_classes': 2,
        'dropout': 0.3
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'val_ratio': 0.2,
        'random_state': 42,
        'optimizer': 'adam',
        'weight_decay': 0.0,
        'scheduler': None
    },
    'tokenizer': {
        'vocab_size': 30000,
        'max_length': 512,
        'min_freq': 2
    },
    'data': {
        'data_dir': 'data'
    }
}

# 라벨 매핑
LABEL_MAPPING = {
    'real': 0,
    'fake': 1,
    0: 0,
    1: 1,
    '0': 0,
    '1': 1,
    True: 1,
    False: 0
}

# 특수 토큰
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1

