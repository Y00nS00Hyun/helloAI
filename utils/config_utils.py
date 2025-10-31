"""
설정 파일 로드 유틸리티
"""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from utils.constants import DEFAULT_CONFIG


def load_config(
    config_path: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로 (None이면 기본 경로 사용)
        model_name: 모델 이름 (config_path가 None일 때 사용)
    
    Returns:
        설정 딕셔너리
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    if model_name:
        default_path = Path('configs') / f'{model_name}.yaml'
        if default_path.exists():
            with open(default_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
    
    # 기본 설정 반환
    config = DEFAULT_CONFIG.copy()
    if model_name:
        config['model']['name'] = model_name
    return config

