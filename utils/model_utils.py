"""
모델 생성 유틸리티
"""
from typing import Dict, Any, Optional, Type
import torch
import torch.nn as nn

from model_definitions import MODEL_REGISTRY
from utils.preprocessing import TextTokenizer


def create_model(
    model_name: str,
    vocab_size: int,
    config: Dict[str, Any],
    device: torch.device
) -> nn.Module:
    """
    모델 생성 (레지스트리 기반)
    
    Args:
        model_name: 모델 이름 (레지스트리의 키)
        vocab_size: 어휘 크기
        config: 모델 설정 딕셔너리
        device: 디바이스
    
    Returns:
        생성된 모델
    
    Raises:
        ValueError: 모델이 레지스트리에 없거나 파라미터가 잘못된 경우
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {available}"
        )
    
    ModelClass = MODEL_REGISTRY[model_name]
    model_config = config.get('model', {})
    
    # 모델별 파라미터 설정
    if model_name == 'bilstm':
        model = ModelClass(
            vocab_size=vocab_size,
            embedding_dim=model_config.get('embedding_dim', 300),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 2),
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3)
        )
    elif model_name == 'transformer':
        # CNN 모델 사용 (Transformer 대신)
        model = ModelClass(
            model_name=None,
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3),
            vocab_size=vocab_size
        )
    else:
        # 다른 모델의 경우 기본 파라미터로 생성 시도
        try:
            model = ModelClass(
                vocab_size=vocab_size,
                num_classes=model_config.get('num_classes', 2),
                dropout=model_config.get('dropout', 0.3),
                **{k: v for k, v in model_config.items() 
                   if k not in ['name', 'model_name']}
            )
        except TypeError as e:
            raise ValueError(
                f"Model {model_name} requires specific parameters. Error: {e}"
            )
    
    return model.to(device)


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    옵티마이저 생성
    
    Args:
        model: 모델
        config: 학습 설정 딕셔너리
    
    Returns:
        옵티마이저
    """
    train_config = config.get('training', {})
    lr = train_config.get('learning_rate', 0.001)
    weight_decay = train_config.get('weight_decay', 0.0)
    optimizer_name = train_config.get('optimizer', 'adam').lower()
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    else:
        return torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    학습률 스케줄러 생성
    
    Args:
        optimizer: 옵티마이저
        config: 학습 설정 딕셔너리
    
    Returns:
        스케줄러 (없으면 None)
    """
    scheduler_type = config.get('training', {}).get('scheduler')
    num_epochs = config.get('training', {}).get('num_epochs', 10)
    
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    elif scheduler_type == 'linear_warmup':
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(step: int) -> float:
            warmup_steps = int(0.1 * num_epochs * 1000)
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        return LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )
    else:
        return None

