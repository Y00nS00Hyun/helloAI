"""
모델 정의 패키지 및 레지스트리
"""
from .bilstm import BiLSTMModel
from .transformer import TransformerModel

# 모델 레지스트리 (필수)
MODEL_REGISTRY = {
    "bilstm": BiLSTMModel,
    "transformer": TransformerModel,
}

__all__ = ['BiLSTMModel', 'TransformerModel', 'MODEL_REGISTRY']
