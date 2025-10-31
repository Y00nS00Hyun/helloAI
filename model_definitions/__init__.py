"""
모델 정의 패키지 및 레지스트리 (C. 모델명 혼동 제거)
"""
from .bilstm import BiLSTMModel
from .transformer import TransformerModel, CNNModel

# 모델 레지스트리 (C. transformer -> cnn으로 명확화)
MODEL_REGISTRY = {
    "bilstm": BiLSTMModel,
    "transformer": TransformerModel,  # 호환성 유지 (CNN 모델)
    "cnn": TransformerModel,  # 명확한 이름 추가
}

__all__ = ['BiLSTMModel', 'TransformerModel', 'CNNModel', 'MODEL_REGISTRY']
