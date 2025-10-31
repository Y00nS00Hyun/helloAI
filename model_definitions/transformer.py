"""
CNN 기반 모델 정의 (Transformer 대신, 사전학습 모델 없이)
"""
import torch
import torch.nn as nn
from typing import Dict


class CNNModel(nn.Module):
    """CNN 기반 가짜뉴스 분류 모델 (사전학습 없음)"""

    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 300,
        num_filters: int = 100,
        filter_sizes: list = [3, 4, 5],
        num_classes: int = 2,
        dropout: float = 0.3,
        model_name: str = None  # 호환성을 위해 유지, 사용하지 않음
    ):
        """
        Args:
            vocab_size: 어휘 크기
            embedding_dim: 임베딩 차원
            num_filters: 필터 개수
            filter_sizes: 필터 크기 리스트
            num_classes: 분류 클래스 수
            dropout: 드롭아웃 비율
            model_name: 사용하지 않음
        """
        super(CNNModel, self).__init__()

        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 여러 크기의 CNN 필터
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        # 분류 레이어
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: 토큰 ID 텐서 [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]

        Returns:
            로짓 텐서 [batch_size, num_classes]
        """
        # 임베딩
        # [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_ids)

        # Conv1d는 (batch, channels, seq_len) 형태를 기대
        # [batch_size, embedding_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)

        # 각 필터 크기별로 Convolution 적용
        conv_outputs = []
        for conv in self.convs:
            # [batch_size, num_filters, new_seq_len]
            conv_out = torch.relu(conv(embedded))
            # Max pooling over the sequence dimension
            # [batch_size, num_filters, 1]
            pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            pooled = pooled.squeeze(2)  # [batch_size, num_filters]
            conv_outputs.append(pooled)

        # 모든 필터 출력 결합
        # [batch_size, len(filter_sizes) * num_filters]
        combined = torch.cat(conv_outputs, dim=1)

        # 분류
        output = self.dropout(combined)
        logits = self.fc(output)  # [batch_size, num_classes]

        return logits


class TransformerModel(nn.Module):
    """
    TransformerModel은 호환성을 위해 CNNModel의 별칭
    (실제 Transformer는 사전학습 모델 필요하므로 사용 불가)
    """

    def __init__(self, model_name: str = None, num_classes: int = 2, dropout: float = 0.3, freeze_encoder: bool = False, vocab_size: int = 30000):
        """
        Args:
            model_name: 사용하지 않음
            num_classes: 분류 클래스 수
            dropout: 드롭아웃 비율
            freeze_encoder: 사용하지 않음
            vocab_size: 어휘 크기
        """
        # CNN 모델을 기본으로 사용
        super(TransformerModel, self).__init__()
        self.model = CNNModel(
            vocab_size=vocab_size,
            embedding_dim=300,
            num_filters=100,
            filter_sizes=[3, 4, 5],
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(input_ids, attention_mask)
