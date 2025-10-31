"""
BiLSTM 모델 정의 (사전학습 모델 없이)
"""
import torch
import torch.nn as nn
from typing import Dict


class BiLSTMModel(nn.Module):
    """BiLSTM 기반 가짜뉴스 분류 모델 (사전학습 없음)"""

    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained_embedding: str = None  # 사용하지 않음 (호환성만)
    ):
        """
        Args:
            vocab_size: 어휘 크기
            embedding_dim: 임베딩 차원
            hidden_dim: LSTM 히든 차원
            num_layers: LSTM 레이어 수
            num_classes: 분류 클래스 수
            dropout: 드롭아웃 비율
            pretrained_embedding: 사용하지 않음 (사전학습 모델 사용 금지)
        """
        super(BiLSTMModel, self).__init__()

        # 임베딩 레이어 (사전학습 없음)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # BiLSTM 레이어
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 양방향이므로 출력 차원은 hidden_dim * 2
        lstm_output_dim = hidden_dim * 2

        # 분류 레이어
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

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

        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: [batch_size, seq_len, hidden_dim * 2]

        # 마지막 타임스텝의 출력 사용
        # 양방향이므로 forward와 backward의 마지막 출력을 결합
        # forward 방향 마지막 레이어 [batch_size, hidden_dim]
        forward_hidden = hidden[-2]
        # backward 방향 마지막 레이어 [batch_size, hidden_dim]
        backward_hidden = hidden[-1]
        # [batch_size, hidden_dim * 2]
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        # 또는 attention mask를 이용한 평균 풀링 (더 좋은 성능)
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len]
            # [batch_size, seq_len, 1]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            # 실제 토큰 위치의 출력만 평균
            # [batch_size, seq_len, hidden_dim * 2]
            masked_lstm_out = lstm_out * mask_expanded
            # [batch_size, hidden_dim * 2]
            sum_pooled = masked_lstm_out.sum(dim=1)
            mask_sum = attention_mask.sum(
                dim=1, keepdim=True).float()  # [batch_size, 1]
            # 0으로 나누기 방지
            mask_sum = torch.clamp(mask_sum, min=1.0)
            # [batch_size, hidden_dim * 2]
            pooled_output = sum_pooled / mask_sum
        else:
            pooled_output = combined_hidden

        # 분류
        output = self.dropout(pooled_output)
        logits = self.fc(output)  # [batch_size, num_classes]

        return logits
