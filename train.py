"""
모델 학습 스크립트
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.data_loader import (
    load_datasets,
    split_train_val,
    prepare_data_for_training
)
from utils.preprocessing import TextTokenizer
from utils.metrics import calculate_metrics, print_classification_report
from utils.model_utils import create_model, create_optimizer, create_scheduler
from utils.config_utils import load_config
from utils.constants import DEFAULT_CONFIG
from model_definitions import MODEL_REGISTRY

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsDataset(Dataset):
    """뉴스 데이터셋"""

    def __init__(
        self,
        texts: list,
        labels: list = None,
        tokenizer: TextTokenizer = None
    ):
        """
        Args:
            texts: 텍스트 리스트
            labels: 라벨 리스트 (None일 수 있음)
            tokenizer: 토크나이저
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        if self.tokenizer:
            encoded = self.tokenizer.tokenize(
                [text],
                padding=True,
                truncation=True
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        else:
            # 기본 토크나이제이션 (fallback)
            words = text.split()[:512]
            input_ids = torch.tensor(
                [hash(word) % 30000 for word in words],
                dtype=torch.long
            )
            attention_mask = torch.ones(len(input_ids), dtype=torch.long)

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        if self.labels is not None:
            result['label'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return result


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    한 에포크 학습

    Args:
        model: 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스

    Returns:
        (평균 손실, 메트릭 딕셔너리)
    """
    model.train()
    total_loss = 0.0
    all_preds: list = []
    all_labels: list = []

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds)
    )

    return avg_loss, metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    검증

    Args:
        model: 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        device: 디바이스

    Returns:
        (평균 손실, 메트릭 딕셔너리)
    """
    model.eval()
    total_loss = 0.0
    all_preds: list = []
    all_labels: list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds)
    )

    return avg_loss, metrics


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='Train Fake News Detection Model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help='Model type to train'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )

    args = parser.parse_args()

    # 설정 파일 로드
    config = load_config(config_path=args.config, model_name=args.model)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # 데이터 로드
    logger.info("Loading datasets...")
    data_dir = config['data'].get('data_dir', 'data')
    df = load_datasets(data_dir=data_dir)

    if df.empty:
        raise ValueError(f"No data loaded from {data_dir}")

    logger.info(f"Loaded {len(df)} samples")

    # 학습/검증 분할
    val_ratio = config['training'].get('val_ratio', 0.2)
    random_state = config['training'].get('random_state', 42)
    train_df, val_df = split_train_val(
        df,
        val_ratio=val_ratio,
        random_state=random_state
    )
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # 데이터 준비
    train_texts, train_labels = prepare_data_for_training(train_df)
    val_texts, val_labels = prepare_data_for_training(val_df)

    if train_labels is None or val_labels is None:
        raise ValueError("Labels are required for training")

    # 토크나이저 설정
    tokenizer_config = config.get('tokenizer', {})
    tokenizer = TextTokenizer(
        vocab_size=tokenizer_config.get('vocab_size', 30000),
        max_length=tokenizer_config.get('max_length', 512)
    )

    # 어휘 구축
    logger.info("Building vocabulary...")
    min_freq = tokenizer_config.get('min_freq', 2)
    tokenizer.build_vocab(train_texts, min_freq=min_freq)
    vocab_size = len(tokenizer.tokenizer.word_to_idx)
    logger.info(f"Vocabulary size: {vocab_size}")

    # 데이터셋 및 데이터로더 생성
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer)

    batch_size = config['training'].get('batch_size', 32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # 모델 생성
    model = create_model(
        model_name=args.model,
        vocab_size=vocab_size,
        config=config,
        device=device
    )
    logger.info(f"Model: {args.model}")
    logger.info(
        f"Total parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # 학습 루프
    num_epochs = config['training'].get('num_epochs', 10)
    best_f1 = 0.0
    patience_counter = 0
    early_stopping_config = config['training'].get('early_stopping', {})
    patience = early_stopping_config.get('patience', 5)
    min_delta = early_stopping_config.get('min_delta', 0.001)

    # 모델 저장 디렉토리 생성
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)

    logger.info("="*50)
    logger.info("Starting Training")
    logger.info("="*50)

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 50)

        # 학습
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        logger.info(
            f"Train Loss: {train_loss:.4f}, "
            f"Macro F1: {train_metrics['macro_f1']:.4f}"
        )

        # 검증
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )
        logger.info(
            f"Val Loss: {val_loss:.4f}, "
            f"Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        # 스케줄러 업데이트
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Learning Rate: {current_lr:.6f}")

        # 최고 모델 저장 (Macro F1 Score 기준)
        if val_metrics['macro_f1'] > best_f1 + min_delta:
            best_f1 = val_metrics['macro_f1']
            model_path = model_dir / 'best.pt'
            torch.save(model.state_dict(), model_path)
            logger.info(f"✓ Best model saved! Macro F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # 조기 종료
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info("="*50)
    logger.info("Training Completed!")
    logger.info(f"Best Macro F1 Score: {best_f1:.4f}")
    logger.info("="*50)

    # 최종 검증 리포트
    best_model_path = model_dir / 'best.pt'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        logger.info("\nFinal Validation Results:")
        logger.info(
            f"Macro F1 Score (대회 평가 기준): {val_metrics['macro_f1']:.4f}"
        )
        logger.info(f"F1 Real: {val_metrics['f1_real']:.4f}")
        logger.info(f"F1 Fake: {val_metrics['f1_fake']:.4f}")
        logger.info(f"Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {val_metrics['precision']:.4f}")
        logger.info(f"Recall: {val_metrics['recall']:.4f}")


if __name__ == '__main__':
    main()
