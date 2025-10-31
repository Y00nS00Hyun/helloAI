"""
모델 학습 스크립트
"""
import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

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
from utils.metrics import calculate_metrics, print_classification_report, find_optimal_threshold
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


# G. 재현성 강화
def set_seed(seed: int = 42):
    """
    재현성을 위한 시드 고정 (G. 재현성 강화)

    Args:
        seed: 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Seed fixed to {seed} for reproducibility")


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
    device: torch.device,
    return_probs: bool = False
) -> Tuple[float, Dict[str, float], Optional[np.ndarray]]:
    """
    검증

    Args:
        model: 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        device: 디바이스
        return_probs: 확률 배열 반환 여부 (D. 임계값 튜닝용)

    Returns:
        (평균 손실, 메트릭 딕셔너리, 확률 배열)
    """
    model.eval()
    total_loss = 0.0
    all_preds: list = []
    all_labels: list = []
    all_probs: list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)

            # Fake 클래스 확률 저장 (D. 임계값 튜닝용)
            if return_probs:
                fake_probs = probabilities[:, 1].cpu().numpy()
                all_probs.extend(fake_probs)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds)
    )

    probs = np.array(all_probs) if return_probs else None

    return avg_loss, metrics, probs


def get_git_sha() -> Optional[str]:
    """
    Git SHA 값 가져오기 (G. 재현성 강화)

    Returns:
        Git SHA 문자열 또는 None
    """
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def save_metadata(
    model_path: Path,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    optimal_threshold: float,
    seed: int,
    selection_criterion: str = "macro_f1"
) -> None:
    """
    학습 메타데이터 저장 (G. 재현성 강화)

    Args:
        model_path: 모델 파일 경로
        config: 설정 딕셔너리
        metrics: 평가 메트릭
        optimal_threshold: 최적 임계값
        seed: 사용된 시드
        selection_criterion: 모델 선택 기준 (기본값: "macro_f1")
    """
    metadata = {
        'model_path': str(model_path),
        'config': config,
        'metrics': metrics,
        'optimal_threshold': optimal_threshold,
        'seed': seed,
        'selection_criterion': selection_criterion,  # 모델 선택 기준 명시
        'macro_f1': metrics.get('macro_f1', 0.0),
        'label_mapping': {
            '0': 'Real (negative)',
            '1': 'Fake (positive)'
        },
        'git_sha': get_git_sha()  # Git SHA 추가
    }

    metadata_path = model_path.parent / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to {metadata_path}")
    logger.info(f"Model selection criterion: {selection_criterion} (macro_f1)")


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
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (G. 재현성 강화)'
    )

    args = parser.parse_args()

    # G. 재현성 강화 - 시드 고정
    set_seed(args.seed)

    # 설정 파일 로드
    config = load_config(config_path=args.config, model_name=args.model)

    # 시드를 config에 저장
    config['training']['random_state'] = args.seed

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

    # E. 클래스 불균형 대응 - 클래스 가중치 계산
    train_labels_arr = np.array(train_labels)
    unique_labels, counts = np.unique(train_labels_arr, return_counts=True)
    total = len(train_labels_arr)

    class_weights = []
    for label in [0, 1]:
        if label in unique_labels:
            idx = np.where(unique_labels == label)[0][0]
            weight = total / (len(unique_labels) * counts[idx])
        else:
            weight = 1.0
        class_weights.append(weight)

    logger.info(f"Class distribution: {dict(zip(unique_labels, counts))}")
    logger.info(
        f"Class weights: Real={class_weights[0]:.3f}, Fake={class_weights[1]:.3f}")

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

    # E. 클래스 불균형 대응 - 가중치 적용 손실 함수
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    logger.info("Using weighted CrossEntropyLoss for class imbalance")

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

        # 검증 (확률 배열도 반환)
        val_loss, val_metrics, val_probs = validate(
            model, val_loader, criterion, device, return_probs=True
        )
        logger.info(
            f"Val Loss: {val_loss:.4f}, "
            f"Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        # D. 임계값 튜닝 - 최적 임계값 탐색
        if val_probs is not None:
            optimal_threshold, optimal_metrics = find_optimal_threshold(
                np.array(val_labels),
                val_probs,
                metric='macro_f1'
            )
            logger.info(
                f"Optimal threshold: {optimal_threshold:.4f} "
                f"(Macro F1: {optimal_metrics['macro_f1']:.4f})"
            )
        else:
            optimal_threshold = 0.5
            optimal_metrics = val_metrics

        # 스케줄러 업데이트
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Learning Rate: {current_lr:.6f}")

        # 최고 모델 저장 (Macro F1 Score 기준)
        current_f1 = optimal_metrics['macro_f1']
        if current_f1 > best_f1 + min_delta:
            best_f1 = current_f1
            model_path = model_dir / 'best.pt'
            torch.save(model.state_dict(), model_path)

            # G. 재현성 강화 - 메타데이터 저장
            save_metadata(
                model_path,
                config,
                optimal_metrics,
                optimal_threshold,
                args.seed,
                selection_criterion="macro_f1"  # B. 모델 선택 기준 명시
            )

            logger.info(f"✓ Best model saved! Macro F1: {best_f1:.4f}")
            logger.info(f"  Optimal threshold: {optimal_threshold:.4f}")
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
        val_loss, val_metrics, val_probs = validate(
            model, val_loader, criterion, device, return_probs=True
        )

        # 최적 임계값 재계산
        if val_probs is not None:
            optimal_threshold, optimal_metrics = find_optimal_threshold(
                np.array(val_labels),
                val_probs,
                metric='macro_f1'
            )
            logger.info(f"\nFinal Optimal Threshold: {optimal_threshold:.4f}")
        else:
            optimal_threshold = 0.5
            optimal_metrics = val_metrics

        logger.info("\nFinal Validation Results:")
        logger.info(
            f"Macro F1 Score (대회 평가 기준): {optimal_metrics['macro_f1']:.4f}"
        )
        logger.info(f"F1 Real: {optimal_metrics['f1_real']:.4f}")
        logger.info(f"F1 Fake: {optimal_metrics['f1_fake']:.4f}")
        logger.info(f"Accuracy: {optimal_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {optimal_metrics['precision']:.4f}")
        logger.info(f"Recall: {optimal_metrics['recall']:.4f}")
        logger.info(f"Optimal Threshold: {optimal_threshold:.4f}")


if __name__ == '__main__':
    main()
