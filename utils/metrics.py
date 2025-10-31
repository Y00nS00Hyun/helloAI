"""
평가 메트릭 유틸리티 (Macro F1 Score 사용)
"""
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from typing import Dict, Tuple
import numpy as np
import torch


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    분류 메트릭 계산 (Macro F1 Score 사용)

    Args:
        y_true: 실제 라벨 (0=Real, 1=Fake)
        y_pred: 예측 라벨 (또는 확률 배열인 경우 threshold 적용)
        threshold: 확률 배열인 경우 사용할 임계값

    Returns:
        메트릭 딕셔너리
    """
    # 확률 배열인 경우 이진 분류로 변환
    if y_pred.dtype == float and (y_pred > 1.0).any() or (y_pred < 0.0).any():
        y_pred = (y_pred >= threshold).astype(int)
    else:
        y_pred = y_pred.astype(int)

    # Macro F1 Score (대회 평가 기준)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 추가 메트릭
    accuracy = accuracy_score(y_true, y_pred)

    # 클래스별 F1 Score
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_real = f1_per_class[0] if len(f1_per_class) > 0 else 0.0
    f1_fake = f1_per_class[1] if len(f1_per_class) > 1 else 0.0

    # Precision, Recall
    precision_macro = precision_score(
        y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(
        y_true, y_pred, average='macro', zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # B. 라벨 매핑 검증 로그
    print(f"\n[B. 라벨 매핑 검증]")
    print(f"  Real (0) - TP: {tn}, FP: {fn}")
    print(f"  Fake (1) - TP: {tp}, FP: {fp}")
    print(
        f"  Macro F1: {macro_f1:.4f} (Real: {f1_real:.4f}, Fake: {f1_fake:.4f})")

    return {
        'macro_f1': float(macro_f1),  # 대회 평가 기준
        'f1_score': float(macro_f1),  # 호환성
        'f1_real': float(f1_real),
        'f1_fake': float(f1_fake),
        'accuracy': float(accuracy),
        'precision': float(precision_macro),
        'recall': float(recall_macro),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric: str = 'macro_f1',
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    step: float = 0.01
) -> Tuple[float, Dict]:
    """
    최적 분류 임계값 탐색 (D. 임계값 튜닝)

    Args:
        y_true: 실제 라벨 (0=Real, 1=Fake)
        y_probs: Fake 클래스에 대한 예측 확률
        metric: 최적화할 메트릭 ('macro_f1', 'f1_fake', 'f1_real')
        threshold_range: 탐색할 임계값 범위
        step: 탐색 스텝

    Returns:
        (최적 임계값, 해당 임계값에서의 메트릭)
    """
    best_threshold = 0.5
    best_score = 0.0
    best_metrics = {}

    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred)

        if metric == 'macro_f1':
            score = metrics['macro_f1']
        elif metric == 'f1_fake':
            score = metrics['f1_fake']
        elif metric == 'f1_real':
            score = metrics['f1_real']
        else:
            score = metrics['macro_f1']

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """
    분류 리포트 출력

    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
    """
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    print("="*50)

    # Macro F1 Score 강조
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n🌟 Macro F1 Score (대회 평가 기준): {macro_f1:.4f}")
    print("="*50 + "\n")
