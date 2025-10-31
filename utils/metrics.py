"""
평가 메트릭 유틸리티 (Macro F1 Score 사용)
"""
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from typing import Dict
import numpy as np
import torch


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    분류 메트릭 계산 (Macro F1 Score 사용)

    Args:
        y_true: 실제 라벨 (0=Real, 1=Fake)
        y_pred: 예측 라벨

    Returns:
        메트릭 딕셔너리
    """
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
