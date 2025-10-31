"""
평가 메트릭 유틸리티
"""
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from typing import Dict
import numpy as np
import torch


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    분류 메트릭 계산

    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨

    Returns:
        메트릭 딕셔너리
    """
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'f1_score': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
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
    print(classification_report(y_true, y_pred))
    print("="*50 + "\n")
