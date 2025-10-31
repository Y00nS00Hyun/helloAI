"""
데이터 로딩 및 전처리 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import os


def load_datasets(data_dir: str = "data") -> pd.DataFrame:
    """
    3개의 데이터셋을 로드하고 통합

    Args:
        data_dir: 데이터 디렉토리 경로

    Returns:
        통합된 DataFrame
    """
    data_dir = Path(data_dir)
    dfs = []

    # dataset_1.csv 및 dataset_2.csv (title, text 포함)
    for i in [1, 2]:
        file_path = data_dir / f"dataset_{i}.csv"
        if file_path.exists():
            try:
                # 다양한 구분자 시도
                df = pd.read_csv(file_path, encoding='utf-8',
                                 sep=',', on_bad_lines='skip')
            except:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8',
                                     sep=';', on_bad_lines='skip')
                except:
                    df = pd.read_csv(
                        file_path, encoding='utf-8', on_bad_lines='skip')

            # title과 text를 결합
            if 'title' in df.columns and 'text' in df.columns:
                df['text'] = df['title'].astype(
                    str) + ' ' + df['text'].astype(str)
            elif 'title' in df.columns:
                df['text'] = df['title'].astype(str)

            # label 컬럼 확인
            if 'label' in df.columns:
                dfs.append(df[['text', 'label']])
            else:
                # label이 없는 경우 (테스트 데이터 가능성)
                dfs.append(df[['text']])

    # dataset_3.csv (text만 존재)
    file_path = data_dir / "dataset_3.csv"
    if file_path.exists():
        try:
            df = pd.read_csv(file_path, encoding='utf-8',
                             sep=',', on_bad_lines='skip')
        except:
            try:
                df = pd.read_csv(file_path, encoding='utf-8',
                                 sep=';', on_bad_lines='skip')
            except:
                df = pd.read_csv(file_path, encoding='utf-8',
                                 on_bad_lines='skip')

        if 'text' in df.columns:
            if 'label' in df.columns:
                dfs.append(df[['text', 'label']])
            else:
                dfs.append(df[['text']])

    # 모든 데이터프레임 결합
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)

        # 중복 제거
        if 'label' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(
                subset=['text'], keep='first')

        return combined_df
    else:
        return pd.DataFrame()


def preprocess_text(text: str) -> str:
    """
    텍스트 전처리

    Args:
        text: 원본 텍스트

    Returns:
        전처리된 텍스트
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # 기본 정제
    text = text.strip()
    text = ' '.join(text.split())  # 여러 공백을 하나로

    return text


def split_train_val(df: pd.DataFrame, val_ratio: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    학습/검증 데이터 분할

    Args:
        df: 전체 데이터프레임
        val_ratio: 검증 데이터 비율
        random_state: 랜덤 시드

    Returns:
        (train_df, val_df)
    """
    if 'label' not in df.columns:
        raise ValueError("DataFrame에 'label' 컬럼이 없습니다.")

    # 라벨별로 분할하여 불균형 고려
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=df['label']  # 라벨 비율 유지
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def prepare_data_for_training(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """
    학습을 위한 데이터 준비

    Args:
        df: 데이터프레임

    Returns:
        (texts, labels)
    """
    texts = df['text'].apply(preprocess_text).tolist()

    if 'label' in df.columns:
        # 라벨을 정수로 변환 (real=1, fake=0 또는 반대)
        labels = []
        for label in df['label']:
            if isinstance(label, str):
                if label.lower() in ['real', '1', 'true']:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(int(label))

        return texts, labels
    else:
        return texts, None
