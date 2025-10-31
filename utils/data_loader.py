"""
데이터 로딩 및 전처리 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging

from utils.constants import LABEL_MAPPING

logger = logging.getLogger(__name__)


def load_single_dataset(file_path: Path) -> Optional[pd.DataFrame]:
    """
    단일 데이터셋 파일 로드

    Args:
        file_path: 파일 경로

    Returns:
        DataFrame 또는 None
    """
    if not file_path.exists():
        return None

    # 다양한 구분자 시도
    for sep in [',', ';']:
        try:
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                sep=sep,
                on_bad_lines='skip'
            )
            return df
        except Exception as e:
            logger.debug(f"Failed to read {file_path} with sep='{sep}': {e}")
            continue

    # 마지막 시도 (자동 감지)
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        return df
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return None


def load_datasets(data_dir: str = "data") -> pd.DataFrame:
    """
    3개의 데이터셋을 로드하고 통합

    Args:
        data_dir: 데이터 디렉토리 경로

    Returns:
        통합된 DataFrame
    """
    data_dir = Path(data_dir)
    dfs: List[pd.DataFrame] = []

    # dataset_1.csv 및 dataset_2.csv (title, text 포함)
    for i in [1, 2]:
        file_path = data_dir / f"dataset_{i}.csv"
        df = load_single_dataset(file_path)

        if df is not None:
            # title과 text를 결합 (A. [SEP] 사용)
            if 'title' in df.columns and 'text' in df.columns:
                df['text'] = df['title'].astype(
                    str) + ' [SEP] ' + df['text'].astype(str)
            elif 'title' in df.columns:
                df['text'] = df['title'].astype(str)

            # 필요한 컬럼만 선택
            if 'label' in df.columns:
                dfs.append(df[['text', 'label']])
            elif 'text' in df.columns:
                dfs.append(df[['text']])

    # dataset_3.csv (text만 존재)
    file_path = data_dir / "dataset_3.csv"
    df = load_single_dataset(file_path)

    if df is not None and 'text' in df.columns:
        if 'label' in df.columns:
            dfs.append(df[['text', 'label']])
        else:
            dfs.append(df[['text']])

    # 모든 데이터프레임 결합
    if not dfs:
        logger.warning(f"No datasets found in {data_dir}")
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)

    # 중복 제거 (E. 중복/누수 방지)
    initial_count = len(combined_df)
    if 'label' in combined_df.columns:
        # text 기준 중복 제거
        combined_df = combined_df.drop_duplicates(
            subset=['text'],
            keep='first'
        )

        # 유사한 텍스트 제거 (간단한 버전: 정규화 후 중복 제거)
        # 텍스트를 소문자로 변환하고 공백 정규화
        combined_df['text_normalized'] = combined_df['text'].astype(
            str).str.lower().str.strip()
        combined_df['text_normalized'] = combined_df['text_normalized'].str.replace(
            r'\s+', ' ', regex=True
        )

        # 정규화된 텍스트 기준으로도 중복 제거
        combined_df = combined_df.drop_duplicates(
            subset=['text_normalized'],
            keep='first'
        )

        # 정규화 컬럼 제거
        combined_df = combined_df.drop(
            columns=['text_normalized'], errors='ignore')

    final_count = len(combined_df)
    removed_count = initial_count - final_count

    if removed_count > 0:
        logger.info(
            f"Removed {removed_count} duplicate/similar samples "
            f"({removed_count/initial_count*100:.1f}%)"
        )

    return combined_df


def combine_title_text(title: Optional[str], text: str, sep: str = " [SEP] ") -> str:
    """
    title과 text를 결합 (A. 입력 스키마 호환성)

    Args:
        title: 제목 (없으면 None)
        text: 본문
        sep: 구분자 (기본값: " [SEP] ")

    Returns:
        결합된 텍스트 (title + [SEP] + text 또는 text만)
    """
    if title and title.strip():
        return f"{title.strip()}{sep}{text.strip()}"
    return text.strip()


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
    text = text.strip()
    text = ' '.join(text.split())  # 여러 공백을 하나로

    return text


def normalize_label(label: any) -> int:
    """
    라벨을 정수로 정규화 (Real=0, Fake=1)

    Args:
        label: 원본 라벨

    Returns:
        정규화된 라벨 (0 또는 1)
    """
    if pd.isna(label):
        return 0

    if isinstance(label, str):
        label_lower = label.lower().strip()
        if label_lower in ['fake', '1']:
            return 1  # Fake = 1 (positive)
        else:
            return 0  # Real = 0 (negative)

    # 숫자인 경우
    try:
        label_int = int(float(str(label)))
        return 1 if label_int == 1 else 0
    except (ValueError, TypeError):
        return 0


def split_train_val(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    학습/검증 데이터 분할

    Args:
        df: 전체 데이터프레임
        val_ratio: 검증 데이터 비율
        random_state: 랜덤 시드

    Returns:
        (train_df, val_df)

    Raises:
        ValueError: label 컬럼이 없는 경우
    """
    if 'label' not in df.columns:
        raise ValueError("DataFrame에 'label' 컬럼이 없습니다.")

    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=df['label']  # 라벨 비율 유지
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def prepare_data_for_training(
    df: pd.DataFrame
) -> Tuple[List[str], Optional[List[int]]]:
    """
    학습을 위한 데이터 준비

    Args:
        df: 데이터프레임

    Returns:
        (texts, labels) - labels가 None일 수 있음
    """
    texts = df['text'].apply(preprocess_text).tolist()

    if 'label' not in df.columns:
        return texts, None

    # 라벨을 정수로 변환 (Real=0, Fake=1)
    labels = df['label'].apply(normalize_label).tolist()

    return texts, labels
